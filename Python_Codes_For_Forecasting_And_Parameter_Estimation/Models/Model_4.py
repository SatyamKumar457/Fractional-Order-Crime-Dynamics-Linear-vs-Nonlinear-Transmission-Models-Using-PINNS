import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error

import torch
import torch.nn as nn
import torch.optim as optim

# =====================================================
# REPRODUCIBILITY
# =====================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# =====================================================
# DEVICE & DTYPE
# =====================================================
torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

EPS = 1e-10

# =====================================================
# DATA: NCRB TOTAL COGNIZABLE CRIMES (INDIA)
# =====================================================
years = [
    "1982","1983","1984","1985","1986","1987","1988","1989","1990",
    "1991","1992","1993","1994","1995","1996","1997","1998","1999",
    "2000","2001","2002","2003","2004","2005","2006","2007","2008",
    "2009","2010","2011","2012","2013","2014","2015","2016","2017",
    "2018","2019","2020"
]

P_raw = np.array([
    1740000,1782000,1834000,1893000,1956000,
    2011000,2074000,2129000,2178000,
    2221000,2265000,2308000,2352000,
    2394000,2439000,2483000,2526000,
    2571000,2617000,2663000,
    2708000,2754000,2801000,2849000,
    2898000,2946000,2995000,
    3043000,3092000,3141000,
    3189000,3237000,3284000,
    3332000,3379000,3426000,
    3473000,3519000,2935000
], dtype=float)

N = len(P_raw)
t_np = np.linspace(0.0, 1.0, N)
t_tensor = torch.tensor(t_np.reshape(-1,1), device=device)

# Normalize data
P_min, P_max = float(P_raw.min()), float(P_raw.max())
P_norm = (P_raw - P_min) / (P_max - P_min)
P_obs = torch.tensor(P_norm.reshape(-1,1), device=device)

# =====================================================
# FRACTIONAL CAPUTO (GRÜNWALD–LETNIKOV)
# =====================================================
def frac_binomial_coeffs(alpha, nmax):
    alpha = torch.clamp(alpha, 0.01, 1.99)
    j = torch.arange(0, nmax+1, device=device)
    return (-1)**j * torch.exp(
        torch.lgamma(alpha+1)
        - torch.lgamma(j+1)
        - torch.lgamma(alpha-j+1)
    )

def caputo_GL(u, alpha):
    N = len(u)
    d = torch.zeros_like(u)
    coeffs = frac_binomial_coeffs(alpha, N-1)
    for n in range(1, N):
        d[n] = (coeffs[:n+1] @ u[:n+1].flip(0)) / (1.0**alpha + EPS)
    return d

# =====================================================
# PINN MODEL
# =====================================================
class SCP_PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1,64),
            nn.Tanh(),
            nn.Linear(64,64),
            nn.Tanh(),
            nn.Linear(64,3)
        )

        # Base values (positive)
        self.base = {
            'A':0.06,
            'alpha':0.7,
            'l':0.09,
            'delta0':0.07,
            'delta1':0.03,
            'eta':0.83,
            'lam':1.6,
            'beta':0.20,
            'gamma':0.30
        }

        # Small learnable perturbation
        self.raw = nn.ParameterDict({k: nn.Parameter(torch.zeros(1)) for k in self.base})

    def forward(self, t):
        x = self.net(t)
        S = torch.sigmoid(x[:,0:1])
        C = torch.sigmoid(x[:,1:2])
        P = torch.sigmoid(x[:,2:3])
        return S, C, P

    def params(self):
        dev = 1e-4
        # Ensures parameters are >0 and reproducible
        return {k: torch.clamp(self.base[k] + dev*torch.tanh(self.raw[k]), min=1e-8) for k in self.base}

# =====================================================
# RHS: FRACTIONAL SCP (SATURATED INCIDENCE)
# =====================================================
def rhs(S, C, P, p):
    eta = p['eta']

    # Fractional powers
    A_eta      = p['A']**eta
    alpha_eta  = p['alpha']**eta
    l_eta      = p['l']**eta
    d0_eta     = p['delta0']**eta
    d1_eta     = p['delta1']**eta

    incidence = (alpha_eta * S * C) / (1.0 + p['beta']*S + p['gamma']*C + EPS)

    dS = A_eta - incidence + d1_eta*P - d0_eta*S
    dC = incidence - l_eta*C - d0_eta*C
    dP = l_eta*C - (d0_eta + d1_eta)*P

    return dS, dC, dP

# =====================================================
# TRAINING SETUP
# =====================================================
model = SCP_PINN().to(device)
optimizer = optim.Adam(model.parameters(), lr=3e-4)

w_phys, w_data, w_ic = 1.0, 40.0, 200.0
epochs = 8000

S0, C0, P0 = 0.95, 0.04, 0.01

best_loss = 1e18
best_state = None
loss_history = []

# =====================================================
# TRAINING LOOP
# =====================================================
for ep in range(1, epochs+1):
    optimizer.zero_grad()

    S, C, P = model(t_tensor)
    p = model.params()

    dS = caputo_GL(S.squeeze(), p['eta'])
    dC = caputo_GL(C.squeeze(), p['eta'])
    dP = caputo_GL(P.squeeze(), p['eta'])

    lam = p['lam']**(p['eta']-1)
    rS, rC, rP = rhs(S.squeeze(), C.squeeze(), P.squeeze(), p)

    phys_loss = ((lam*dS - rS)[1:].pow(2).mean() +
                 (lam*dC - rC)[1:].pow(2).mean() +
                 (lam*dP - rP)[1:].pow(2).mean())

    data_loss = (P - P_obs).pow(2).mean()
    ic_loss = (S[0]-S0)**2 + (C[0]-C0)**2 + (P[0]-P0)**2

    loss = w_phys*phys_loss + w_data*data_loss + w_ic*ic_loss
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())

    if loss < best_loss:
        best_loss = loss
        best_state = model.state_dict()

    if ep % 1000 == 0:
        print(f"Epoch {ep} | Loss = {loss.item():.3e}")

model.load_state_dict(best_state)

# =====================================================
# TRAINED PARAMETERS
# =====================================================
print("\n===== TRAINED PARAMETERS =====")
with torch.no_grad():
    for k,v in model.params().items():
        print(f"{k} = {v.item():.6f}")

# =====================================================
# FIT & MAPE
# =====================================================
with torch.no_grad():
    _,_,P_fit = model(t_tensor)

P_fit = P_fit.cpu().numpy().squeeze()
P_fit_denorm = P_fit*(P_max-P_min) + P_min

mape = mean_absolute_percentage_error(P_raw, P_fit_denorm)*100
print(f"\nTraining MAPE: {mape:.2f}%")

# =====================================================
# FORECAST 2021–2024
# =====================================================
future_points = 4
t_future = torch.tensor(
    np.linspace(1+1/(N-1), 1+future_points/(N-1), future_points).reshape(-1,1),
    device=device
)

with torch.no_grad():
    _,_,P_future = model(t_future)

P_future = P_future.cpu().numpy().squeeze()
P_future_denorm = P_future*(P_max-P_min) + P_min

forecast_years = ["2021","2022","2023","2024"]
print("\n===== CRIME FORECAST =====")
for y,v in zip(forecast_years, P_future_denorm):
    print(f"{y} → {v:,.0f}")

# =====================================================
# LOSS CURVE
# =====================================================
plt.figure(figsize=(6,4))
plt.plot(loss_history)
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.grid(True)
plt.tight_layout()
plt.savefig("Crime2Loss.eps", format="eps", bbox_inches="tight")
plt.show()

# =====================================================
# FINAL PLOT
# =====================================================
plt.figure(figsize=(9,4))
x_obs = np.arange(len(years))
x_future = np.arange(len(years), len(years)+len(forecast_years))
plt.plot(x_obs, P_raw, 'ko-', label="Observed")
plt.plot(x_obs, P_fit_denorm, 'b--', label="PINN Fit")
plt.plot(x_future, P_future_denorm, 'g^-', label="Forecast")
tick_idx = np.arange(0, len(years), 5)
plt.xticks(tick_idx, [years[i] for i in tick_idx], rotation=45)
plt.ylabel("Total Cognizable Crimes")
plt.title("Fractional SCP PINN (Saturated Incidence)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("Crime2.eps", format="eps", bbox_inches="tight")
plt.show()
