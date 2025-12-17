clc; clear; close all;

% ================================================================
% Fractional RK-2 Simulation for SCP Crime Model (Corrected)
% ================================================================

% ----------------------------
% Numerical parameters
% ----------------------------
h     = 0.01;           % time step (smaller for stability)
T     = 20;             % total time
N     = round(T/h);     % number of steps
t     = 0:h:T;          % time grid

eta = 0.830094;
gamma_eta = gamma(eta + 1);

% ----------------------------
% Trained model parameters
% ----------------------------
A0 = 0.060089;
alpha = 0.699931;
l = 0.090095;
delta0 = 0.069907;
delta1 = 0.029904;
lam = 1.600093;
% ----------------------------
% RHS: SCP Model
% ----------------------------
dS = @(S,C,P) A0 ...
              - alpha*S.*C ...
              + delta1*P ...
              - delta0*S;

dC = @(S,C)   alpha*S.*C ...
              - l*C ...
              - delta0*C;

dP = @(C,P)   l*C ...
              - (delta0 + delta1)*P;

% ----------------------------
% Initial conditions
% ----------------------------
S = zeros(1, N+1);
C = zeros(1, N+1);
P = zeros(1, N+1);

S(1) = 0.95;
C(1) = 0.04;
P(1) = 0.10;

% ----------------------------
% Fractional RK-2 Integration (Corrected)
% ----------------------------
for k = 1:N

    % ---- k1 ----
    k1S = dS(S(k), C(k), P(k));
    k1C = dC(S(k), C(k));
    k1P = dP(C(k), P(k));

    % ---- Midpoint ----
    S_mid = S(k) + 0.5 * (h^eta)/gamma_eta * k1S;
    C_mid = C(k) + 0.5 * (h^eta)/gamma_eta * k1C;
    P_mid = P(k) + 0.5 * (h^eta)/gamma_eta * k1P;

    % ---- k2 ----
    k2S = dS(S_mid, C_mid, P_mid);
    k2C = dC(S_mid, C_mid);
    k2P = dP(C_mid, P_mid);

    % ---- Update (NO lambda amplification) ----
    S(k+1) = S(k) + (h^eta)/gamma_eta * k2S;
    C(k+1) = C(k) + (h^eta)/gamma_eta * k2C;
    P(k+1) = P(k) + (h^eta)/gamma_eta * k2P;
end

% ----------------------------
% Plot results
% ----------------------------
figure('Color','w'); hold on; grid on;

plot(t, S, 'LineWidth',2, 'Color',[0 0.447 0.741]);
plot(t, C, 'LineWidth',2, 'Color',[0.850 0.325 0.098]);
plot(t, P, 'LineWidth',2, 'Color',[0.929 0.694 0.125]);

xlabel('Time (t)','FontSize',12,'FontWeight','bold');
ylabel('State Variables','FontSize',12,'FontWeight','bold');

title(sprintf('Fractional RK-2 SCP Model (\\eta = %.3f)', eta), ...
      'FontSize',14,'FontWeight','bold');

legend({'S(t) – Susceptible', ...
        'C(t) – Criminally Active', ...
        'P(t) – Prison Population'}, ...
        'Location','best','FontSize',11);

xlim([0 T]);
ylim([0 max([S C P],[],'all')*1.1]);