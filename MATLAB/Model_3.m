clc; clear; close all;

% ================================================================
% Fractional RK-2 Simulation for SCP Model (gamma-saturated)
% ================================================================

% ----------------------------
% Numerical parameters
% ----------------------------
h     = 0.01;           % time step
T     = 20;             % total time
N     = round(T/h);     % number of steps
t     = 0:h:T;          % time grid

eta = 0.830094;
gamma_eta = gamma(eta + 1);

% ----------------------------
% Trained model parameters
% ----------------------------
A0 = 0.060089;
alpha = 0.699929;
l = 0.090095;
delta0 = 0.069907;
delta1 = 0.029904;
lam = 1.600093;
gamma_ = 0.300013;

% ---- Fractional-powered parameters ----
A_eta      = A0^eta;
alpha_eta = alpha^eta;
l_eta     = l^eta;
delta0_eta = delta0^eta;
delta1_eta = delta1^eta;

% ----------------------------
% RHS: Modified SCP Model
% ----------------------------
dS = @(S,C,P) A_eta ...
              - (alpha_eta .* S .* C) ./ (1 + gamma_ .* C) ...
              + delta1_eta .* P ...
              - delta0_eta .* S;

dC = @(S,C)   (alpha_eta .* S .* C) ./ (1 + gamma_ .* C) ...
              - l_eta .* C ...
              - delta0_eta .* C;

dP = @(C,P)   l_eta .* C ...
              - (delta0_eta + delta1_eta) .* P;

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
% Fractional RK-2 Integration
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

    % ---- Update ----
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

title(sprintf('Fractional RK-2 SCP Model with \\gamma-Saturation (\\eta = %.2f)', eta), ...
      'FontSize',14,'FontWeight','bold');

legend({'S(t) – Susceptible', ...
        'C(t) – Criminally Active', ...
        'P(t) – Prison Population'}, ...
        'Location','best','FontSize',11);

xlim([0 T]);
ylim([0 max(1e-6, max([S C P],[],'all')*1.1)]);