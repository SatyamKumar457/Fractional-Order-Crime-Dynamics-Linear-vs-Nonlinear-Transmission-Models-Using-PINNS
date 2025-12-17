# Fractional-Order Crime Dynamics  
## Linear vs Nonlinear Transmission Models using Physics-Informed Neural Networks (PINNs)

---

## Overview

This repository presents a **fractional-order mathematical modeling framework** for analyzing **crime dynamics**, with a focused comparison between **linear** and **nonlinear transmission mechanisms**.  
The system is solved using **Physics-Informed Neural Networks (PINNs)**, ensuring that the learned solutions strictly obey the governing **fractional differential equations**.

Unlike classical integer-order models, this work incorporates **memory effects** via fractional derivatives, allowing past crime behavior to influence present dynamics in a mathematically consistent way.

This project sits at the intersection of:
- Fractional calculus  
- Dynamical systems  
- Physics-informed machine learning  
- Socio-economic modeling  

---

## Key Contributions

- **Fractional-Order Crime Modeling**  
  Introduces memory-aware crime dynamics using fractional derivatives.

- **Linear vs Nonlinear Transmission Comparison**  
  Direct analytical and numerical comparison of different transmission mechanisms.

- **Physics-Informed Neural Networks (PINNs)**  
  Neural networks constrained by the underlying physics instead of pure data fitting.

- **Stability and Behavioral Analysis**  
  Shows how fractional order and transmission structure affect long-term dynamics.

---

## Mathematical Formulation

The general fractional-order system is defined as:

\[
D_t^\alpha X(t) = F(X(t), \theta), \quad 0 < \alpha \leq 1
\]

Where:
- \( D_t^\alpha \) is the fractional derivative  
- \( X(t) \) represents crime-related state variables  
- \( \theta \) denotes model parameters  

Both **linear** and **nonlinear** transmission terms are implemented and evaluated.

---

## Why Fractional Order?

Traditional integer-order models assume:
- No historical dependency  
- Instantaneous system response  

These assumptions are unrealistic for crime dynamics.

Fractional-order models:
- Capture long-term memory effects  
- Reflect delayed social and behavioral responses  
- Provide more realistic persistence and decay patterns  

Ignoring memory simplifies the math, not the reality.

---

## Methodology

1. **Model Construction**
   - Define fractional-order governing equations
   - Implement linear and nonlinear transmission mechanisms

2. **PINN Design**
   - Neural networks approximate state variables
   - Loss function enforces:
     - Fractional differential equations
     - Initial and boundary conditions
     - Physical consistency

3. **Training and Analysis**
   - Train PINNs for different fractional orders
   - Compare system behavior across models

---


## Technologies Used

- Python  
- Physics-Informed Neural Networks (PINNs)  
- Fractional Calculus  
- NumPy, SciPy  
- TensorFlow / PyTorch  

---

## Results and Insights

- Fractional-order models exhibit **fundamentally different dynamics** than integer-order systems.
- Nonlinear transmission leads to **richer and more realistic behavior**, especially under strong memory effects.
- PINNs successfully learn system dynamics **without relying on traditional numerical solvers**.

This is physics-constrained learning, not curve fitting.

---

## Applications

- Crime and socio-economic modeling  
- Fractional dynamical systems  
- PINNs for non-classical differential equations  
- Research in applied mathematics and AI-driven modeling  

---

## How to Run

```bash
pip install -r requirements.txt
python main.py

