import pandas as pd  
import numpy as np 

def secant_solve_brine_pref(
    rho0,         # kg/m³ surface brine density
    b_brine,      # 1/MPa brine compressibility
    z,            # m depth
    Psurf=0.101325,  # MPa surface (atmospheric) pressure
    g=9.81,       # m/s²
    tol=1e-6,     # MPa convergence tolerance
    max_iter=20
):

    # initial guess: hydrostatic + Psurf
    P0 = Psurf + rho0 * g * z / 1e6     # MPa
    P1 = P0 * 1.05                       # MPa

    def f(P):
        # density at pressure P [kg/m³]
        rho = rho0 * np.exp(b_brine * (P - Psurf))
        # residual = P - [Psurf + rho*g*z/1e6]
        return P - (Psurf + rho * g * z / 1e6)

    for _ in range(max_iter):
        f0, f1 = f(P0), f(P1)
        if abs(f1) < tol:
            break
        denom = f1 - f0
        if abs(denom) < 1e-12:
            break
        P2 = P1 - f1 * (P1 - P0) / denom
        P0, P1 = P1, P2

    Pref = P1
    rho_br = rho0 * np.exp(b_brine * (Pref - Psurf))
    return Pref, rho_br
