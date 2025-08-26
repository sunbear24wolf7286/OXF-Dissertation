import math
import pandas as pd
import numpy as np
import sympy as sp
import sys 
from getvaluefun import get_value
import matplotlib.pyplot as plt

def find_P_bottom_secant(
    T_aquifer,
    z_aquifer,
    Ptop,          # in MPa
    g,             # in m/s²
    fric_rate,     # in MPa
    rho_top,       # density at top in kg/m³
    thermo_file,
    P0,       # first guess in MPa
    P1,       # second guess in MPa
    tol,
    max_iter
):

    P_iters = [P0, P1]

    def f(P):
        rho_bot, _ = get_value(
            [P, 0],
            ['T', T_aquifer, 0],
            'p',
            thermo_file
        )
        rho_avg = 0.5 * (rho_top + rho_bot)
        head_MPa = (rho_avg * g * z_aquifer) / 1e6
        P_calc = Ptop - fric_rate + head_MPa
        return P - P_calc

    f0 = f(P0)
    f1 = f(P1)

    for _ in range(1, max_iter+1):
        if abs(f1) < tol:
            break
        if f1 == f0:
            raise RuntimeError("Zero denominator in secant update")
            
        P2 = P1 - f1 * (P1 - P0) / (f1 - f0)
        P0, f0 = P1, f1
        P1, f1 = P2, f(P2)
        P_iters.append(P1)
    else:
        raise RuntimeError(
            f"Did not converge after {max_iter} iterations; last |f|={abs(f1):.2e}"
        )

    rho_bot_final, _ = get_value(
        [P1, 0],
        ['T', T_aquifer, 0],
        'p',
        thermo_file
    )
    rho_avg_final = 0.5 * (rho_top + rho_bot_final)
    head_final    = (rho_avg_final * g * z_aquifer) / 1e6
    P_calc_final  = Ptop - fric_rate + head_final
    error_final   = P1 - P_calc_final

    print("Final iteration results:")
    print(f"  P_guess     = {P1:.6f} MPa")
    print(f"  P_calc      = {P_calc_final:.6f} MPa")
    print(f"  Error       = {error_final:.2e} MPa")
    print(f"  rho_bottom  = {rho_bot_final:.6f} kg/m³")
    print(f"  rho_avg     = {rho_avg_final:.6f} kg/m³")

    fig = plt.figure(figsize=(3,3))
    ax = fig.add_subplot(1,1,1)
    ax.plot(range(len(P_iters)), P_iters, marker='o')
    ax.set_xlabel("Iteration")
    ax.set_ylabel("P_guess (MPa)")
    ax.set_title("Secant Iterations")
    plt.tight_layout()
    plt.show()

    return P1, rho_avg_final, rho_bot_final