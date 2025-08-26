import io
import contextlib
import numpy as np
from getvaluefun import get_value

def secant_solve_compressibility_implicit(
    P0: float,            # previous pressure [MPa]
    delta_m: float,       # mass change [kg]
    Vp0: float,           # pore‐volume at last step [m³]
    b_total: float,       # compressibility [1/MPa]
    T_aq: float,          # temperature [K]
    thermo_file: str,     # EOS lookup file
    Pguess0: float,       # secant first guess [MPa]
    Pguess1: float,       # secant second guess [MPa]
    tol: float = 1e-6,
    max_iter: int = 20
) -> (float, float, float):


    def silent_get(*args, **kwargs):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            return get_value(*args, **kwargs)

    def residual(P):
        rho, _ = silent_get([P,0], ['T', T_aq,0], 'p', thermo_file)
        Vp = Vp0*(1 + b_total*(P - P0))
        # f(P) = P - (P0 + Δm/(rho*Vp*b_total))
        return P - (P0 + delta_m/(rho * Vp * b_total))

    Pm0, Pm1 = Pguess0, Pguess1
    f0, f1 = residual(Pm0), residual(Pm1)

    for _ in range(max_iter):
        if abs(f1) < tol:
            break
        denom = (f1 - f0)
        if abs(denom) < 1e-12:
            break
        Pm2 = Pm1 - f1*(Pm1 - Pm0)/denom
        Pm0, f0, Pm1, f1 = Pm1, f1, Pm2, residual(Pm2)

    P_new = Pm1
    rho_new, _ = silent_get([P_new,0], ['T', T_aq,0], 'p', thermo_file)
    Vp_new = Vp0*(1 + b_total*(P_new - P0))
    return P_new, rho_new, Vp_new
