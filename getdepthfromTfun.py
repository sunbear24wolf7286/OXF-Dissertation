import math
import numpy as np
import pandas as pd
import sympy as sp
from evalwitherrfun import eval_with_err

def get_depth_from_T(T, Terr, Ts, G):
    T_sym, Ts_sym, G_sym = sp.symbols('T_sym Ts_sym G_sym')
    expr = (Ts_sym - T_sym) / G_sym
    Z, Zerr = eval_with_err(
        expr,
        [Ts_sym, T_sym, G_sym],
        [Ts, T, G],
        [0.0, Terr, 0.0]
    )
    Z = abs(Z); Zerr = abs(Zerr)
    print(f"Depth = {Z:.1f} +/- {Zerr:.1f} m for Temperature = {T:.3f} +/- {Terr:.3f} K")
    return Z, Zerr