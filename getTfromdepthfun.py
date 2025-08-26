import math
import numpy as np
import pandas as pd
import sympy as sp
from evalwitherrfun import eval_with_err

def get_T_from_depth(z, zerr, Ts, G):
    z_sym, Ts_sym, G_sym = sp.symbols('z_sym Ts_sym G_sym')
    expr = Ts_sym + G_sym * z_sym
    T, Terr = eval_with_err(expr,
                              [Ts_sym, G_sym, z_sym],
                              [Ts, G, z],
                              [0.0, 0.0, zerr])
    print(f"Temperature = {T:.3f} +/- {Terr:.3f} K at depth {z:.1f} +/- {zerr:.1f} m")
    return T, Terr
