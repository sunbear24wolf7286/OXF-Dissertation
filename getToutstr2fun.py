import math
import pandas as pd
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from evalwitherrfun import eval_with_err
from getvaluefun import get_value
from getDutyfun import get_Duty
from getToutfun import get_Tout

def get_Tout_str2(Pinfostr1, Tin_str1_info, Tout_str1_info, Pinfostr2, Tin_str2_info, mCO2_info, filename, show_steps):
    strm1 = get_Duty(Pinfostr1, Tin_str1_info, Tout_str1_info, mCO2_info, filename, show_steps)
    Q = strm1[0]
    Q_err = strm1[1]

    strm2 = get_Tout(Pinfostr2, Tin_str2_info, [-Q, Q_err], mCO2_info, filename, show_steps)
    Tout_str2 = strm2[0]
    Tout_str2_err = strm2[1]
    
    import io, contextlib
    with io.StringIO() as buf, contextlib.redirect_stdout(buf):
        Cp_str1_in, Cp_str1_in_err = get_value(Pinfostr1, ['T', Tin_str1_info[0], Tin_str1_info[1]], 'Cp', filename)
        Cp_str1_out, Cp_str1_out_err = get_value(Pinfostr1, ['T', Tout_str1_info[0], Tout_str1_info[1]], 'Cp', filename)
    Cp_str1avg_expr = (sp.symbols('Cp_str1_in') + sp.symbols('Cp_str1_out')) / 2
    Cp_str1avg, Cp_str1avg_err = eval_with_err(
        Cp_str1avg_expr,
        [sp.symbols('Cp_str1_in'), sp.symbols('Cp_str1_out')],
        [Cp_str1_in, Cp_str1_out],
        [Cp_str1_in_err, Cp_str1_out_err]
    )

    with io.StringIO() as buf, contextlib.redirect_stdout(buf):
        Cp_str2_in, Cp_str2_in_err = get_value(Pinfostr2, ['T', Tin_str2_info[0], Tin_str2_info[1]], 'Cp', filename)
        Cp_str2_out, Cp_str2_out_err = get_value(Pinfostr2, ['T', Tout_str2, Tout_str2_err], 'Cp', filename)
    Cp_str2avg_expr = (sp.symbols('Cp_str2_in') + sp.symbols('Cp_str2_out')) / 2
    Cp_str2avg, Cp_str2avg_err = eval_with_err(
        Cp_str2avg_expr,
        [sp.symbols('Cp_str2_in'), sp.symbols('Cp_str2_out')],
        [Cp_str2_in, Cp_str2_out],
        [Cp_str2_in_err, Cp_str2_out_err]
    )

    T_str1_in, T_str1_in_err = Tin_str1_info
    T_str1_out, T_str1_out_err = Tout_str1_info
    T_str2_in, T_str2_in_err = Tin_str2_info

    expr_triangle = sp.symbols('T_str2_in') - \
        (sp.symbols('Cp_str1avg')/sp.symbols('Cp_str2avg')) * \
        (sp.symbols('T_str1_out') - sp.symbols('T_str1_in'))
    Tout_str2approx, Tout_str2approx_err = eval_with_err(
        expr_triangle,
        [sp.symbols('T_str2_in'), sp.symbols('Cp_str1avg'), sp.symbols('Cp_str2avg'),
         sp.symbols('T_str1_out'), sp.symbols('T_str1_in')],
        [T_str2_in, Cp_str1avg, Cp_str2avg, T_str1_out, T_str1_in],
        [T_str2_in_err, Cp_str1avg_err, Cp_str2avg_err, T_str1_out_err, T_str1_in_err]
    )

    print(f"Tout stream 2 from numerical integration: {Tout_str2:.3f} ± {Tout_str2_err:.3f}, "
          f"from triangle approximation: {Tout_str2approx:.3f} ± {Tout_str2approx_err:.3f}")

    return Tout_str2, Tout_str2_err