import math
import pandas as pd
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from evalwitherrfun import eval_with_err
from getvaluefun import get_value

def get_Duty(P_info, T_in_info, T_target_info, m_CO2_info, file_name, show_steps=False, tol=1e-3):
    import pandas as _pd
    from functools import lru_cache
    _orig = _pd.read_excel
    _pd.read_excel = lru_cache(maxsize=1)(_pd.read_excel)
    try:
        P_val, P_err = P_info
        T_in, T_in_err = T_in_info
        T_tgt, T_tgt_err = T_target_info
        m_CO2, m_err = m_CO2_info

        dt = (T_tgt - T_in) / 100.0
        sign = 1 if T_tgt >= T_in else -1
        T_low, T_high = T_tgt - T_tgt_err, T_tgt + T_tgt_err

        cum_Q, cum_Q_err = 0.0, 0.0
        T_cur, T_cur_err = T_in, T_in_err

        T_list, T_err_list = [], []
        Cp_list, Cp_err_list = [], []
        Q_list, Q_err_list = [], []

        m_sym, Cp0_sym, Cp1_sym, Cp2_sym, dT_sym = sp.symbols('m_CO2 Cp0 Cp1 Cp2 dT')
        Q_simpson_expr = m_sym * dT_sym * (Cp0_sym + 4*Cp1_sym + Cp2_sym) / 6
        T_sym, dT2_sym = sp.symbols('T_sym dT_sym')
        expr_T = T_sym + dT2_sym

        while sign * (T_tgt - T_cur) > T_tgt_err:
            if dt > 0 and T_cur + dt > T_high:
                dt = T_high - T_cur
            if dt < 0 and T_cur + dt < T_low:
                dt = T_low - T_cur
            import io, contextlib
            with io.StringIO() as buf, contextlib.redirect_stdout(buf):
                Cp0, Cp0_err = get_value(P_info, ['T', T_cur, T_cur_err], 'Cp', file_name)
            dt2 = dt / 2.0
            T_mid, T_mid_err = eval_with_err(expr_T, [T_sym, dT2_sym], [T_cur, dt2], [T_cur_err, 0])
            with io.StringIO() as buf, contextlib.redirect_stdout(buf):
                Cp1, Cp1_err = get_value(P_info, ['T', T_mid, T_mid_err], 'Cp', file_name)
            T_full, T_full_err = eval_with_err(expr_T, [T_sym, dT2_sym], [T_cur, dt], [T_cur_err, 0])
            with io.StringIO() as buf, contextlib.redirect_stdout(buf):
                Cp2, Cp2_err = get_value(P_info, ['T', T_full, T_full_err], 'Cp', file_name)
            Q_simpson, Q_simpson_err = eval_with_err(
                Q_simpson_expr,
                [m_sym, Cp0_sym, Cp1_sym, Cp2_sym, dT_sym],
                [m_CO2, Cp0, Cp1, Cp2, dt],
                [m_err, Cp0_err, Cp1_err, Cp2_err, 0]
            )
            Q_used, Q_err_used = Q_simpson, Q_simpson_err
            if show_steps:
                print(f"T={T_cur:.3f}±{T_cur_err:.3f}, dt={dt:.3f}, Q_step={Q_used:.6f}, cum={cum_Q:.6f}")
            cum_Q, cum_Q_err = eval_with_err(
                sp.symbols('Q'), [sp.symbols('Q')], [cum_Q + Q_used], [cum_Q_err, Q_err_used]
            )
            T_cur, T_cur_err = T_full, T_full_err
            T_list.append(T_cur)
            T_err_list.append(T_cur_err)
            Cp_list.append(Cp2)
            Cp_err_list.append(Cp2_err)
            Q_list.append(cum_Q)
            Q_err_list.append(cum_Q_err)
        if T_cur < T_low: T_cur = T_low
        if T_cur > T_high: T_cur = T_high

        print(f"Final T={T_cur:.3f}±{T_cur_err:.3f}")
        print(f"Total Q={cum_Q:.6f} ± {cum_Q_err:.6f}")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))
        ax1.errorbar(T_list, Cp_list, xerr=T_err_list, yerr=Cp_err_list, fmt='o', capsize=2)
        ax1.set_xlabel('T'); ax1.set_ylabel('Cp'); ax1.set_title('Cp vs T')
        ax2.errorbar(T_list, Q_list, xerr=T_err_list, yerr=Q_err_list, fmt='o', capsize=2)
        ax2.set_xlabel('T'); ax2.set_ylabel('Cum Q'); ax2.set_title('Duty vs T')
        plt.tight_layout(); plt.show()

        return cum_Q, cum_Q_err
    finally:
        _pd.read_excel = _orig