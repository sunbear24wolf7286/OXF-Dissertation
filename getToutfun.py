import math
import pandas as pd
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from evalwitherrfun import eval_with_err
from getvaluefun import get_value

def get_Tout(P_info, T_in_info, duty_info, m_CO2_info, file_name, show_steps=False, tol=1e-3):
    from functools import lru_cache
    _orig_read = pd.read_excel
    pd.read_excel = lru_cache(maxsize=1)(pd.read_excel)
    try:
        P_val, P_err = P_info
        T_in, T_in_err = T_in_info
        Q_tgt, Q_tgt_err = duty_info
        m_CO2, m_err = m_CO2_info
        dt = 1.0 
        cum_Q, cum_Q_err = 0.0, 0.0
        T_cur, T_cur_err = T_in, T_in_err
        Q_low = Q_tgt - Q_tgt_err
        Q_high = Q_tgt + Q_tgt_err
        T_list, T_err_list = [], []
        Cp_list, Cp_err_list = [], []
        duty_list = []
        m_sym, Cp0_sym, Cp1_sym, Cp2_sym, dT_sym = sp.symbols('m_CO2 Cp0 Cp1 Cp2 dT')
        Q_simpson_expr = m_sym * dT_sym * (Cp0_sym + 4*Cp1_sym + Cp2_sym) / 6
        T_sym, dT2_sym = sp.symbols('T_sym dT_sym')
        expr_T = T_sym + dT2_sym

        while cum_Q < Q_low:
            Cp0, Cp0_err = get_value(P_info, ['T', T_cur, T_cur_err], 'Cp', file_name)
            dt2 = dt / 2.0
            T_mid, T_mid_err = eval_with_err(expr_T,
                                             [T_sym, dT2_sym],
                                             [T_cur, dt2],
                                             [T_cur_err, 0])
            Cp1, Cp1_err = get_value(P_info, ['T', T_mid, T_mid_err], 'Cp', file_name)
            Q_est = m_CO2 * Cp0 * dt
            if cum_Q + Q_est > Q_high and Cp0 != 0:
                Q_remain = Q_tgt - cum_Q
                dt = float(Q_remain / (m_CO2 * Cp0))
                for _ in range(10):
                    dt2 = dt / 2.0
                    T_mid, T_mid_err = eval_with_err(expr_T,
                                                     [T_sym, dT2_sym],
                                                     [T_cur, dt2],
                                                     [T_cur_err, 0])
                    Cp1, Cp1_err = get_value(P_info, ['T', T_mid, T_mid_err], 'Cp', file_name)
                    T_full, T_full_err = eval_with_err(expr_T,
                                                       [T_sym, dT2_sym],
                                                       [T_cur, dt],
                                                       [T_cur_err, 0])
                    Cp2, Cp2_err = get_value(P_info, ['T', T_full, T_full_err], 'Cp', file_name)
                    Q_step, Q_step_err = eval_with_err(
                        Q_simpson_expr,
                        [m_sym, Cp0_sym, Cp1_sym, Cp2_sym, dT_sym],
                        [m_CO2, Cp0, Cp1, Cp2, dt],
                        [m_err, Cp0_err, Cp1_err, Cp2_err, 0]
                    )
                    if abs(Q_step - Q_remain) < tol:
                        break
                    dt *= float(Q_remain / Q_step)
            T_full, T_full_err = eval_with_err(expr_T,
                                               [T_sym, dT2_sym],
                                               [T_cur, dt],
                                               [T_cur_err, 0])
            Cp2, Cp2_err = get_value(P_info, ['T', T_full, T_full_err], 'Cp', file_name)
            Q_step, Q_step_err = eval_with_err(
                Q_simpson_expr,
                [m_sym, Cp0_sym, Cp1_sym, Cp2_sym, dT_sym],
                [m_CO2, Cp0, Cp1, Cp2, dt],
                [m_err, Cp0_err, Cp1_err, Cp2_err, 0]
            )
            cum_Q, cum_Q_err = eval_with_err(
                sp.symbols('Q'),
                [sp.symbols('Q')],
                [cum_Q + Q_step],
                [cum_Q_err, Q_step_err]
            )
            T_cur, T_cur_err = T_full, T_full_err
            T_list.append(T_cur)
            T_err_list.append(T_cur_err)
            Cp_list.append(Cp2)
            Cp_err_list.append(Cp2_err)
            duty_list.append(cum_Q)

            if show_steps:
                print(f"T={T_cur:.3f}±{T_cur_err:.3f} K, Q_accum={cum_Q:.6f}±{cum_Q_err:.6f}")
            if cum_Q >= Q_low:
                break
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))
        ax1.errorbar(T_list, Cp_list,
                     xerr=T_err_list, yerr=Cp_err_list,
                     fmt='o', markersize=3, capsize=2)
        ax1.set_xlabel('T (K)')
        ax1.set_ylabel('Cp')
        ax1.set_title('Cp vs. T')

        ax2.errorbar(T_list, duty_list,
                     xerr=T_err_list,
                     fmt='o', markersize=3, capsize=2)
        ax2.set_xlabel('T (K)')
        ax2.set_ylabel('Accumulated Duty')
        ax2.set_title('Accumulated Duty vs. T')

        plt.tight_layout()
        plt.show()

        return T_cur, T_cur_err

    finally:
        pd.read_excel = _orig_read