import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import trapz

from evalwitherrfun import eval_with_err
from getvaluefun import get_value

def get_Duty_new(P_info,
                 T_in_info,
                 T_target_info,
                 m_CO2_info,
                 file_name,
                 n_points: int = 250,
                 show_plots: bool = True):
    P_val, P_err       = P_info
    T_in, T_in_err     = T_in_info
    T_tgt, T_tgt_err   = T_target_info
    m_CO2, m_err       = m_CO2_info
    T_pts   = np.linspace(T_in, T_tgt, n_points)
    Cp_vals = np.zeros(n_points)
    Cp_errs = np.zeros(n_points)
    for i, T in enumerate(T_pts):
        cp, cp_err = get_value(P_info, ['T', T, T_in_err], 'Cp', file_name)
        Cp_vals[i] = cp
        Cp_errs[i] = cp_err
    Q1, Q3 = np.percentile(Cp_vals, [25, 75])
    IQR    = Q3 - Q1
    mask   = (Cp_vals >= Q1 - 1.5 * IQR) & (Cp_vals <= Q3 + 1.5 * IQR)

    T_clean   = T_pts[mask]
    Cp_clean  = Cp_vals[mask]
    Err_clean = Cp_errs[mask]

    if T_clean.size < 2:
        raise RuntimeError("Not enough Cp points after outlier removal.")

    area = trapz(y=Cp_clean, x=T_clean)

    Q = m_CO2 * area

    Q_err_mass = abs(Q) * (m_err / m_CO2)

    dT = np.diff(T_clean)
    w = np.zeros_like(Err_clean)
    w[1:-1] = (dT[:-1] + dT[1:]) / 2
    w[0]     = dT[0] / 2
    w[-1]    = dT[-1] / 2

    Q_err_cp = m_CO2 * np.sqrt(np.sum((w * Err_clean)**2))

    Q_err = np.sqrt(Q_err_mass**2 + Q_err_cp**2)

    if show_plots:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        ax1.scatter(T_pts,  Cp_vals,  s=8, alpha=0.3, label='raw Cp')
        ax1.scatter(T_clean, Cp_clean, s=8, alpha=0.8, label='inliers')
        ax1.set(xlabel='T', ylabel='Cp', title='Cp vs T')
        ax1.legend()

        cumQ = m_CO2 * np.concatenate([
            [0],
            np.cumsum((Cp_clean[:-1] + Cp_clean[1:]) / 2 * np.diff(T_clean))
        ])
        ax2.plot(T_clean, cumQ, '-o', markersize=3)
        ax2.set(xlabel='T', ylabel='Cum Q', title='Duty vs T')

        plt.tight_layout()
        plt.show()

    return Q, Q_err
