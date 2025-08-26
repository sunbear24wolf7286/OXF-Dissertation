import math
import pandas as pd
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from evalwitherrfun import eval_with_err
    
def get_value(P_info, second_info, target_variable, file_name):
    xs, x1s, x2s, y1s, y2s = sp.symbols('xs x1s x2s y1s y2s')
    linear_interp_expr = y1s + (xs - x1s)*( (y2s - y1s)/(x2s - x1s) )

    P_value, P_err = P_info
    second_def_var, second_val, second_err = second_info

    var_to_col = {
        'T':'T_K','p':'p_kg_m3','h':'h_kJ_kg','s':'s_kJ_kg_K',
        'e':'e_kJ_kg','Cv':'Cv_kJ_kg_K','Cp':'Cp_kJ_kg_K','v':'v_m_s'
    }
    if target_variable not in var_to_col or second_def_var not in var_to_col:
        raise ValueError("Incorrect variable")

    target_col = var_to_col[target_variable]
    second_col = var_to_col[second_def_var]

    df = pd.read_excel(f"{file_name}.xlsx")
    summary = []

    def interp_second(dfP):
        exact = dfP[dfP[second_col]==second_val]
        if not exact.empty:
            summary.append(f"{second_def_var} exact: {second_val:.3f}")
            return float(exact.iloc[0][target_col]), 0.0
        lb = dfP[dfP[second_col]<=second_val]
        ub = dfP[dfP[second_col]>=second_val]
        if lb.empty or ub.empty:
            raise ValueError("Out of bounds")
        x1 = lb[second_col].max(); y1 = lb[lb[second_col]==x1].iloc[0][target_col]
        x2 = ub[second_col].min(); y2 = ub[ub[second_col]==x2].iloc[0][target_col]
        vals = [second_val, x1, x2, y1, y2]
        errs = [second_err,0,0,0,0]
        val, err = eval_with_err(linear_interp_expr,[xs,x1s,x2s,y1s,y2s],vals,errs)
        summary.append(f"interp {second_def_var}@{P_value:.3f}: {val:.3f}±{err:.3f}")
        return val, err

    # pressure exact?
    dfP = df[df['P_MPa']==P_value]
    if not dfP.empty:
        summary.insert(0,f"P exact: {P_value:.3f}")
        v,e = interp_second(dfP)
        print(*summary,sep="\n"); return v,e

    # else pressure interp
    dfL = df[df['P_MPa']<P_value]; dfU = df[df['P_MPa']>P_value]
    if dfL.empty or dfU.empty:
        raise ValueError("P out of bounds")
    PL = dfL['P_MPa'].max(); PU = dfU['P_MPa'].min()
    summary.insert(0,f"P interp bounds: {PL:.3f},{PU:.3f}")
    vL,eL = interp_second(df[df['P_MPa']==PL])
    vU,eU = interp_second(df[df['P_MPa']==PU])
    vals = [P_value,PL,PU,vL,vU]; errs=[P_err,0,0,eL,eU]
    v,e = eval_with_err(linear_interp_expr,[xs,x1s,x2s,y1s,y2s],vals,errs)
    summary.append(f"→ {v:.3f}±{e:.3f}")
    print(*summary,sep="\n"); return v,e