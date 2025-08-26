import math
import pandas as pd
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from evalwitherrfun import eval_with_err
from getvaluefun import get_value
    
def get_h_from_eta(P_out_info, h_in_info, s_in_info, eta, equip, file_name):
    import pandas as _pd
    from functools import lru_cache
    _orig = _pd.read_excel
    _pd.read_excel = lru_cache(maxsize=1)(_pd.read_excel)
    try:
        P_out, P_out_err = P_out_info
        h_in, h_in_err = h_in_info
        s_in, s_in_err = s_in_info

        h_iso, h_iso_err = get_value([P_out,P_out_err],['s',s_in,s_in_err],'h',file_name)
        h_sym,h_out_sym,η_sym = sp.symbols('h_sym h_out_sym eta_sym')
        if equip=='Compressor':
            expr = h_sym + (h_out_sym-h_sym)/η_sym
        else:
            expr = h_sym - η_sym*(h_sym-h_out_sym)
        vals=[h_in,h_iso,eta]; errs=[h_in_err,h_iso_err,0]
        h_out,h_out_err = eval_with_err(expr,[h_sym,h_out_sym,η_sym],vals,errs)

        lbl = equip
        print(f"{lbl}: h_in={h_in:.3f}±{h_in_err:.3f}, h_iso={h_iso:.3f}±{h_iso_err:.3f}, eta={eta:.3f} → h_out={h_out:.3f}±{h_out_err:.3f}")
        plt.figure(figsize=(3,3))
        plt.errorbar(s_in,h_in,xerr=s_in_err,yerr=h_in_err,fmt='ko',label='Inlet')
        plt.errorbar(s_in,h_iso,xerr=s_in_err,yerr=h_iso_err,fmt='bo',label='Isentropic')
        plt.errorbar(s_in,h_out,xerr=s_in_err,yerr=h_out_err,fmt='ro',label='Actual')
        plt.xlabel('s');plt.ylabel('h');plt.legend(fontsize=6);plt.tight_layout();plt.show()
        return h_out,h_out_err
    finally:
        _pd.read_excel = _orig