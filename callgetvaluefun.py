import math
import numpy as np
import pandas as pd
import sympy as sp
from evalwitherrfun import eval_with_err
from getvaluefun import get_value

def call_get_value(P_info, second_info, target_var, file_name, lookup_summary_mode):
    if lookup_summary_mode:
        return get_value(P_info, second_info, target_var, file_name)
    else:
        import sys, io
        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            result = get_value(P_info, second_info, target_var, file_name)
        finally:
            sys.stdout = old
        return result