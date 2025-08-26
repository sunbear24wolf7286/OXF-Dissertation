import math
import pandas as pd
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from evalwitherrfun import eval_with_err
from getvaluefun import get_value
from getDutyfun import get_Duty

from getsolvedstatefun import get_solved_state

def Q_fun(df, start_stream_num, end_stream_num, mCO2info, filename):
    Pstart, Pstarte = get_solved_state(start_stream_num, df, 'P')
    Tstart, Tstarte = get_solved_state(start_stream_num, df, 'T')
    Tend, Tende     = get_solved_state(end_stream_num,   df, 'T')
    return get_Duty([Pstart,    Pstarte], [Tstart,    Tstarte], [Tend,      Tende], mCO2info, filename, show_steps=False)