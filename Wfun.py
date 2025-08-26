import math
import pandas as pd
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from subtractfun import subtract_fun
from getsolvedstatefun import get_solved_state

def W_fun(df, start_stream_num, end_stream_num):
    h_start,     h_start_err = get_solved_state(start_stream_num, df, 'h')
    h_end,       h_end_err   = get_solved_state(end_stream_num,   df, 'h')
    return subtract_fun(h_end, h_end_err, h_start, h_start_err)
