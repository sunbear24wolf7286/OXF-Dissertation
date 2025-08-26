import math
import pandas as pd
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

def get_solved_state(stream_number, df, var):
    mask = (df['Stream_#'] == stream_number)
    if mask.any():
        value = df.loc[mask, var].iat[0]
        error = df.loc[mask, f"{var}err"].iat[0]
    else:
        value = np.nan
        error = np.nan

    return value, error

