import math
import pandas as pd
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from evalwitherrfun import eval_with_err
from getvaluefun import get_value

def get_remaining_thermo_vars(idx, df, filename):
    thermo = ['P','T','e','h','s','v','p','Cv','Cp']
    row = df.loc[idx]
    d1, d2 = row['Definer_1'], row['Definer_2']
    v1, e1 = row[d1], row[f"{d1}err"]
    v2, e2 = row[d2], row[f"{d2}err"]
    for var in thermo:
        if var in (d1, d2):
            continue
        val, err = get_value(
            [v1, e1],
            [d2, v2, e2],
            var,
            filename
        )
        df.at[idx, var]      = val
        df.at[idx, f"{var}err"] = err
