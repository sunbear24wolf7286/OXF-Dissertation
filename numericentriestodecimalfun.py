import numpy as np
import pandas as pd

def convert_numeric_entries_to_decimal(x):
    if pd.isna(x):
        return x
    if isinstance(x, (float, np.floating)):
        return "{:f}".format(x)
    return x