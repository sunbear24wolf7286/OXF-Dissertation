import pandas as pd

def select_value_columns(df: pd.DataFrame) -> pd.DataFrame:
    mask = ~df.columns.str.contains('err', case=False)
    return df.loc[:, mask]
