import pandas as pd
import numpy as np

def get_vis(P, T, near_critical_point, filename):
    sheet_name = 'Sheet2' if str(near_critical_point).strip().upper() == 'Y' else 'Sheet1'
    df = pd.read_excel(f"{filename}.xlsx", sheet_name=sheet_name)
    df = df.sort_values('P').reset_index(drop=True)
    temp_cols = list(df.columns[1:])
    temp_map = {float(col): col for col in temp_cols}
    P_values = df['P'].values
    if not (P_values.min() <= P <= P_values.max()):
        print(f"Error: Pressure {P} outside data range [{P_values.min()}, {P_values.max()}]")
        return None
    P_exact = any(np.isclose(P_values, P))
    def find_bounds(val, array):
        idx = np.searchsorted(array, val)
        if idx < len(array) and np.isclose(array[idx], val):
            return val, val
        if idx == 0 or idx >= len(array):
            return None, None
        return array[idx-1], array[idx]
    def get_T_values_for_P(p):
        row = df[np.isclose(df['P'], p)].iloc[0]
        temps = [float(c) for c in temp_cols if pd.notnull(row[c])]
        return np.array(sorted(temps))
    if P_exact:
        row = df[np.isclose(df['P'], P)].iloc[0]
        T_vals = get_T_values_for_P(P)
        lower_T, upper_T = find_bounds(T, T_vals)
        if lower_T is None:
            print(f"Error: Temperature {T} outside available range for P={P}: [{T_vals.min()}, {T_vals.max()}]")
            return None
        col_low = temp_map[lower_T]
        col_high = temp_map[upper_T]
        v_low = row[col_low]
        v_high = row[col_high]
        if lower_T == upper_T:
            print(f"Exact lookup: P={P}, T={T} (Sheet: {sheet_name}), viscosity={v_low}")
            return v_low
        vis = np.interp(T, [lower_T, upper_T], [v_low, v_high])
        print(f"Interpolated in T at exact P={P} (Sheet: {sheet_name})")
        print(f"  T bounds and viscosities: {lower_T}→{v_low}, {upper_T}→{v_high}")
        return vis
    else:
        lower_P, upper_P = find_bounds(P, P_values)
        if lower_P is None:
            print(f"Error: Pressure {P} cannot be bounded for interpolation")
            return None
        row_low = df[np.isclose(df['P'], lower_P)].iloc[0]
        row_high = df[np.isclose(df['P'], upper_P)].iloc[0]

        T_lowP = get_T_values_for_P(lower_P)
        T_highP = get_T_values_for_P(upper_P)
        common_T = np.intersect1d(T_lowP, T_highP)
        lower_T, upper_T = find_bounds(T, common_T)
        if lower_T is None:
            print(f"Error: Temperature {T} not in common T range for P bounds {lower_P}, {upper_P}")
            print(f"  Available at {lower_P}: [{T_lowP.min()}, {T_lowP.max()}]")
            print(f"  Available at {upper_P}: [{T_highP.min()}, {T_highP.max()}]")
            return None
        col_l = temp_map[lower_T]
        col_u = temp_map[upper_T]
        v_ll = row_low[col_l]
        v_lu = row_low[col_u]
        v_ul = row_high[col_l]
        v_uu = row_high[col_u]

        vis_low = np.interp(T, [lower_T, upper_T], [v_ll, v_lu])
        vis_high = np.interp(T, [lower_T, upper_T], [v_ul, v_uu])
        vis = np.interp(P, [lower_P, upper_P], [vis_low, vis_high])

        print(f"Interpolated in P and T (Sheet: {sheet_name})")
        print(f"  P bounds and viscosities at T bounds:")
        print(f"    P={lower_P}: T={lower_T}→{v_ll}, T={upper_T}→{v_lu}")
        print(f"    P={upper_P}: T={lower_T}→{v_ul}, T={upper_T}→{v_uu}")
        print(f"  Final interpolation at P={P}, T={T}: {vis}")
        return vis