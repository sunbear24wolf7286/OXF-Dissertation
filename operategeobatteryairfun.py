import io                                 
import contextlib                          
import pandas as pd                       
import numpy as np                        
from getvaluefun import get_value           
from secantsolvecompressibilityfun import secant_solve_compressibility_implicit  
from secantsolvbrinepref import secant_solve_brine_pref                

import pandas as pd
import numpy as np
from secantsolvecompressibilityfun import secant_solve_compressibility_implicit 
from secantsolvbrinepref import secant_solve_brine_pref                  

def operate_geobattery_air(
    Vrock_HS, phi, b_pore, b_brine,            # Vrock [m³], φ [–], b_* [1/MPa]
    z_HS,                                       # depth [m]
    P_HS_ss, rho_air_HS_ss,                     # steady-state pressure & air density [MPa], [kg/m³]
    rho_brine_0,                                # surface brine density [kg/m³]
    T_HS, thermo_file,                          # temperature [K], EOS file
    stable_power_requirement,                   # kW
    W_turbine_T, W_compressor_T,                # kWh/kg
    power_rating,                               # kW
    dt,                                         # hr
    price: np.ndarray, generation: np.ndarray,  # price [$/(kWh)], generation [kW]
    pressure_tolerance,                         # fraction
    pressure_buffer_fraction=0.25               # fraction
):
    # === Phase 0: initial steady-state ===
    # Compute brine reference pressure & density at HS depth
    Pref_HS, rho_br_HS_ss = secant_solve_brine_pref(
        rho_brine_0, b_brine, z_HS
    )  # returns Pref_HS [MPa], rho_br_HS_ss [kg/m³]

    # Pressure differential in HS
    deltaP0_HS = P_HS_ss - Pref_HS  # MPa

    # Initial pore & fluid volumes
    Vp0_HS = phi * Vrock_HS         # m³
    Vp_HS_ss = Vp0_HS * (1 + b_pore * deltaP0_HS)
    Vbr_HS_ss = Vp0_HS * (1 - b_brine * deltaP0_HS)
    Vair_HS_ss = Vp_HS_ss - Vbr_HS_ss

    # Volume fractions
    volfrac_air_HS_ss = Vair_HS_ss / Vp_HS_ss
    volfrac_br_HS_ss  = 1 - volfrac_air_HS_ss

    # Masses at steady state
    Mair_HS_ss = rho_air_HS_ss * Vair_HS_ss
    Mbr_HS_ss  = rho_br_HS_ss * Vbr_HS_ss

    # Mass fractions
    massfrac_air_HS_ss = Mair_HS_ss / (Mair_HS_ss + Mbr_HS_ss)
    massfrac_br_HS_ss  = 1 - massfrac_air_HS_ss

    df0 = pd.DataFrame({
        'z_HS': [z_HS],
        'Pref_HS': [Pref_HS],
        'rho_br_HS_ss': [rho_br_HS_ss],
        'deltaP0_HS': [deltaP0_HS],
        'Vp0_HS': [Vp0_HS],
        'Vp_HS_ss': [Vp_HS_ss],
        'Vbr_HS_ss': [Vbr_HS_ss],
        'Vair_HS_ss': [Vair_HS_ss],
        'volfrac_air_HS_ss': [volfrac_air_HS_ss],
        'volfrac_br_HS_ss': [volfrac_br_HS_ss],
        'massfrac_air_HS_ss': [massfrac_air_HS_ss],
        'massfrac_br_HS_ss': [massfrac_br_HS_ss],
        'Mair_HS_ss': [Mair_HS_ss],
        'Mbr_HS_ss': [Mbr_HS_ss]
    })

    # === Phase 1: dispatch ===
    Vp_HS = Vp_HS_ss
    rho_br_HS = rho_br_HS_ss * np.exp(b_brine * (P_HS_ss - Pref_HS))
    Vbr_HS    = Mbr_HS_ss   / rho_br_HS

    n = len(generation)
    demand = np.full(n, stable_power_requirement)
    b_eff = b_pore + b_brine

    # State variables
    P_HS = P_HS_ss
    rho_air_HS = rho_air_HS_ss
    Mair_HS = Mair_HS_ss
    Mbr_HS = Mbr_HS_ss

    records = []
    for t in range(n):
        P_HS_prev = P_HS
        Vp_HS_prev = Vp_HS
        Vbr_HS_prev = Vbr_HS

        gen = generation[t]
        dem = demand[t]
        E_res = (gen - dem) * dt
        mode = 'charge' if E_res > 0 else 'discharge' if E_res < 0 else 'standby'

        ekg_chg = W_compressor_T * dt
        ekg_dis = W_turbine_T * dt
        if mode == 'charge':
            m_req = E_res / ekg_chg
            m_pow = power_rating * dt / ekg_chg
        elif mode == 'discharge':
            m_req = -E_res / ekg_dis
            m_pow = power_rating * dt / ekg_dis
        else:
            m_req = m_pow = 0.0

        # Pressure bounds
        Pmin_HS = P_HS_ss * (1 - pressure_tolerance)
        Pmax_HS = P_HS_ss * (1 + pressure_tolerance)
        head_HS = max(0.0,
              (Pmax_HS - P_HS) if mode == 'charge'
              else (P_HS - Pmin_HS))

        mlim_HS = head_HS * rho_air_HS * Vp_HS_prev * b_eff
        m_constraint = min(m_req, m_pow, pressure_buffer_fraction * mlim_HS)
        constraint = (
            'residual' if abs(m_constraint - m_req) < 1e-12 else
            'pressure' if abs(m_constraint - pressure_buffer_fraction * mlim_HS) < 1e-12 else
            'power'
        )
        dM_HS = m_constraint if mode == 'charge' else (-m_constraint if mode == 'discharge' else 0.0)

        # 1) implicit solve
        P_HS, rho_air_HS, Vp_HS = secant_solve_compressibility_implicit(
            P_HS_prev, dM_HS, Vp_HS_prev, b_eff, T_HS, thermo_file,
            P_HS_prev * 0.99, P_HS_prev * 1.01
        )

        # 2) update CO₂ mass
        Mair_HS += dM_HS

        # 3) brine update
        rho_br_HS = rho_br_HS_ss * np.exp(b_brine * (P_HS - Pref_HS))
        Vbr_HS = Mbr_HS / rho_br_HS

        # 4) diagnostics
        deltaP_HS = P_HS - P_HS_prev
        pore_gain_HS = Vp_HS - Vp_HS_prev
        extra_cap_HS = pore_gain_HS / Vp_HS_prev
        brine_shrink_HS = Vbr_HS_prev - Vbr_HS
        deltaV_HS_br = Vbr_HS - Vbr_HS_prev

        # 5) recompute CO₂ volumes & fractions
        Vair_HS = Mair_HS / rho_air_HS
        volfrac_air_HS = Vair_HS / Vp_HS
        volfrac_br_HS = 1 - volfrac_air_HS
        massfrac_air_HS = Mair_HS / (Mair_HS + Mbr_HS)

        # Energy flows
        charge = (
            m_constraint * ekg_chg if mode == 'charge' else
            -m_constraint * ekg_dis if mode == 'discharge' else
            0.0
        )
        if mode == 'charge':
            sold = gen * dt + (E_res - charge)
            bought = 0.0
        elif mode == 'standby':
            sold = gen * dt
            bought = 0.0
        else:
            sold = gen * dt + (-charge)
            bought = max(0.0, dem * dt - sold)

        revenue = price[t] * sold
        cost = price[t] * bought
        cashflow = revenue - cost
        balance = (records[-1]['balance'] + cashflow) if records else cashflow

        deltaP_HS_QSS = P_HS - P_HS_ss
        errorP_HS = deltaP_HS_QSS / P_HS_ss * 100

        records.append({
            'mode': mode,
            'residual': E_res,
            'charge': charge,
            'charge_mass_air': m_constraint,
            'pressure_constraint_mass_air': pressure_buffer_fraction * mlim_HS,
            'power_constraint': power_rating,
            'deltaM_air_HS': dM_HS,
            'M_air_HS': Mair_HS,
            'M_br_HS': Mbr_HS,
            'pore_gain_HS': pore_gain_HS,
            'extra_cap_HS': extra_cap_HS,
            'brine_shrink_HS': brine_shrink_HS,
            'Vp_HS': Vp_HS,
            'V_HS_air': Vair_HS,
            'Vbr_HS': Vbr_HS,
            'volfrac_HS_air': volfrac_air_HS,
            'volfrac_HS_br': volfrac_br_HS,
            'massfrac_HS_air': massfrac_air_HS,
            'massfrac_HS_br': 1 - massfrac_air_HS,
            'deltaV_HS_br': deltaV_HS_br,
            'P_HS': P_HS,
            'diffP_HS': deltaP_HS,
            'QSS_P_drift_HS': deltaP_HS_QSS,
            'errorP_HS': errorP_HS,
            'purchase': bought,
            'sale': sold,
            'revenue': revenue,
            'expense': cost,
            'cashflow': cashflow,
            'balance': balance,
            'generation': gen,
            'demand': dem,
            'price': price[t],
            'rho_air_HS': rho_air_HS,
            'rho_br_HS': rho_br_HS,
            'b_eff': b_eff,
            'residual_mass': m_req,
            'constraint': constraint
        })

    df1 = pd.DataFrame(records)
    return df0, df1
