import io                                   
import contextlib                          
import pandas as pd                      
import numpy as np                         
from getvaluefun import get_value           
from secantsolvecompressibilityfun import secant_solve_compressibility_implicit  # CO₂ pressure solver, no units
from secantsolvbrinepref import secant_solve_brine_pref                # brine hydrostatic solver, no units

def operate_geobattery(
    Vrock_LS, Vrock_HS, phi, b_pore, b_brine,     # Vrock [m³], φ [–], b_* [1/MPa]
    z_LS, z_HS,                                   # depths [m]
    P_LS_ss, P_HS_ss,                             # steady pressures [MPa]
    rho_co2_LS_ss, rho_co2_HS_ss,                 # CO₂ density @ ss [kg/m³]
    rho_brine_0,                                  # surface brine density [kg/m³]
    T_LS, T_HS, thermo_file,                      # temperature [K], EOS file
    stable_power_requirement,                     # kW
    W_turbine_T, W_compressor_T,                  # kWh/kg
    power_rating,                                 # kW
    dt,                                           # hr
    price: np.ndarray, generation: np.ndarray,    # price [$/(kWh)], generation [kW]
    pressure_tolerance,                           # fraction
    pressure_buffer_fraction=0.25                 # fraction
):
    # === Phase 0: initial steady‑state ===
    # Psurf = 0.101325 MPa (atm) by default
    Pref_LS, rho_br_LS_ss = secant_solve_brine_pref(
    rho_brine_0, b_brine, z_LS
    )  # returns Pref_LS [MPa], rho_br_LS_ss [kg/m³]

    Pref_HS, rho_br_HS_ss = secant_solve_brine_pref(
    rho_brine_0, b_brine, z_HS
    )  # returns Pref_HS [MPa], rho_br_HS_ss [kg/m³]


    deltaP0_LS = P_LS_ss - Pref_LS      # MPa
    deltaP0_HS = P_HS_ss - Pref_HS      # MPa

    Vp0_LS = phi * Vrock_LS             # m³
    Vp0_HS = phi * Vrock_HS             # m³

    Vp_LS_ss = Vp0_LS * (1 + b_pore * deltaP0_LS)  # m³
    Vp_HS_ss = Vp0_HS * (1 + b_pore * deltaP0_HS)  # m³

    Vbr_LS_ss = Vp0_LS * (1 - b_brine * deltaP0_LS)  # m³
    Vbr_HS_ss = Vp0_HS * (1 - b_brine * deltaP0_HS)  # m³

    Vco2_LS_ss = Vp_LS_ss - Vbr_LS_ss    # m³
    Vco2_HS_ss = Vp_HS_ss - Vbr_HS_ss    # m³

    volfrac_co2_LS_ss = Vco2_LS_ss / Vp_LS_ss  # –
    volfrac_br_LS_ss  = 1 - volfrac_co2_LS_ss  # –
    volfrac_co2_HS_ss = Vco2_HS_ss / Vp_HS_ss  # –
    volfrac_br_HS_ss  = 1 - volfrac_co2_HS_ss  # –

    Mco2_LS_ss = rho_co2_LS_ss * Vco2_LS_ss  # kg
    Mco2_HS_ss = rho_co2_HS_ss * Vco2_HS_ss  # kg
    Mbr_LS_ss  = rho_br_LS_ss   * Vbr_LS_ss   # kg
    Mbr_HS_ss  = rho_br_HS_ss   * Vbr_HS_ss   # kg

    massfrac_co2_LS_ss = Mco2_LS_ss / (Mco2_LS_ss + Mbr_LS_ss)  # –
    massfrac_br_LS_ss  = 1 - massfrac_co2_LS_ss              # –
    massfrac_co2_HS_ss = Mco2_HS_ss / (Mco2_HS_ss + Mbr_HS_ss)  # –
    massfrac_br_HS_ss  = 1 - massfrac_co2_HS_ss              # –

    df0 = pd.DataFrame({
        'z_LS':[z_LS],'z_HS':[z_HS],                       # m
        'Pref_LS':[Pref_LS],'Pref_HS':[Pref_HS],           # MPa
        'rho_br_LS_ss':[rho_br_LS_ss],'rho_br_HS_ss':[rho_br_HS_ss],  # kg/m³
        'deltaP0_LS':[deltaP0_LS],'deltaP0_HS':[deltaP0_HS],        # MPa
        'Vp0_LS':[Vp0_LS],'Vp0_HS':[Vp0_HS],               # m³
        'Vp_LS_ss':[Vp_LS_ss],'Vp_HS_ss':[Vp_HS_ss],       # m³
        'Vbr_LS_ss':[Vbr_LS_ss],'Vbr_HS_ss':[Vbr_HS_ss],   # m³
        'Vco2_LS_ss':[Vco2_LS_ss],'Vco2_HS_ss':[Vco2_HS_ss],# m³
        'volfrac_co2_LS_ss':[volfrac_co2_LS_ss],'volfrac_br_LS_ss':[volfrac_br_LS_ss], # –
        'volfrac_co2_HS_ss':[volfrac_co2_HS_ss],'volfrac_br_HS_ss':[volfrac_br_HS_ss], # –
        'massfrac_co2_LS_ss':[massfrac_co2_LS_ss],'massfrac_br_LS_ss':[massfrac_br_LS_ss],# –
        'massfrac_co2_HS_ss':[massfrac_co2_HS_ss],'massfrac_br_HS_ss':[massfrac_br_HS_ss],# –
        'Mco2_LS_ss':[Mco2_LS_ss],'Mbr_LS_ss':[Mbr_LS_ss],  # kg
        'Mco2_HS_ss':[Mco2_HS_ss],'Mbr_HS_ss':[Mbr_HS_ss] })

    # === Phase 1: dispatch ===
    Vp_LS, Vp_HS = Vp_LS_ss, Vp_HS_ss

    # initialize brine volume with the same exponential law used in the loop  
    rho_br_LS = rho_br_LS_ss * np.exp(b_brine * (P_LS_ss - Pref_LS))  
    rho_br_HS = rho_br_HS_ss * np.exp(b_brine * (P_HS_ss - Pref_HS))  
    Vbr_LS    = Mbr_LS_ss   / rho_br_LS  
    Vbr_HS    = Mbr_HS_ss   / rho_br_HS  


    n = len(generation)
    demand = np.full(n, stable_power_requirement)  # kW
    b_eff = b_pore + b_brine                    # 1/MPa

    # state
    P_LS, P_HS = P_LS_ss, P_HS_ss               # MPa
    rho_co2_LS, rho_co2_HS = rho_co2_LS_ss, rho_co2_HS_ss  # kg/m³
    Mco2_LS, Mco2_HS = Mco2_LS_ss, Mco2_HS_ss  # kg
    Mbr_LS, Mbr_HS = Mbr_LS_ss, Mbr_HS_ss     # kg

    records = []
    for t in range(n):
        # right at top of your timestep loop, save "previous" state:
        P_LS_prev, P_HS_prev     = P_LS,    P_HS
        Vp_LS_prev, Vp_HS_prev   = Vp_LS,   Vp_HS
        Vbr_LS_prev, Vbr_HS_prev = Vbr_LS,  Vbr_HS
        
        gen, dem = generation[t], demand[t]    # kW
        E_res = (gen - dem) * dt               # kWh
        mode = 'charge' if E_res>0 else 'discharge' if E_res<0 else 'standby'

        ekg_chg = W_compressor_T * dt          # kWh/kg
        ekg_dis = W_turbine_T * dt             # kWh/kg
        if mode=='charge':
            m_req = E_res/ekg_chg              # kg
            m_pow = power_rating*dt/ekg_chg    # kg
        elif mode=='discharge':
            m_req = -E_res/ekg_dis             # kg
            m_pow = power_rating*dt/ekg_dis    # kg
        else:
            m_req = m_pow = 0.0                # kg

        # headroom ΔP
        abs_dP_LS   = pressure_tolerance * P_LS_ss       # MPa
        abs_dP_HS   = pressure_tolerance * P_HS_ss       # MPa
        # allowable bounds [MPa]
        Pmin_LS = P_LS_ss * (1 - pressure_tolerance)
        Pmax_LS = P_LS_ss * (1 + pressure_tolerance)
        Pmin_HS = P_HS_ss * (1 - pressure_tolerance)
        Pmax_HS = P_HS_ss * (1 + pressure_tolerance)

        # headroom toward the tighter bound in each reservoir [MPa]
        head_LS = max(0.0,
              # if charging, headroom toward upper bound;
              # if discharging, headroom toward lower bound
              (Pmax_LS - P_LS) if mode=='charge'
              else (P_LS - Pmin_LS))
        head_HS = max(0.0,
              (Pmax_HS - P_HS) if mode=='charge'
              else (P_HS - Pmin_HS))

        # pick which reservoir is most restrictive
        reservoir_responsible_pressure_constraint = (
        'LS' if head_LS < head_HS else 'HS'
        )

        # mass limits [kg]
        mlim_LS = head_LS * rho_co2_LS * Vp_LS_prev * b_eff  
        mlim_HS = head_HS * rho_co2_HS * Vp_HS_prev * b_eff  

        m_constraint = min(m_req, m_pow, pressure_buffer_fraction*min(mlim_LS, mlim_HS))
        constraint = (
        'residual' if abs(m_constraint - m_req) < 1e-12 else
        'pressure' if abs(m_constraint - pressure_buffer_fraction*min(mlim_LS, mlim_HS)) < 1e-12 else
        'power'
        )
        dM_LS = -m_constraint if mode=='charge' else (m_constraint if mode=='discharge' else 0.0)
        dM_HS =  m_constraint if mode=='charge' else (-m_constraint if mode=='discharge' else 0.0)


        # 1) implicit solve for new P, rho_co2, AND Vp
        P_LS, rho_co2_LS, Vp_LS = secant_solve_compressibility_implicit(
        P_LS_prev, dM_LS, Vp_LS_prev, b_eff, T_LS, thermo_file,
        P_LS_prev*0.99, P_LS_prev*1.01
        )
        P_HS, rho_co2_HS, Vp_HS = secant_solve_compressibility_implicit(
        P_HS_prev, dM_HS, Vp_HS_prev, b_eff, T_HS, thermo_file,
        P_HS_prev*0.99, P_HS_prev*1.01
        )

        # 2) update CO2 masses
        Mco2_LS += dM_LS
        Mco2_HS += dM_HS

        # 3) recompute brine density & volumes (brine mass constant)
        rho_br_LS = rho_br_LS_ss * np.exp(b_brine * (P_LS - Pref_LS))
        rho_br_HS = rho_br_HS_ss * np.exp(b_brine * (P_HS - Pref_HS))
        Vbr_LS    = Mbr_LS   / rho_br_LS
        Vbr_HS    = Mbr_HS   / rho_br_HS

        # 4) compute your diagnostics
        deltaP_LS      = P_LS   - P_LS_prev         # MPa
        deltaP_HS      = P_HS   - P_HS_prev         # MPa

        pore_gain_LS   = Vp_LS - Vp_LS_prev         # m³
        pore_gain_HS   = Vp_HS - Vp_HS_prev         # m³

        extra_cap_LS   = pore_gain_LS / Vp_LS_prev  # –  
        extra_cap_HS   = pore_gain_HS / Vp_HS_prev  # –

        brine_shrink_LS = Vbr_LS_prev - Vbr_LS      # m³
        brine_shrink_HS = Vbr_HS_prev - Vbr_HS      # m³
        
        deltaV_LS_br    = Vbr_LS    - Vbr_LS_prev   # m³  (negative of shrink)
        deltaV_HS_br    = Vbr_HS    - Vbr_HS_prev   # m³

        # 5) now you can recompute CO2 volumes & fractions with the new Vp:
        Vco2_LS        = Mco2_LS    / rho_co2_LS   # m³
        Vco2_HS        = Mco2_HS    / rho_co2_HS   # m³
        volfrac_co2_LS = Vco2_LS    / Vp_LS        # –
        volfrac_co2_HS = Vco2_HS    / Vp_HS        # –
        volfrac_br_LS  = 1 - volfrac_co2_LS       # –
        volfrac_br_HS  = 1 - volfrac_co2_HS       # –

        massfrac_co2_LS = Mco2_LS  / (Mco2_LS + Mbr_LS)  # –
        massfrac_co2_HS = Mco2_HS  / (Mco2_HS + Mbr_HS)  # –

        # kWh charged into (positive) or discharged from (negative) storage
        charge = (
            m_constraint * ekg_chg        if mode == 'charge'    else
            -m_constraint * ekg_dis       if mode == 'discharge' else
            0.0
        )  # kWh

        if mode == 'charge':
            # sell all renewables + any residual you couldn't stuff into storage
            sold = gen*dt + (E_res - charge)  # kWh
            bought = 0.0                     # kWh

        elif mode == 'standby':
            # sell exactly your renewables, no storage movement
            sold = gen*dt                   # kWh
            bought = 0.0                     # kWh

        else:  # discharge
            # sell renewables plus whatever you discharge
            sold = gen*dt + (-charge)       # kWh  (–charge is the discharge kWh)
            # if that's still below demand, buy the shortfall at PPA price
            bought = max(0.0, dem*dt - sold)  # kWh

        # economic flows
        revenue  = price[t] * sold         # $ 
        cost      = price[t] * bought      # $ 
        cashflow  = revenue - cost         # $
        balance   = (records[-1]['balance'] + cashflow) if records else cashflow  # $

        deltaP_LS_QSS = P_LS - P_LS_ss
        deltaP_HS_QSS = P_HS - P_HS_ss


        records.append({
            'mode':mode,
            'residual':E_res,
            'charge':charge,
            'charge_mass_co2':m_constraint,
            'pressure_constraint_mass_co2':pressure_buffer_fraction*min(mlim_LS,mlim_HS),
            'power_constraint':power_rating,
            'deltaM_co2_LS':dM_LS,'deltaM_co2_HS':dM_HS,
            'M_co2_LS':Mco2_LS,'M_co2_HS':Mco2_HS,
            'M_br_LS':Mbr_LS,'M_br_HS':Mbr_HS,
            'pore_gain_LS':pore_gain_LS,'pore_gain_HS':pore_gain_HS,
            'extra_cap_LS':extra_cap_LS,'extra_cap_HS':extra_cap_HS,
            'brine_shrink_LS':brine_shrink_LS,'brine_shrink_HS':brine_shrink_HS,
            'Vp_LS':Vp_LS,'Vp_HS':Vp_HS,
            'V_LS_co2':Vco2_LS,'V_HS_co2':Vco2_HS,
            'Vbr_LS':Vbr_LS,'Vbr_HS':Vbr_HS,
            'volfrac_LS_co2':volfrac_co2_LS,'volfrac_HS_co2':volfrac_co2_HS,
            'volfrac_LS_br':volfrac_br_LS,'volfrac_HS_br':volfrac_br_HS,
            'massfrac_LS_co2':massfrac_co2_LS,'massfrac_HS_co2':massfrac_co2_HS,
            'massfrac_LS_br':1-massfrac_co2_LS,'massfrac_HS_br':1-massfrac_co2_HS,
            'deltaV_LS_br':deltaV_LS_br,'deltaV_HS_br':deltaV_HS_br,
            'P_LS':P_LS,'P_HS':P_HS,
            'diffP_LS':deltaP_LS,'diffP_HS':deltaP_HS,
            'QSS_P_drift_LS': deltaP_LS_QSS, 'QSS_P_drift_HS': deltaP_HS_QSS,
            'errorP_LS':deltaP_LS_QSS/P_LS_ss*100,'errorP_HS':deltaP_HS_QSS/P_HS_ss*100,
            'purchase': bought, 'sale': sold, 'revenue': revenue, 'expense': cost,
            'cashflow':cashflow,'balance':balance,
            'generation':gen,'demand':dem,'price':price[t],
            'rho_co2_LS':rho_co2_LS,'rho_co2_HS':rho_co2_HS,
            'rho_br_LS':rho_br_LS,'rho_br_HS':rho_br_HS,
            'b_eff':b_eff,'residual_mass':m_req,
             'constraint': constraint, 'reservoir_responsible_pressure_constraint': reservoir_responsible_pressure_constraint
        })

    df1 = pd.DataFrame(records)
    return df0, df1
