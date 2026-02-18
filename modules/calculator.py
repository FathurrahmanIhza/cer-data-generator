import numpy as np
import pandas as pd
from numba import jit

MODE_SHOULDER = 0  
MODE_CHARGE   = 1 
MODE_DISCHARGE= 2 
MODE_PEAK     = 3  

@jit(nopython=True)
def simulate_battery_numba(
    net_load_arr,      
    strategy_arr,     
    bat_cap,           
    init_soc_pct,     
    min_soc_pct,       
    max_soc_pct,     
    max_chg_kw,      
    max_dis_kw,      
    eff_roundtrip    
):
    n = len(net_load_arr)
    
    # Pre-allocation array hasil
    soc_tracker = np.zeros(n)
    bat_power_out = np.zeros(n) 
    
    # Setup Variable Awal
    current_kwh = bat_cap * init_soc_pct
    min_kwh = bat_cap * min_soc_pct
    max_kwh = bat_cap * max_soc_pct
    
    # Efisiensi satu arah (akar kuadrat dari roundtrip)
    eff_oneway = eff_roundtrip ** 0.5
    
    dt = 5.0 / 60.0 
    
    for i in range(n):
        net_load = net_load_arr[i]
        mode = strategy_arr[i]
        
        target_power = 0.0
        
        
        if mode == MODE_DISCHARGE:
            # Kasus VPP: Paksa Discharge Maksimal
            target_power = max_dis_kw
            
        elif mode == MODE_CHARGE:
            # Kasus Off-peak / Negative Price: Paksa Charge Maksimal
            target_power = -max_chg_kw
            
        elif mode == MODE_PEAK:
            # Kasus Peak: Self Consumption / Shaving
            if net_load > 0:
                target_power = net_load
            else:
                target_power = 0.0
                
        else: 
            # Kasus Shoulder: Hanya charge jika ada surplus solar (net_load negatif)
            if net_load < 0:
                target_power = net_load
            else:
                target_power = 0.0
        
        # Batasi Power sesuai Rating Inverter
        target_power = max(-max_chg_kw, min(max_dis_kw, target_power))
        
        real_power = 0.0
        
        if target_power < 0: 
            # --- CHARGING ---
            max_energy_in = max_kwh - current_kwh
            
            limit_p_charge = -(max_energy_in / (eff_oneway * dt))
            real_power = max(target_power, limit_p_charge)
            # Update SoC
            energy_change = real_power * eff_oneway * dt
            current_kwh -= energy_change 
            
        else: 
            # --- DISCHARGING ---
            max_energy_out = current_kwh - min_kwh
            
            limit_p_discharge = (max_energy_out * eff_oneway) / dt
            real_power = min(target_power, limit_p_discharge)
            # Update SoC
            energy_change = (real_power / eff_oneway) * dt
            current_kwh -= energy_change
            
        if current_kwh < 0: current_kwh = 0.0
        if current_kwh > bat_cap: current_kwh = bat_cap
            
        # Simpan Hasil
        bat_power_out[i] = real_power
        if bat_cap > 0:
            soc_tracker[i] = (current_kwh / bat_cap) * 100.0
        else:
            soc_tracker[i] = 0.0
            
    return soc_tracker, bat_power_out


def get_time_mask(time_float_arr, start_t, end_t):
    """
    Membuat array True/False apakah jam saat ini masuk rentang waktu.
    Support lintas hari (misal 22:00 - 05:00).
    """
    s_val = start_t.hour + start_t.minute / 60.0
    e_val = end_t.hour + end_t.minute / 60.0
    
    if s_val < e_val:
        return (time_float_arr >= s_val) & (time_float_arr < e_val)
    else:
        return (time_float_arr >= s_val) | (time_float_arr < e_val)


def run_simulation(df, params):
    
    arr_irr = df['irradiance'].to_numpy(dtype=np.float64)
    arr_temp = df['temperature'].to_numpy(dtype=np.float64)
    arr_load = df['load_profile'].to_numpy(dtype=np.float64)
    
    # Hitung Solar Output 
    temp_factor = 1 + (params['temp_coeff'] * (arr_temp))
    solar_kw = params['solar_capacity_kw'] * (arr_irr / 1000.0) * temp_factor * params['pr']
    solar_kw = np.maximum(solar_kw, 0.0) 
    
    # Hitung Net Load Awal (Beban Murni - Solar)
    net_load_pure = arr_load - solar_kw
    
    # PERSIAPAN STRATEGI MODE 
    timestamps = df['timestamp']
    time_float = timestamps.dt.hour + timestamps.dt.minute / 60.0
    time_float = time_float.to_numpy(dtype=np.float64)
    
    is_offpeak = get_time_mask(time_float, params['t_offpeak_start'], params['t_offpeak_end'])
    is_peak    = get_time_mask(time_float, params['t_peak_start'], params['t_peak_end'])
    
    arr_price = df['price_import'].to_numpy(dtype=np.float64)
    vpp_thresh = params['dispatch_price_threshold']
    
    # Default: Shoulder (Mode 0)
    strategy_map = np.full(len(df), MODE_SHOULDER, dtype=np.int8)
    
    # 1. Base Time of Use
    strategy_map[is_peak] = MODE_PEAK       # Mode 3
    strategy_map[is_offpeak] = MODE_CHARGE  # Mode 1
    # 2. Override: Negative Price
    strategy_map[arr_price < 0] = MODE_CHARGE # Mode 1
    # 3. Override: VPP (Highest Priority)
    strategy_map[arr_price > vpp_thresh] = MODE_DISCHARGE # Mode 2

    # --- 4. Kalkulasi Baterai ---
    soc_pct, bat_power = simulate_battery_numba(
        net_load_pure,
        strategy_map,
        params['battery_capacity_kwh'],
        params['battery_initial_soc'],
        params['soc_min_pct'],
        params['soc_max_pct'],
        params['max_charge_kw'],
        params['max_discharge_kw'],
        params['battery_efficiency']
    )
    

    df_res = df.copy()
    
    df_res['solar_output_kw'] = solar_kw
    df_res['battery_power_ac_kw'] = bat_power
    df_res['battery_soc_pct'] = soc_pct
    
    # Hitung Grid Net 
    df_res['grid_net_kw'] = arr_load - solar_kw - bat_power
    df_res['vpp_status'] = (arr_price > vpp_thresh)
    
    if 'price_import' in df_res.columns:
        df_res.rename(columns={'price_import': 'price_profile'}, inplace=True)
        
    final_cols = [
        'timestamp',
        'irradiance',
        'temperature',
        'load_profile',
        'price_profile',
        'solar_output_kw',
        'vpp_status',
        'battery_soc_pct',
        'battery_power_ac_kw',
        'grid_net_kw'
    ]
    
    avail_cols = [c for c in final_cols if c in df_res.columns]
    
    return df_res[avail_cols].round(2)