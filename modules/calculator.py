import pandas as pd
import numpy as np

def check_time_window(current_time, start_time, end_time):
    """
    Cek apakah waktu sekarang berada dalam rentang start-end.
    Menghandle kasus lintas hari (misal 23:00 s/d 07:00).
    """
    if start_time < end_time:
        return start_time <= current_time < end_time
    else:
        return current_time >= start_time or current_time < end_time

def run_simulation(df, params):
    
    # --- 1. PRE-CALCULATION (SOLAR) ---
    temp_factor = 1 + (params['temp_coeff'] * (df['temperature']))
    df['solar_output_kw'] = params['solar_capacity_kw'] * (df['irradiance'] / 1000) * temp_factor * params['pr']
    df['solar_output_kw'] = df['solar_output_kw'].clip(lower=0)
    
    # --- 2. SETUP BATERAI & VPP ---
    bat_cap = params['battery_capacity_kwh']
    soc_min_kwh = bat_cap * params['soc_min_pct']
    soc_max_kwh = bat_cap * params['soc_max_pct']
    eff_one_way = params['battery_efficiency'] ** 0.5 
    
    max_charge_kw = params['max_charge_kw']
    max_discharge_kw = params['max_discharge_kw']
    vpp_threshold = params['dispatch_price_threshold']
    
    # List Penampung Hasil
    bat_power_list = [] 
    soc_list = []       
    vpp_status_list = [] 
    
    current_soc_kwh = bat_cap * params['battery_initial_soc']
    dt = 5 / 60 
    
    records = df.to_dict('records')
    
    # --- 3. CORE LOOP (SIMULASI PER BARIS) ---
    for row in records:
        load = row['load_profile']
        solar = row['solar_output_kw']
        
        market_price = row['price_import'] 
        
        cur_time = row['timestamp'].time()
        
        net_load_pure = load - solar
        target_bat_power = 0 
        
        # --- A. CEK VPP STATUS (CONTROL LOGIC) ---
        # Trigger: Apakah Harga Pasar > Threshold user?
        is_vpp_active = market_price > vpp_threshold
        vpp_status_list.append(is_vpp_active)
        
        # --- B. ENERGY MANAGEMENT STRATEGY ---
        if is_vpp_active:
            # MODE 1: VPP DISPATCH (Prioritas Tertinggi)
            # Paksa discharge maksimum untuk respon harga tinggi
            target_bat_power = max_discharge_kw
            
        else:
            # MODE 2: BASELINE STRATEGY (Cek Jam)
            is_offpeak = check_time_window(cur_time, params['t_offpeak_start'], params['t_offpeak_end'])
            is_peak = check_time_window(cur_time, params['t_peak_start'], params['t_peak_end'])
            
            if is_offpeak:
                # Charge saat Off-Peak
                target_bat_power = -max_charge_kw 
                
            elif is_peak:
                # Peak Shaving / Self-Consumption
                if net_load_pure > 0:
                    target_bat_power = net_load_pure 
                else:
                    target_bat_power = 0
            else: 
                # Shoulder (Siang)
                if net_load_pure < 0:
                    target_bat_power = net_load_pure # Simpan surplus solar
                else:
                    target_bat_power = net_load_pure # Support beban (opsional)

        # --- C. BATTERY PHYSICS CONSTRAINTS ---
        # 1. Batasi Power (Inverter Rating)
        target_bat_power = max(-max_charge_kw, min(max_discharge_kw, target_bat_power))
        
        # 2. Batasi Energy (Kapasitas kWh)
        if target_bat_power < 0: # CHARGING
            max_energy_in = soc_max_kwh - current_soc_kwh
            max_p_charge = - (max_energy_in / (eff_one_way * dt))
            real_bat_power = max(target_bat_power, max_p_charge)
            
            # Update SoC (Ingat: Charge = Negatif power)
            energy_change = real_bat_power * eff_one_way * dt 
            current_soc_kwh = current_soc_kwh - energy_change
            
        else: # DISCHARGING
            max_energy_out = current_soc_kwh - soc_min_kwh
            max_p_discharge = (max_energy_out * eff_one_way) / dt
            real_bat_power = min(target_bat_power, max_p_discharge)
            
            # Update SoC
            energy_change = (real_bat_power / eff_one_way) * dt 
            current_soc_kwh = current_soc_kwh - energy_change
            
        # Safety Clip SoC
        current_soc_kwh = max(0, min(bat_cap, current_soc_kwh))
        
        # Simpan Data
        bat_power_list.append(real_bat_power)
        soc_list.append((current_soc_kwh / bat_cap) * 100) 
        
    # --- 4. FORMAT OUTPUT ---
    
    df['vpp_status'] = vpp_status_list
    df['battery_power_ac_kw'] = bat_power_list 
    df['battery_soc_pct'] = soc_list
    
    df['grid_net_kw'] = df['load_profile'] - df['solar_output_kw'] - df['battery_power_ac_kw']
    df.rename(columns={'price_import': 'price_profile'}, inplace=True)
    
    final_columns = [
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

    return df[final_columns].round(2)