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
    price_arr,           
    is_offpeak_arr,     
    is_vpp_arr,          
    tariff_mode_int,   
    bat_cap,           
    init_soc_pct,     
    min_soc_pct,       
    max_soc_pct,     
    max_chg_kw,      
    max_dis_kw,      
    eff_roundtrip    
):
    n = len(net_load_arr)
    
    soc_tracker = np.zeros(n)
    bat_power_out = np.zeros(n) 
    
    current_kwh = bat_cap * init_soc_pct
    min_kwh = bat_cap * min_soc_pct
    max_kwh = bat_cap * max_soc_pct
    
    eff_oneway = eff_roundtrip ** 0.5
    dt = 5.0 / 60.0 
    
    # =====================================================================
    # PENGATURAN ARBITRASE 
    # =====================================================================
    TARGET_SOC_ARB_PCT = 0.30     
    PRICE_WHOLESALE_CHEAP = 0.05  
    PRICE_NEGATIVE = 0.0          
    # =====================================================================

    for i in range(n):
        net_load = net_load_arr[i]
        
        # Ambil kondisi spesifik pada jam ini dari array
        current_price = price_arr[i]
        is_off_peak = is_offpeak_arr[i]
        is_vpp_dispatch = is_vpp_arr[i]

        target_power = 0.0
        target_soc_kwh = bat_cap * TARGET_SOC_ARB_PCT
        
        # 1. PRIORITAS TERTINGGI: VPP DISPATCH
        if is_vpp_dispatch:
            target_power = max_dis_kw
            
        # 2. PRIORITAS KEDUA: VPP CHARGE (Harga Minus)
        elif current_price < PRICE_NEGATIVE:
            target_power = -max_chg_kw
            
        # 3. PRIORITAS KETIGA: ARBITRASE TOU & WHOLESALE MURAH
        # tariff_mode_int: 1 = ToU, 2 = Wholesale Price
        elif (tariff_mode_int == 1 and is_off_peak) or (tariff_mode_int == 2 and current_price <= PRICE_WHOLESALE_CHEAP):
            if current_kwh < target_soc_kwh:
                power_to_target = -((target_soc_kwh - current_kwh) / (eff_oneway * dt))
                if net_load < 0:
                    target_power = min(net_load, power_to_target)
                else:
                    target_power = power_to_target
            else:
                if net_load < 0:
                    target_power = net_load
                else:
                    target_power = 0.0

        # 4. PRIORITAS KEEMPAT: BASELINE NORMAL
        else:
            if net_load > 0:
                target_power = net_load
            else:
                target_power = net_load
                
        # --- FISIKA BATERAI (JANGAN DIUBAH) ---
        target_power = max(-max_chg_kw, min(max_dis_kw, target_power))
        real_power = 0.0
        
        if target_power < 0: 
            max_energy_in = max_kwh - current_kwh
            limit_p_charge = -(max_energy_in / (eff_oneway * dt))
            real_power = max(target_power, limit_p_charge)
            energy_change = real_power * eff_oneway * dt
            current_kwh -= energy_change 
        else: 
            max_energy_out = current_kwh - min_kwh
            limit_p_discharge = (max_energy_out * eff_oneway) / dt
            real_power = min(target_power, limit_p_discharge)
            energy_change = (real_power / eff_oneway) * dt
            current_kwh -= energy_change
            
        if current_kwh < 0: current_kwh = 0.0
        if current_kwh > bat_cap: current_kwh = bat_cap
            
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
    
    # 1. Siapkan Array Waktu
    is_offpeak = get_time_mask(time_float, params['t_offpeak_start'], params['t_offpeak_end'])
    
    # 2. Siapkan Array Harga & VPP
    arr_price = df['price_import'].to_numpy(dtype=np.float64)
    vpp_thresh = params['dispatch_price_threshold']
    is_vpp_arr = arr_price >= vpp_thresh
    
    # 3. Konversi Nama Skema menjadi Angka (Karena Numba tidak bisa baca huruf)
    scheme_name = params.get('tariff_scheme', 'Flat')
    if scheme_name == 'Time of Use':
        tariff_mode_int = 1
    elif scheme_name == 'Wholesale Price':
        tariff_mode_int = 2
    else:
        tariff_mode_int = 0 # Flat
        
    # --- 4. Kalkulasi Baterai (Panggil dengan argumen yang baru) ---
    soc_pct, bat_power = simulate_battery_numba(
        net_load_pure,
        arr_price,         # Data Harga
        is_offpeak,        # Data Kapan Off-Peak
        is_vpp_arr,        # Data Kapan Harga Meledak
        tariff_mode_int,   # Skema Tarif dalam bentuk angka
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

    # Hitung Kapasitas Baterai kWh
    df_res['battery_soc_kwh'] = (df_res['battery_soc_pct'] / 100.0) * params['battery_capacity_kwh']
    
    # 2. Hitung Tarif Ekspor & Impor
    scheme = params.get('tariff_scheme', 'Flat')
    
    if scheme == 'Wholesale Price': 
        df_fees = params.get('df_wholesale_fees', pd.DataFrame())
        
        if 'price_profile' in df_res.columns and not df_fees.empty:
            
            spot_kwh = df_res['price_profile'] / 1000.0
            
            
            years = df_res['timestamp'].dt.year
            months = df_res['timestamp'].dt.month
            
            fy_start_yr = np.where(months >= 7, years, years - 1)
            
            fy_str = (pd.Series(fy_start_yr) % 100).astype(str).str.zfill(2) + '/' + ((pd.Series(fy_start_yr) + 1) % 100).astype(str).str.zfill(2)
            
            m_map = df_fees.set_index('FY_Year')['Market_Fee'].to_dict()
            n_map = df_fees.set_index('FY_Year')['Network_Fee'].to_dict()
            o_map = df_fees.set_index('FY_Year')['Other_Fee'].to_dict()
            
            m_fee = fy_str.map(m_map).fillna(0).values
            n_fee = fy_str.map(n_map).fillna(0).values
            o_fee = fy_str.map(o_map).fillna(0).values
            
            df_res['tariff_import_AUD'] = spot_kwh + m_fee + n_fee + o_fee
            df_res['tariff_export_AUD'] = spot_kwh + m_fee
            
        else:
            df_res['tariff_import_AUD'] = 0.0
            df_res['tariff_export_AUD'] = 0.0
        
    elif scheme == 'Time of Use':
        p_start = params['t_peak_start'].hour
        p_end = params['t_peak_end'].hour
        s_start = params['t_shoulder_start'].hour
        s_end = params['t_shoulder_end'].hour
        
        hours = df_res['timestamp'].dt.hour
        
        def get_mask(h_array, start, end):
            if start < end:
                return (h_array >= start) & (h_array < end)
            elif start > end:
                return (h_array >= start) | (h_array < end)
            else:
                return pd.Series(False, index=h_array.index)
        
        cond_peak = get_mask(hours, p_start, p_end)
        cond_shoulder = get_mask(hours, s_start, s_end)
        
        # Eksekusi Numpy Select untuk IMPORT
        df_res['tariff_import_AUD'] = np.select(
            [cond_peak, cond_shoulder], 
            [params['peak_price'], params['shoulder_price']], 
            default=params['offpeak_price']
        )
        
        # Eksekusi Numpy Select untuk EXPORT
        df_res['tariff_export_AUD'] = np.select(
            [cond_peak, cond_shoulder], 
            [params.get('exp_peak', 0.0), params.get('exp_shoulder', 0.0)], 
            default=params.get('exp_offpeak', 0.0)
        )
        
    else: # Default: Flat
        df_res['tariff_import_AUD'] = params['import_flat']
        df_res['tariff_export_AUD'] = params['export_price']
        
    final_cols = [
        'timestamp',
        'irradiance',
        'temperature',
        'load_profile',
        'price_profile',
        'solar_output_kw',
        'vpp_status',
        'battery_soc_pct',
        'battery_soc_kwh',
        'battery_power_ac_kw',
        'grid_net_kw',
        'tariff_import_AUD',
        'tariff_export_AUD'
    ]
    
    avail_cols = [c for c in final_cols if c in df_res.columns]

    df_export = df_res[avail_cols].copy()
    
    tariff_cols = ['tariff_import_AUD', 'tariff_export_AUD']
    for c in tariff_cols:
        if c in df_export.columns:
            df_export[c] = df_export[c].round(5)
            
    other_cols = [c for c in df_export.columns if c not in tariff_cols and c != 'timestamp']
    for c in other_cols:
        df_export[c] = df_export[c].round(2)
        
    return df_export