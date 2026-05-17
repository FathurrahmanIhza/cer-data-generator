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
    is_peak_arr,        
    is_shoulder_arr,    
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
    
    TARGET_SOC_ARB_PCT = 0.30     
    PRICE_WHOLESALE_CHEAP = 0.05  # Diubah ke AUD/kWh (50 AUD/MWh)
    PRICE_WHOLESALE_HIGH = 0.10   # Diubah ke AUD/kWh (100 AUD/MWh)
    PRICE_NEGATIVE = 0.0          

    for i in range(n):
        net_load = net_load_arr[i]
        
        current_price = price_arr[i] # Sekarang membaca Tariff Export matang
        is_off_peak = is_offpeak_arr[i]
        is_peak = is_peak_arr[i]
        is_shoulder = is_shoulder_arr[i]
        is_vpp_dispatch = is_vpp_arr[i]

        target_power = 0.0
        target_soc_kwh = bat_cap * TARGET_SOC_ARB_PCT
        
        # ---------------------------------------------------------
        # 1. PRIORITAS TERTINGGI: VPP DISPATCH
        # ---------------------------------------------------------
        if is_vpp_dispatch:
            target_power = max_dis_kw
            
        # ---------------------------------------------------------
        # 2. PRIORITAS KEDUA: VPP CHARGE (Harga Minus)
        # ---------------------------------------------------------
        elif current_price < PRICE_NEGATIVE:
            target_power = -max_chg_kw
            
        # ---------------------------------------------------------
        # 3. SKEMA TIME OF USE (ToU)
        # ---------------------------------------------------------
        elif tariff_mode_int == 1:
            if is_off_peak:
                # Off-peak: Boleh discharge, TAPI batas bawahnya 30%.
                if current_kwh < target_soc_kwh:
                    power_to_target = -((target_soc_kwh - current_kwh) / (eff_oneway * dt))
                    target_power = min(net_load, power_to_target) if net_load < 0 else power_to_target
                else:
                    if net_load > 0:
                        max_allowed_discharge = (current_kwh - target_soc_kwh) * eff_oneway / dt
                        target_power = min(net_load, max_allowed_discharge)
                    else:
                        target_power = net_load
            
            elif is_shoulder:
                # Shoulder: Boleh discharge & charge normal (Flat)
                target_power = net_load
                
            else: # Jam Peak 
                # Peak: Boleh discharge & charge normal
                target_power = net_load

        # ---------------------------------------------------------
        # 4. SKEMA WHOLESALE PRICE
        # ---------------------------------------------------------
        elif tariff_mode_int == 2:
            if current_price <= PRICE_WHOLESALE_CHEAP:
                if current_kwh < target_soc_kwh:
                    if net_load < 0:
                        # ADA excess matahari: Murni pakai matahari saja, DILARANG beli tambahan dari grid
                        target_power = net_load
                    else:
                        # TIDAK ADA excess matahari: Beli murni dari grid untuk kejar 30%
                        power_to_target = -((target_soc_kwh - current_kwh) / (eff_oneway * dt))
                        target_power = power_to_target
                else:
                    # Sudah 30% ke atas: Hanya terima excess matahari, tidak beli grid
                    target_power = net_load if net_load < 0 else 0.0
            
            elif current_price >= PRICE_WHOLESALE_HIGH:
                # Boleh discharge. 
                target_power = net_load
                
            else:
                # BATERAI DIAM (0.0). Rumah murni pakai listrik Grid jika tidak ada matahari.
                target_power = net_load if net_load < 0 else 0.0
        # ---------------------------------------------------------
        # 5. SKEMA FLAT (Baseline Normal)
        # ---------------------------------------------------------
        else:
            # Apapun jamnya, prioritas utama: Solar -> Baterai -> Grid
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

# =====================================================================
# FUNGSI NUMBA UNTUK EXTRA IMPORT VPP 
# =====================================================================
@jit(nopython=True)
def calculate_extra_import_numba(vpp_discharge_arr, bat_power_arr, grid_net_arr, soc_kwh_arr, dt_hours):
    n_rows = len(grid_net_arr)
    arr_extra_import = np.zeros(n_rows)
    
    arr_bat_discharge = np.where(bat_power_arr > 0, bat_power_arr, 0.0)
    arr_grid_import = np.where(grid_net_arr > 0, grid_net_arr, 0.0)
    arr_vpp_bat_discharge = np.where(vpp_discharge_arr, arr_bat_discharge, 0.0)
    
    idx = 0
    while idx < n_rows:
        if vpp_discharge_arr[idx]:
            start_idx = idx
            end_idx = idx
            gap = 0
            j = idx + 1
            
            while j < n_rows:
                if vpp_discharge_arr[j]:
                    end_idx = j
                    gap = 0
                else:
                    gap += 1
                    if gap > 1:
                        break
                j += 1
                
            soc_start = soc_kwh_arr[start_idx]
            
            e_vpp_kwh = 0.0
            for sum_idx in range(start_idx, end_idx + 1):
                e_vpp_kwh += arr_vpp_bat_discharge[sum_idx]
            e_vpp_kwh *= dt_hours
            
            e_import_kwh = 0.0
            max_tracking_steps = min(n_rows, end_idx + 1 + 288)
            
            for k in range(end_idx + 1, max_tracking_steps):
                imp_kw = arr_grid_import[k]
                soc_now = soc_kwh_arr[k]
                
                if imp_kw > 0:
                    arr_extra_import[k] = imp_kw
                    e_import_kwh += imp_kw * dt_hours
                    
                if soc_now >= soc_start or e_import_kwh >= e_vpp_kwh:
                    break
                    
            idx = end_idx + 1
        else:
            idx += 1
            
    return arr_extra_import


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
    
    df_res = df.copy()
    if 'price_import' in df_res.columns:
        df_res.rename(columns={'price_import': 'price_profile'}, inplace=True)

    # -------------------------------------------------------------
    # 2. Hitung Tarif Ekspor & Impor (Dipindah ke atas)
    # -------------------------------------------------------------
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
        
    # -------------------------------------------------------------
    # PERSIAPAN STRATEGI MODE BATERAI
    # -------------------------------------------------------------
    timestamps = df_res['timestamp']
    time_float = timestamps.dt.hour + timestamps.dt.minute / 60.0
    time_float = time_float.to_numpy(dtype=np.float64)
    
    # 1. Siapkan Semua Array Waktu untuk ToU
    is_offpeak = get_time_mask(time_float, params['t_offpeak_start'], params['t_offpeak_end'])
    is_peak    = get_time_mask(time_float, params['t_peak_start'], params['t_peak_end'])
    is_shoulder = get_time_mask(time_float, params['t_shoulder_start'], params['t_shoulder_end'])
    
    # 2. Siapkan Array Harga (VPP tetap pakai raw price, Arbitrase pakai Tariff Export Matang)
    arr_price_raw = df_res['price_profile'].to_numpy(dtype=np.float64)
    vpp_thresh = params['dispatch_price_threshold']
    is_vpp_arr = arr_price_raw >= vpp_thresh
    
    arr_price_export = df_res['tariff_export_AUD'].to_numpy(dtype=np.float64)
    
    # 3. Konversi Nama Skema
    scheme_name = params.get('tariff_scheme', 'Flat')
    if scheme_name == 'Time of Use':
        tariff_mode_int = 1
    elif scheme_name == 'Wholesale Price':
        tariff_mode_int = 2
    else:
        tariff_mode_int = 0
        
    # --- 4. Kalkulasi Baterai (Melempar arr_price_export yang sudah ada Market Fee) ---
    soc_pct, bat_power = simulate_battery_numba(
        net_load_pure,
        arr_price_export,
        is_offpeak,
        is_peak,         
        is_shoulder,      
        is_vpp_arr,
        tariff_mode_int,
        params['battery_capacity_kwh'],
        params['battery_initial_soc'],
        params['soc_min_pct'],
        params['soc_max_pct'],
        params['max_charge_kw'],
        params['max_discharge_kw'],
        params['battery_efficiency']
    )
    
    # -------------------------------------------------------------
    # PENGGABUNGAN HASIL BATERAI KE DATAFRAME
    # -------------------------------------------------------------
    df_res['solar_output_kw'] = solar_kw
    df_res['battery_power_ac_kw'] = bat_power
    df_res['battery_soc_pct'] = soc_pct
    
    # Hitung Grid Net 
    df_res['grid_net_kw'] = arr_load - solar_kw - bat_power
    df_res['vpp_status'] = is_vpp_arr
    
    # Hitung Kapasitas Baterai kWh
    df_res['battery_soc_kwh'] = (df_res['battery_soc_pct'] / 100.0) * params['battery_capacity_kwh']
        
    # =====================================================================
    # FINALISASI KALKULASI & EKONOMI VPP
    # =====================================================================
    dt_hours = 5.0 / 60.0
    
    # 1. Aliran Daya Dasar
    df_res['grid_import_kw'] = np.where(df_res['grid_net_kw'] > 0, df_res['grid_net_kw'], 0)
    df_res['grid_export_kw'] = np.where(df_res['grid_net_kw'] < 0, -df_res['grid_net_kw'], 0)
    df_res['vpp_charge'] = arr_price_raw < 0
    
    # 2. Akuntansi VPP Discharge
    df_res['vpp_battery_discharge_kw'] = np.where(df_res['vpp_status'] > 0, np.where(df_res['battery_power_ac_kw'] > 0, df_res['battery_power_ac_kw'], 0), 0)
    df_res['vpp_grid_export_kw'] = np.where(df_res['vpp_status'] > 0, df_res['grid_export_kw'], 0)
    
    # 3. Kalkulasi Extra Import Menggunakan Numba (Sangat Cepat)
    arr_soc_kwh = df_res['battery_soc_kwh'].to_numpy()
    arr_extra_import = calculate_extra_import_numba(
        df_res['vpp_status'].to_numpy() > 0, 
        df_res['battery_power_ac_kw'].to_numpy(), 
        df_res['grid_net_kw'].to_numpy(), 
        arr_soc_kwh, 
        dt_hours
    )
    df_res['vpp_grid_import_after_discharge_kw'] = arr_extra_import

    # 4. Kalkulasi Ekonomi (Financials)
    tariff_import = df_res.get('tariff_import_AUD', 0.20)
    tariff_export = df_res.get('tariff_export_AUD', 0.08)
    
    df_res['vpp_export_value_AUD'] = (df_res['vpp_grid_export_kw'] * dt_hours) * tariff_export
    df_res['vpp_extra_import_cost_AUD'] = (df_res['vpp_grid_import_after_discharge_kw'] * dt_hours) * tariff_import
    df_res['vpp_operational_net_value_AUD'] = df_res['vpp_export_value_AUD'] - df_res['vpp_extra_import_cost_AUD']
    
    # 5. Kalkulasi Perbandingan Tagihan (Bill Comparison)
    df_res['bill_actual'] = (df_res['grid_import_kw'] * dt_hours * tariff_import) - (df_res['grid_export_kw'] * dt_hours * tariff_export)
    
    # Skenario Solar Only
    col_load = 'load_profile' if 'load_profile' in df_res.columns else 'beban_rumah_kw'
    net_solar_only = df_res[col_load] - df_res['solar_output_kw']
    import_solar = np.where(net_solar_only > 0, net_solar_only, 0)
    export_solar = np.where(net_solar_only < 0, -net_solar_only, 0)
    df_res['bill_solar_only'] = (import_solar * dt_hours * tariff_import) - (export_solar * dt_hours * tariff_export)
    
    # Skenario Grid Only
    df_res['bill_grid_only'] = (df_res[col_load] * dt_hours) * tariff_import

    # =====================================================================
    # DAFTAR KOLOM FINAL (FINAL COLS)
    # =====================================================================
    final_cols = [
        'timestamp', 'irradiance', 'temperature', 'load_profile', 'price_profile', 
        'solar_output_kw', 'battery_soc_pct', 'battery_soc_kwh', 'battery_power_ac_kw', 
        'grid_net_kw', 'tariff_import_AUD', 'tariff_export_AUD',
        'vpp_status', 'vpp_charge', 'grid_import_kw', 'grid_export_kw',
        'vpp_battery_discharge_kw', 'vpp_grid_export_kw', 'vpp_grid_import_after_discharge_kw',
        'vpp_export_value_AUD', 'vpp_extra_import_cost_AUD', 'vpp_operational_net_value_AUD',
        'bill_actual', 'bill_solar_only', 'bill_grid_only'
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