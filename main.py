import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time as tm
import random
import json
import calendar

from datetime import time
from modules import loader, calculator
from modules import tariff_utils as t_utils
from modules import visualizer

def time_encoder(obj):
    """Mengubah object datetime.time menjadi string 'HH:MM' untuk JSON"""
    if isinstance(obj, time):
        return obj.strftime("%H:%M")
    raise TypeError("Type not serializable")

def apply_config(uploaded_file):
    """Membaca JSON dan update session_state"""
    if uploaded_file is not None:
        try:
            data = json.load(uploaded_file)
            for k, v in data.items():
                if (k.startswith("t_") or k.startswith("time_")) and isinstance(v, str):
                    h, m = map(int, v.split(':'))
                    st.session_state[k] = time(h, m)
                else:
                    st.session_state[k] = v
            st.success("Config Loaded! Rerunning...")
            tm.sleep(1)
            st.rerun()
        except Exception as e:
            st.error(f"Error loading config: {e}")




st.set_page_config(page_title="CER Simulation Data Generator", layout="wide")

if 'hasil_simulasi' not in st.session_state:
    st.session_state['hasil_simulasi'] = None
    st.session_state['used_params'] = {} 
    st.session_state['info_simulasi'] = ""

with st.sidebar:
    st.header("💾 Config Manager")
    st.markdown("Save your current setup or load existing one.")
    
    uploaded_file = st.file_uploader("📂 Load Config (JSON)", type=["json"])
    if uploaded_file:
        if st.button("Apply Loaded Config"):
            apply_config(uploaded_file)
    
    st.divider()
    if st.button("Prepare Config for Download"):
        config_data = {}
        exclude_keys = ['hasil_simulasi', 'used_params', 'info_simulasi']
        
        for key in st.session_state:
            if key not in exclude_keys and not key.startswith("FormSubmit"):
                config_data[key] = st.session_state[key]
        
        json_str = json.dumps(config_data, default=time_encoder, indent=2)
        
        st.download_button(
            label="📥 Download Config (.json)",
            data=json_str,
            file_name="simulation_config.json",
            mime="application/json"
        )



st.title("CER Simulation Data Generator")
st.markdown("Set parameter region and period to start generate data")
st.divider()

col_dp, col_spec = st.columns([1, 1], gap="medium")

with col_dp:
    st.subheader("📁 Data Parameter")
    
    col_location, col_tariff = st.columns([1, 1.4])

    with col_location:
        list_lokasi = loader.get_list_lokasi()
        if not list_lokasi:
            st.error("Database empty! Run script 'setup_database_v6.py first!")
            st.stop()
            
        st.info("🌍 Location")
        l1 , l2 = st.columns(2)

        selected_loc = l1.selectbox("1. Choose Region", list_lokasi, key="loc_region")
        
        list_titik = loader.get_list_titik(selected_loc)
        selected_point = l2.selectbox("2. Choose Point", list_titik, key="loc_point")
        
        available_years = loader.get_available_years(selected_loc, selected_point)
        
        st.info("🕒 Duration")
        if available_years:
            min_year, max_year = min(available_years), max(available_years)
            y1, y2 = st.columns(2)
            start_y = y1.selectbox("Start Date", available_years, index=0, key="date_start")
            
            valid_end_years = [y for y in available_years if y >= start_y]
            end_y = y2.selectbox("End Date", valid_end_years, index=len(valid_end_years)-1, key="date_end")
        else:
            st.warning("There is no data on this point!")
            st.stop()

    with col_tariff:
        st.info("⚙️ VPP Setting")
        vpp_price = st.number_input("Dispatch Price Threshold (AUD/MWh)", 0, 2000, 800, 10, key="vpp_threshold")

        st.info("💲 Tariff")

        st.text("Export")
        exp_price = st.number_input("Flat Price (AUD/kWh)", 0.0, 1.0, 0.08, 0.01, key="exp_tariff")

        st.text("Import")
        use_ToU = st.toggle("Flat / Time-Of-Use (ToU)", key="chk_tou")
        
        t_utils.initialize_session_state()
        
        if use_ToU:
            st.markdown("Peak Time")
            c1, c2, c3 = st.columns([1, 1, 1])
            c1.time_input("Start", key="t_p_start", value=st.session_state.t_p_start, on_change=t_utils.sync_peak_start)
            c2.time_input("End", key="t_p_end", value=st.session_state.t_p_end, on_change=t_utils.sync_peak_end)
            p_peak = c3.number_input("Price (AUD/kWh)", 0.0, 2.0, 0.45, 0.01, key="pp")

            st.markdown("Off-Peak")
            c1, c2, c3 = st.columns([1, 1, 1])
            c1.time_input("Start", key="t_o_start", value=st.session_state.t_o_start, on_change=t_utils.sync_offpeak_start, label_visibility="collapsed")
            c2.time_input("End", key="t_o_end", value=st.session_state.t_o_end, on_change=t_utils.sync_offpeak_end, label_visibility="collapsed")
            p_offpeak = c3.number_input("Price (AUD/kWh)", 0.0, 2.0, 0.15, 0.01, label_visibility="collapsed", key="po")

            st.markdown("Shoulder Time")
            c1, c2, c3 = st.columns([1, 1, 1])
            c1.time_input("Start", key="t_s_start", value=st.session_state.t_s_start, on_change=t_utils.sync_shoulder_start, label_visibility="collapsed")
            c2.time_input("End", key="t_s_end", value=st.session_state.t_s_end, on_change=t_utils.sync_shoulder_end, label_visibility="collapsed")
            p_shoulder = c3.number_input("Price (AUD/kWh)", 0.0, 2.0, 0.25, 0.01, label_visibility="collapsed", key="ps")
        else:
            p_flat = st.number_input("Flat Price (AUD/kWh)", 0.0, 2.0, 0.2, 0.01, key="imp_tariff")


with col_spec:
    st.subheader("⚙️ System Specification")
    
    col_panel, col_battery = st.columns(2)
    with col_panel:
        st.info("☀️ Solar Panel / Photovoltaics")
        use_rand_solar = st.toggle("Randomize / Fixed Size", key="chk_solar")
        if not use_rand_solar:
            sc1, sc2 = st.columns(2)
            p_solar_min = sc1.number_input("Min (kWp)", 0.0, 1000.0, 4.0, step=0.5, key="sol_min")
            p_solar_max = sc2.number_input("Max (kWp)", 0.0, 1000.0, 6.0, step=0.5, key="sol_max")
        else:
            p_solar_fix = st.number_input("Capacity (kWp)", 1.0, 100.0, 5.0, 0.5, key="sol_fix")

        p_temp = st.number_input("Temp Coeff", -0.01, 0.0, -0.004, 0.0001, format="%.4f", key="sol_temp")
        p_pr = st.number_input("PR (except temperature derated)", 0.5, 1.0, 0.8, 0.01, format="%.2f", key="sol_pr")
        
    with col_battery:
        st.info("🔋 Battery")
        use_rand_bat = st.toggle("Randomize / Fixed Size", key="chk_bat")
        if not use_rand_bat:
            bc1, bc2 = st.columns(2)
            p_bat_min = bc1.number_input("Min (kWh)", 0.0, 1000.0, 8.0, step=1.0, key="bat_min")
            p_bat_max = bc2.number_input("Max (kWh)", 0.0, 1000.0, 12.0, step=1.0, key="bat_max")
        else:
            p_bat_fix = st.number_input("Capacity (kWh)", 1.0, 200.0, 10.0, 1.0, key="bat_fix")

        b1, b2 = st.columns(2)
        p_charger_pwr = b1.number_input("(-) Charge (kW)", 0, 10, 5, 1, key="bat_chg")
        p_discharger_pwr = b2.number_input("(+) Discharge (kW)", 0, 10, 5, 1, key="bat_dis")
        p_eff = st.number_input("Round-Trip Efficiency (%)", 50, 100, 95, key="bat_eff") / 100
        p_soc = st.slider("Initial SoC (%)", 0, 100, 50, key="bat_soc_init") / 100
        range_soc = st.slider("SoC Constraint (%)", min_value=0, max_value=100, value=(10, 90), key="bat_soc_range")
        p_min_soc = range_soc[0] / 100
        p_max_soc = range_soc[1] / 100              

st.markdown("---")
btn_run = st.button("Process Parameter and Generate Data", type="primary", use_container_width=True)

if btn_run:
    if not use_rand_solar:
        final_p_solar = round(random.uniform(p_solar_min, p_solar_max), 2)
    else:
        final_p_solar = p_solar_fix
        
    if not use_rand_bat:
        final_p_bat = round(random.uniform(p_bat_min, p_bat_max), 2)
    else:
        final_p_bat = p_bat_fix

    with st.spinner(f"Combine data {selected_loc} ({selected_point}) dari {start_y}-{end_y}..."):
        df_input = loader.load_and_merge_data(selected_loc, selected_point, start_y, end_y)
        tm.sleep(0.5) 
    
    if df_input is not None:
        params = {
            'solar_capacity_kw': final_p_solar, 
            'temp_coeff': p_temp,
            'pr': p_pr,
            'battery_capacity_kwh': final_p_bat, 
            'battery_efficiency': p_eff,
            'battery_initial_soc': p_soc,
            'max_charge_kw': p_charger_pwr,
            'max_discharge_kw': p_discharger_pwr,
            'soc_min_pct': p_min_soc,
            'soc_max_pct': p_max_soc,
            'dispatch_price_threshold': vpp_price, 
            't_offpeak_start': st.session_state.t_o_start,
            't_offpeak_end': st.session_state.t_o_end,
            't_peak_start': st.session_state.t_p_start,
            't_peak_end': st.session_state.t_p_end
        }
        
        with st.spinner("Calculate Energy Flow..."):
            df_result = calculator.run_simulation(df_input, params)
        
        st.session_state['hasil_simulasi'] = df_result
        st.session_state['info_simulasi'] = f"{selected_loc}_{selected_point}_{start_y}-{end_y}"
        
        tariff_snapshot = {
            'is_tou': use_ToU,
            'export_price': exp_price
        }
        
        if use_ToU:
            tariff_snapshot.update({
                'peak_price': p_peak,
                'peak_start': st.session_state.t_p_start.strftime("%H:%M"),
                'peak_end': st.session_state.t_p_end.strftime("%H:%M"),
                'offpeak_price': p_offpeak,
                'offpeak_start': st.session_state.t_o_start.strftime("%H:%M"),
                'offpeak_end': st.session_state.t_o_end.strftime("%H:%M"),
                'shoulder_price': p_shoulder,
                'shoulder_start': st.session_state.t_s_start.strftime("%H:%M"),
                'shoulder_end': st.session_state.t_s_end.strftime("%H:%M"),
            })
        else:
            tariff_snapshot['import_flat'] = p_flat

        st.session_state['used_params'] = {
            'solar': final_p_solar, 
            'bat': final_p_bat,
            'vpp_thresh': vpp_price,
            'tariff_data': tariff_snapshot
        }
        
        st.success("Data Has Been Generated!")
    else:
        st.error("Failed to Generate the Data")

if st.session_state['hasil_simulasi'] is not None:
    
    df_result = st.session_state['hasil_simulasi']
    file_name_info = st.session_state['info_simulasi']
    used_p = st.session_state['used_params']
    t_data = used_p['tariff_data']

    st.divider()
    
    st.info(f"""
    ✅ **Generated Simulation Info:** Solar: {used_p['solar']} kWp | Battery: {used_p['bat']} kWh | VPP Threshold: {used_p['vpp_thresh']} AUD/MWh
    """)

    with st.expander("💲 View Applied Tariff Details", expanded=True):
        tc1, tc2 = st.columns(2)
        with tc1:
            st.markdown(f"**Export Tariff:**")
            st.markdown(f"⚡ Flat Rate: **{t_data['export_price']} AUD/kWh**")
        with tc2:
            st.markdown(f"**Import Tariff:**")
            if t_data['is_tou']:
                st.markdown("🕒 **Time-of-Use (ToU) Profile:**")
                st.markdown(f"""
                - **Peak:** {t_data['peak_price']} AUD <br> &nbsp;&nbsp;&nbsp; *({t_data['peak_start']} - {t_data['peak_end']})*
                - **Shoulder:** {t_data['shoulder_price']} AUD <br> &nbsp;&nbsp;&nbsp; *({t_data['shoulder_start']} - {t_data['shoulder_end']})*
                - **Off-Peak:** {t_data['offpeak_price']} AUD <br> &nbsp;&nbsp;&nbsp; *({t_data['offpeak_start']} - {t_data['offpeak_end']})*
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"🟦 Flat Rate: **{t_data['import_flat']} AUD/kWh**")

    st.markdown("### 💾 Export Data")
    csv = df_result.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Dataset (CSV)",
        data=csv,
        file_name=f"Data_{file_name_info}.csv",
        mime="text/csv",
        key='download-csv' 
    )

    st.divider()
    st.subheader("📊 Detailed Analysis")
    
    df_result['year']  = df_result['timestamp'].dt.year
    df_result['month'] = df_result['timestamp'].dt.month
    
    c_sel_1, c_sel_2 = st.columns(2)
    
    with c_sel_1:
        available_years_vis = sorted(df_result['year'].unique())
        selected_vis_year = st.selectbox("Select Year:", available_years_vis)
        df_vis_year = df_result[df_result['year'] == selected_vis_year].copy()

    with c_sel_2:
        available_months = sorted(df_vis_year['month'].unique())
        month_map = {m: calendar.month_name[m] for m in available_months}
        selected_month_name = st.selectbox("Select Month for Profile:", list(month_map.values()))
        selected_vis_month = [k for k, v in month_map.items() if v == selected_month_name][0]
        df_vis_month = df_vis_year[df_vis_year['month'] == selected_vis_month].copy()
    
    factor = 5/60
    
    col_load = 'load_profile' if 'load_profile' in df_vis_year.columns else 'beban_rumah_kw'
    col_bat  = 'battery_power_ac_kw' if 'battery_power_ac_kw' in df_vis_year.columns else 'battery_power_kw'
    
    total_solar = df_vis_year['solar_output_kw'].sum() * factor
    total_load  = df_vis_year[col_load].sum() * factor
    total_import = df_vis_year['grid_net_kw'].apply(lambda x: x if x > 0 else 0).sum() * factor
    
    m1, m2, m3 = st.columns(3)
    m1.metric(f"Total Solar ({selected_vis_year})", f"{total_solar:,.2f} kWh")
    m2.metric(f"Total Load ({selected_vis_year})", f"{total_load:,.2f} kWh")
    m3.metric(f"Grid Import ({selected_vis_year})", f"{total_import:,.2f} kWh", delta_color="inverse")

    visualizer.plot_annual_overview(df_vis_year, col_bat, selected_vis_year)
    st.divider()
    visualizer.plot_monthly_analysis(df_vis_month, col_load, selected_month_name, selected_vis_year)