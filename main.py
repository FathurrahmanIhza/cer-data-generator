import streamlit as st
import pandas as pd
import numpy as np
import time as tm
import random
import calendar
import math
import json

from datetime import time, datetime
from modules import loader, calculator
from modules import tariff_utils as t_utils
from modules import visualizer
from modules import config as cfg 
from modules import student_log as s_log
from st_aggrid import AgGrid, GridOptionsBuilder

st.set_page_config(page_title="CER Simulation Data Generator", layout="wide")

st.markdown(
    """
    <style>
    div[data-testid="stStatusWidget"] { visibility: hidden; }
    div[data-testid="stToastContainer"] { display: none; }
    div[data-testid="stSpinner"]:has(code) { display: none; }
    </style>
    """, unsafe_allow_html=True
)

cfg.init_default_states()

if 'app_initialized' not in st.session_state:
    st.session_state['app_initialized'] = True
    df_hist = cfg.load_config_history()
    if not df_hist.empty:
        latest_config = df_hist.iloc[0]
        cfg.apply_row_to_session(latest_config)
        st.session_state['active_config'] = latest_config['Config_Name']

if 'hasil_simulasi' not in st.session_state:
    st.session_state['hasil_simulasi'] = None
    st.session_state['used_params'] = {} 
    st.session_state['info_simulasi'] = ""

if 'role' not in st.session_state:
    st.session_state['role'] = 'student'

if st.query_params.get("admin") == "true":
    if st.session_state['role'] != 'admin':
        with st.sidebar:
            st.warning("🔒 Admin Access Required")
            pwd = st.text_input("Enter Password", type="password")
            
            if pwd == st.secrets["admin_password"]: 
                st.session_state['role'] = 'admin'
                
                df_hist = cfg.load_config_history()
                if not df_hist.empty:
                    active_cfg = st.session_state.get('active_config')
                    if active_cfg:
                        matched = df_hist[df_hist['Config_Name'] == active_cfg]
                        if not matched.empty:
                            cfg.apply_row_to_session(matched.iloc[0])
                        else:
                            cfg.apply_row_to_session(df_hist.iloc[0])
                    else:
                        cfg.apply_row_to_session(df_hist.iloc[0])
                st.rerun()
            elif pwd != "":
                st.error("Access Denied!")
        st.stop()


st.title("CER Simulation Data Generator")

btn_run = False 

if st.session_state['role'] == 'admin':
    
    with st.sidebar:
        st.header("☁️ Setup Config Manager")
        active_cfg = st.session_state.get('active_config', 'Default Config')
        st.success(f"**Active Config:** {active_cfg}")
        st.markdown("Save and Load configuration")
        
        st.subheader("📂 Load History Config")
        df_history = cfg.load_config_history()
        
        if not df_history.empty:
            history_options = df_history['Timestamp'].astype(str) + " | " + df_history['Config_Name'].astype(str)
            selected_history_str = st.selectbox("Select Config:", history_options.tolist())
            
            if st.button("Apply Config", width="stretch"):
                selected_row = df_history[history_options == selected_history_str].iloc[0]
                cfg.apply_row_to_session(selected_row)
                st.session_state['active_config'] = selected_row['Config_Name']
                st.success("✅ Config Applied! Rerunning...")
                st.rerun()
        else:
            st.info("No History Config Available.")
            
        st.divider()
        
        st.subheader("💾 Save Current Config")
        new_config_name = st.text_input("Config Name (ex: Exam Config 1)")
        
        if st.button("Save Config", type="primary", width="stretch"):
            if new_config_name.strip() == "":
                st.warning("⚠️ Empty Config Name")
            else:
                with st.spinner("Saving to Database..."):
                    success = cfg.save_config_to_sheets(new_config_name, st.session_state)
                    if success:
                        st.session_state['active_config'] = new_config_name
                        st.success("✅ Successfully Saved Config!")
                        tm.sleep(1)
                        st.rerun()
        st.divider()

    tab_config, tab_tracker = st.tabs(["⚙️ Config Manager", "👨‍🎓 Student Tracker"])

    
    with tab_config:
        st.markdown("Set region and period parameters to generate data")

        col_dp, col_spec = st.columns([1, 1], gap="medium")

        with col_dp:
            st.subheader("📁 Data Parameters")
            col_location, col_tariff = st.columns([1, 1.4])

            with col_location:
                list_lokasi = loader.get_list_lokasi()
                if not list_lokasi:
                    st.error("Database empty!")
                    st.stop()
                    
                st.info("🌍 Location")
                use_rand_location = st.toggle("Randomize / Fixed Location", key="chk_loc")
                
                selected_loc = None
                selected_point = None

                if use_rand_location:
                    l1, l2 = st.columns(2)
                    
                    saved_region = st.session_state.get('loc_region', list_lokasi[0])
                    idx_reg = list_lokasi.index(saved_region) if saved_region in list_lokasi else 0
                    
                    ui_region = l1.selectbox("1. Choose Region", list_lokasi, index=idx_reg, key="ui_loc_region")
                    st.session_state['loc_region'] = ui_region
                    selected_loc = ui_region
                    
                    list_titik = loader.get_list_titik(selected_loc)
                    
                    list_titik_extended = ["Randomize"] + list_titik 
                    
                    saved_point = st.session_state.get('loc_point', list_titik_extended[0])
                    idx_pt = list_titik_extended.index(saved_point) if saved_point in list_titik_extended else 0
                    
                    ui_point = l2.selectbox("2. Choose Point", list_titik_extended, index=idx_pt, key="ui_loc_point")
                    st.session_state['loc_point'] = ui_point 

                    if ui_point == "Randomize":
                        selected_point = random.choice(list_titik) if list_titik else None
                    else:
                        selected_point = ui_point
                        
                else: 
                    selected_loc = random.choice(list_lokasi)
                    list_titik_random = loader.get_list_titik(selected_loc)
                    selected_point = random.choice(list_titik_random) if list_titik_random else None
                


                available_years = loader.get_available_years(selected_loc, selected_point)
                
                st.info("🕒 Duration")
                if available_years:
                    use_rand_dur = st.toggle("Randomize / Fixed Duration", key="chk_dur")
                    
                    if use_rand_dur: 
                        y1, y2 = st.columns(2)
                        
                        saved_start = st.session_state.get('date_start', available_years[0])
                        idx_start = available_years.index(saved_start) if saved_start in available_years else 0
                        
                        ui_start_y = y1.selectbox(
                            "Start Date", 
                            available_years, 
                            index=idx_start, 
                            key="ui_date_start" 
                        )
                        st.session_state['date_start'] = ui_start_y # 
                        
                    
                        valid_end_years = [y for y in available_years if y >= ui_start_y]
                        
                        saved_end = st.session_state.get('date_end', valid_end_years[-1])
                        idx_end = valid_end_years.index(saved_end) if saved_end in valid_end_years else len(valid_end_years) - 1
                        
                        ui_end_y = y2.selectbox(
                            "End Date", 
                            valid_end_years, 
                            index=idx_end, 
                            key="ui_date_end" 
                        )
                        st.session_state['date_end'] = ui_end_y
                        
                    else: 
                        total_years = len(available_years)
                        
                        saved_rand_dur = st.session_state.get('rand_dur_years', 1)
                        saved_rand_dur = min(saved_rand_dur, total_years) 
                        
                        ui_dur = st.number_input(
                            f"Duration (Years)", 
                            min_value=1, 
                            max_value=total_years, 
                            value=int(saved_rand_dur), 
                            key="ui_rand_dur_years"
                        )
                        st.session_state['rand_dur_years'] = ui_dur
                        
                else:
                    st.warning("No data available for this location!")
                    st.stop()

                st.info("🏠 Load Profile")
                use_rand_load = st.toggle("Randomize / Fixed Load Profile", key="chk_load")
                selected_load_file = None 
                
                if use_rand_load: 
                    list_load_files = loader.get_list_load_profiles()
                    if list_load_files:
                        saved_file = st.session_state.get('sel_load_file', list_load_files[0])
                        idx = list_load_files.index(saved_file) if saved_file in list_load_files else 0
                        
                        ui_load_file = st.selectbox(
                            "Select Profile Source", 
                            list_load_files, 
                            index=idx,
                            key="ui_sel_load_file" 
                        )
                        st.session_state['sel_load_file'] = ui_load_file 
                        selected_load_file = ui_load_file
                        
                        saved_mult = st.session_state.get('load_mult', 15.0)
                        
                        ui_mult = st.slider(
                            "Load Multiplier", 
                            min_value=8.0, 
                            max_value=32.0, 
                            value=float(saved_mult), 
                            step=0.1, 
                            key="ui_load_mult" 
                        )
                        st.session_state['load_mult'] = ui_mult
                    else:
                        st.error("No Parquet/CSV files found!")
                        st.stop()
                

            with col_tariff:
                st.info("⚙️ VPP Settings")
                vpp_price = st.number_input("Dispatch Price Threshold (AUD/MWh)", 0, 2000, step=10, key="vpp_threshold")

                st.info("💲 Tariff")
                list_scheme = ["Flat", "Time of Use", "Wholesale Price", "Random"]
                
                def _sync_scheme():
                    st.session_state['tariff_scheme'] = st.session_state['ui_tariff_scheme']

                saved_scheme = st.session_state.get('tariff_scheme', 'Flat')
                if saved_scheme not in list_scheme: 
                    saved_scheme = 'Flat'
                
                if "ui_tariff_scheme" not in st.session_state:
                    st.session_state["ui_tariff_scheme"] = saved_scheme
                
                ui_scheme = st.selectbox(
                    "Select Tariff Scheme", 
                    list_scheme, 
                    key="ui_tariff_scheme", 
                    on_change=_sync_scheme,
                    label_visibility="collapsed"
                )
                
                t_utils.initialize_session_state()

                if ui_scheme == "Flat":
                    st.markdown("**💲 Set Prices (AUD/kWh)**")
                    c1, c2 = st.columns(2)
                    
                    def _sync_flat_imp(): st.session_state['imp_tariff'] = st.session_state['ui_imp_tariff']
                    def _sync_flat_exp(): st.session_state['exp_tariff'] = st.session_state['ui_exp_tariff']
                    
                    c1.number_input("Import", 0.0, 2.0, step=0.01, key="ui_imp_tariff", 
                                    value=float(st.session_state.get('imp_tariff', 0.20)), on_change=_sync_flat_imp)
                    c2.number_input("Export", 0.0, 1.0, step=0.01, key="ui_exp_tariff", 
                                    value=float(st.session_state.get('exp_tariff', 0.08)), on_change=_sync_flat_exp)
                
                elif ui_scheme == "Time of Use":
                    st.markdown("**🕒 Set Time Periods**")
                    
                    def _sync_t_p_start():
                        st.session_state['t_p_start'] = st.session_state['ui_t_p_start']
                        t_utils.sync_peak_start()
                    def _sync_t_p_end():
                        st.session_state['t_p_end'] = st.session_state['ui_t_p_end']
                        t_utils.sync_peak_end()
                    def _sync_t_o_start():
                        st.session_state['t_o_start'] = st.session_state['ui_t_o_start']
                        t_utils.sync_offpeak_start()
                    def _sync_t_o_end():
                        st.session_state['t_o_end'] = st.session_state['ui_t_o_end']
                        t_utils.sync_offpeak_end()
                    def _sync_t_s_start():
                        st.session_state['t_s_start'] = st.session_state['ui_t_s_start']
                        t_utils.sync_shoulder_start()
                    def _sync_t_s_end():
                        st.session_state['t_s_end'] = st.session_state['ui_t_s_end']
                        t_utils.sync_shoulder_end()
                    
                    st.markdown("Peak Time")
                    c1, c2 = st.columns(2)
                    c1.time_input("Start", key="ui_t_p_start", value=st.session_state.get('t_p_start', time(19,0)), on_change=_sync_t_p_start)
                    c2.time_input("End", key="ui_t_p_end", value=st.session_state.get('t_p_end', time(23,0)), on_change=_sync_t_p_end)

                    st.markdown("Off-Peak Time")
                    c1, c2 = st.columns(2)
                    c1.time_input("Start", key="ui_t_o_start", value=st.session_state.get('t_o_start', time(23,0)), on_change=_sync_t_o_start, label_visibility="collapsed")
                    c2.time_input("End", key="ui_t_o_end", value=st.session_state.get('t_o_end', time(7,0)), on_change=_sync_t_o_end, label_visibility="collapsed")

                    st.markdown("Shoulder Time")
                    c1, c2 = st.columns(2)
                    c1.time_input("Start", key="ui_t_s_start", value=st.session_state.get('t_s_start', time(7,0)), on_change=_sync_t_s_start, label_visibility="collapsed")
                    c2.time_input("End", key="ui_t_s_end", value=st.session_state.get('t_s_end', time(19,0)), on_change=_sync_t_s_end, label_visibility="collapsed")

                    st.markdown("**💲 Set Prices (AUD/kWh)**")
                    cp1, cp2 = st.columns(2)
                    
                    def _sync_pp(): st.session_state['pp'] = st.session_state['ui_pp']
                    def _sync_po(): st.session_state['po'] = st.session_state['ui_po']
                    def _sync_ps(): st.session_state['ps'] = st.session_state['ui_ps']
                    def _sync_ep(): st.session_state['e_peak'] = st.session_state['ui_e_peak']
                    def _sync_eo(): st.session_state['e_offpeak'] = st.session_state['ui_e_offpeak']
                    def _sync_es(): st.session_state['e_shoulder'] = st.session_state['ui_e_shoulder']
                    
                    with cp1:
                        st.markdown("Import")
                        st.number_input("Peak", 0.0, 2.0, step=0.01, key="ui_pp", value=float(st.session_state.get('pp', 0.45)), on_change=_sync_pp)
                        st.number_input("Off-Peak", 0.0, 2.0, step=0.01, key="ui_po", value=float(st.session_state.get('po', 0.15)), on_change=_sync_po)
                        st.number_input("Shoulder", 0.0, 2.0, step=0.01, key="ui_ps", value=float(st.session_state.get('ps', 0.25)), on_change=_sync_ps)
                    with cp2:
                        st.markdown("Export")
                        st.number_input("Peak", 0.0, 2.0, step=0.01, key="ui_e_peak", value=float(st.session_state.get('e_peak', 0.15)), on_change=_sync_ep)
                        st.number_input("Off-Peak", 0.0, 2.0, step=0.01, key="ui_e_offpeak", value=float(st.session_state.get('e_offpeak', 0.05)), on_change=_sync_eo)
                        st.number_input("Shoulder", 0.0, 2.0, step=0.01, key="ui_e_shoulder", value=float(st.session_state.get('e_shoulder', 0.10)), on_change=_sync_es)

                elif ui_scheme == "Wholesale Price":
                    st.info("- **Import:** Spot Price + Market + Network + Other Fees\n- **Export:** Spot Price + Market Fees")
                
                elif ui_scheme == "Random":
                    st.info("The simulation will randomly select between Flat, Time of Use, or Wholesale Price.\n")

        with col_spec:
            st.subheader("⚙️ System Specifications")
            
            col_panel, col_battery = st.columns(2)
            with col_panel:
                st.info("☀️ Solar Panel / Photovoltaics")
                use_rand_solar = st.toggle("Randomize / Fixed Size", key="chk_solar")
                if not use_rand_solar:
                    sc1, sc2 = st.columns(2)
                    p_solar_min = sc1.number_input("Min (kWp)", 0.0, 1000.0, step=0.5, key="sol_min")
                    p_solar_max = sc2.number_input("Max (kWp)", 0.0, 1000.0, step=0.5, key="sol_max")
                else:
                    p_solar_fix = st.number_input("Capacity (kWp)", 1.0, 100.0, step=0.5, key="sol_fix")

                p_temp = st.number_input("Temp Coeff", -0.01, 0.0, step=0.0001, format="%.4f", key="sol_temp")
                p_pr = st.number_input("PR (except temperature derated)", 0.5, 1.0, step=0.01, format="%.2f", key="sol_pr")
                
            with col_battery:
                st.info("🔋 Battery")
                use_rand_bat = st.toggle("Randomize / Fixed Size", key="chk_bat")
                if not use_rand_bat:
                    bc1, bc2 = st.columns(2)
                    p_bat_min = bc1.number_input("Min (kWh)", 0.0, 1000.0, step=0.5, key="bat_min")
                    p_bat_max = bc2.number_input("Max (kWh)", 0.0, 1000.0, step=0.5, key="bat_max")
                else:
                    p_bat_fix = st.number_input("Capacity (kWh)", 1.0, 200.0, step=0.5, key="bat_fix")
                
                p_eff = st.number_input("Round-Trip Efficiency (%)", 50, 100, key="bat_eff") / 100
                p_soc = st.slider("Initial SoC (%)", 0, 100, key="bat_soc_init") / 100
                range_soc = st.slider("SoC Constraint (%)", min_value=0, max_value=100, key="bat_soc_range")
                p_min_soc = range_soc[0] / 100
                p_max_soc = range_soc[1] / 100   

        st.markdown("---")
        btn_run = st.button("Generate Data", type="primary", width="stretch", key="btn_admin")
        res_container = st.container()

    with tab_tracker:
        
        # Membungkus UI Tracker ke dalam Fragment agar tidak refresh 1 halaman penuh
        @st.fragment
        def tracker_ui():
            df_logs = s_log.get_student_logs()
            
            if df_logs.empty:
                st.info("There is no Data Available.")
                return 
                
            st.markdown("### 📋 Student Generate Tracker")
            
            df_logs = df_logs.sort_values(by="Timestamp", ascending=False).reset_index(drop=True)
            df_logs.index = df_logs.index + 1
            df_logs.reset_index(inplace=True)
            df_logs['NIM'] = df_logs['NIM'].astype(str).str.replace(r'\.0$', '', regex=True)
            
            df_logs.rename(columns={
                'index': 'No', 
                'Timestamp': 'Timestamp (UTC)',
                'NIM': 'Student ID',
                'Config_Name': 'Parameter Used'
            }, inplace=True)
            
            def extract_summary(json_str):
                try:
                    p = json.loads(json_str) 
                    return f"Loc: {p.get('location','')} | PV: {p.get('solar','')}kWp | Bat: {p.get('bat','')}kWh | Load: {p.get('load_source','')}"
                except:
                    return "Invalid Data"
                    
            df_logs['Result Parameter'] = df_logs['Parameter_Snapshot'].apply(extract_summary)

            gb = GridOptionsBuilder.from_dataframe(df_logs)
            
            gb.configure_column("Parameter_Snapshot", hide=True)
            gb.configure_column("id", hide=True)
            gb.configure_column("created_at", hide=True)
            
            gb.configure_default_column(resizable=True, filterable=True, sortable=True)
            
            gb.configure_column("No", minWidth=60, maxWidth=80, filter='agNumberColumnFilter')
            
            gb.configure_column("Timestamp (UTC)", minWidth=160, flex=1, filter='agTextColumnFilter')
            gb.configure_column("Student ID", minWidth=130, flex=1, filter='agTextColumnFilter')
            gb.configure_column("Parameter Used", minWidth=150, flex=1, filter='agTextColumnFilter')
            gb.configure_column("Result Parameter", minWidth=300, flex=2, wrapText=True, autoHeight=True, filter='agTextColumnFilter')
            
            gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=10)
            gb.configure_selection('single', use_checkbox=True)
            
            gridOptions = gb.build()
            
            grid_response = AgGrid(
                df_logs,
                gridOptions=gridOptions,
                update_on=['selectionChanged'], 
                fit_columns_on_grid_load=False, 
                theme='streamlit', 
                height=375
            )
            
            st.divider()
            
            selected_rows = grid_response['selected_rows']
            
            if selected_rows is not None and len(selected_rows) > 0:
                
                if isinstance(selected_rows, pd.DataFrame):
                    sel_dict = selected_rows.iloc[0].to_dict()
                else:
                    sel_dict = selected_rows[0]
                    
                nim_target = sel_dict['Student ID']
                st.info(f"📌 Selected Data — Student ID: **{nim_target}** | Parameter Used: **{sel_dict['Parameter Used']}**")
                
                if st.button("Re-generate Data", width="stretch", type="primary", key="btn_regen_tracker"):
                    try:
                        saved_params = json.loads(sel_dict['Parameter_Snapshot'])
                        with st.spinner(f"Re-generating data for Student ID {nim_target}..."):
                            
                            # --- SET SEED ULANG ---
                            config_used = sel_dict['Parameter Used'] 
                            seed_val = s_log.generate_seed(nim_target, config_used)
                            
                            random.seed(seed_val)
                            np.random.seed(seed_val)
                            
                            # --- SAFE SPLIT LOCATION ---
                            loc_split = saved_params['location'].split(" - ") 
                            reg = loc_split[0].strip()
                            pt = " - ".join(loc_split[1:]).strip() 
                            
                            yr_split = str(saved_params['period']).split(" to ")
                            sy = int(yr_split[0])
                            ey = int(yr_split[1]) if len(yr_split) > 1 else sy
                            
                            df_input_regen = loader.load_and_merge_data(
                                reg, pt, sy, ey, fixed_load_file=saved_params['load_source']
                            )
                            
                            if df_input_regen is None:
                                st.error(f"❌ Dataset Failed to Load! Check Folder 'dataset/{reg}/{pt}'")
                            else:
                                col_load_regen = 'load_profile' if 'load_profile' in df_input_regen.columns else 'beban_rumah_kw'
                                df_input_regen[col_load_regen] = df_input_regen[col_load_regen] * saved_params['load_multiplier']

                                # Di dalam btn_regen_tracker
                                sim_params = {
                                    'solar_capacity_kw': saved_params['solar'], 
                                    'temp_coeff': saved_params['solar_temp'],
                                    'pr': saved_params['solar_pr'],
                                    'battery_capacity_kwh': saved_params['bat'], 
                                    'battery_efficiency': saved_params['bat_eff'],
                                    'battery_initial_soc': saved_params['bat_soc_init'],
                                    'max_charge_kw': saved_params['bat_charge_kw'],
                                    'max_discharge_kw': saved_params['bat_discharge_kw'],
                                    'soc_min_pct': saved_params['soc_min'],
                                    'soc_max_pct': saved_params['soc_max'],
                                    'dispatch_price_threshold': saved_params['vpp_thresh'],
                                    'df_wholesale_fees': loader.get_wholesale_fees(reg), 
                                }
                                
                                t_data = saved_params['tariff_data']
                                sim_params['tariff_scheme'] = t_data.get('tariff_scheme', 'Flat')

                                sim_params.update({
                                    't_peak_start': time(17, 0),
                                    't_peak_end': time(20, 0),
                                    't_offpeak_start': time(22, 0),
                                    't_offpeak_end': time(6, 0),
                                    't_shoulder_start': time(14, 0),
                                    't_shoulder_end': time(17, 0)
                                })

                                if sim_params['tariff_scheme'] == "Time of Use":
                                    sim_params.update({
                                        'peak_price': t_data['peak_price'],
                                        'exp_peak': t_data['exp_peak'],
                                        't_peak_start': datetime.strptime(t_data['peak_start'], "%H:%M").time(),
                                        't_peak_end': datetime.strptime(t_data['peak_end'], "%H:%M").time(),
                                        'offpeak_price': t_data['offpeak_price'],
                                        'exp_offpeak': t_data['exp_offpeak'],
                                        't_offpeak_start': datetime.strptime(t_data['offpeak_start'], "%H:%M").time(),
                                        't_offpeak_end': datetime.strptime(t_data['offpeak_end'], "%H:%M").time(),
                                        'shoulder_price': t_data['shoulder_price'],
                                        'exp_shoulder': t_data['exp_shoulder'],
                                        't_shoulder_start': datetime.strptime(t_data['shoulder_start'], "%H:%M").time(),
                                        't_shoulder_end': datetime.strptime(t_data['shoulder_end'], "%H:%M").time(),
                                    })
                                elif sim_params['tariff_scheme'] == "Flat":
                                    sim_params['import_flat'] = t_data.get('import_flat', 0.20)
                                    sim_params['export_price'] = t_data.get('export_price', 0.08)
                                    
                                df_result_regen = calculator.run_simulation(df_input_regen, sim_params)
                                
                                df_export = df_result_regen.round(2).rename(columns={
                                    'irradiance': 'irradiance_Wh/m^2',
                                    'temperature': 'temperature_C',
                                    'load_profile': 'load_kW',
                                    'price_profile': 'price_AUD',
                                    'battery_soc_pct': 'battery_soc_%'
                                })
                                
                                st.session_state['regen_csv_data'] = df_export.to_csv(index=False).encode('utf-8')
                                st.session_state['regen_nim'] = nim_target
                                st.session_state['regen_reg'] = reg
                                st.session_state['regen_pt'] = pt
                                st.session_state['regen_params'] = saved_params

                    except Exception as e:
                        st.error(f"Failed To Process Data: {e}")
            else:
                st.info("Select one of the rows to Re-generate the Data.")
                
            if st.session_state.get('regen_csv_data') is not None:
                st.success(f"✅ Data Has Been Re-generated!")
                
                used_p = st.session_state['regen_params']
                t_data = used_p['tariff_data']

                st.divider()
                st.markdown("### 📋 Generated Simulation Info")
                
                with st.container(border=True):
                    st.markdown(f"**📍 Location:** `{used_p['location']}` | **🗓️ Period:** `{used_p['period']}` | **🏠 Load:** `{used_p['load_source']}` **(x {used_p.get('load_multiplier', 1.0)})**")
                    st.divider()
                    
                    c_sys1, c_sys2, c_sys3 = st.columns(3)
                    
                    with c_sys1:
                        st.markdown("#### ☀️ Solar PV")
                        st.markdown(f"""
                        - Capacity: **{used_p['solar']} kWp**
                        - PR: **{used_p['solar_pr']}**
                        - Temp Coeff: **{used_p['solar_temp']}**
                        """)
                        
                    with c_sys2:
                        st.markdown("#### 🔋 Battery Storage")
                        st.markdown(f"""
                        - Capacity: **{used_p['bat']} kWh**
                        - Power: **-{used_p['bat_charge_kw']} / +{used_p['bat_discharge_kw']} kW**
                        - Efficiency: **{int(used_p['bat_eff']*100)}%**
                        """)
                        
                    with c_sys3:
                        st.markdown("#### ⚡ Control Logic")
                        st.markdown(f"""
                        - VPP Threshold: **{used_p['vpp_thresh']} AUD**
                        - SoC Limits: **{int(used_p['soc_min']*100)}% - {int(used_p['soc_max']*100)}%**
                        - Initial SoC: **{int(used_p['bat_soc_init']*100)}%**
                        """)

                with st.expander("💲 View Applied Tariff Details", expanded=False):
                    schema_name = t_data.get('tariff_scheme', "Flat")
                    st.markdown(f"**⚡ Scheme:** `{schema_name}`")
                    
                    if schema_name == "Wholesale Price":
                        st.markdown("- **Import:** Spot Price + Market + Network + Other Fees\n- **Export:** Spot Price + Market Fees")
                    else:
                        tc1, tc2 = st.columns(2)
                        with tc1:
                            st.markdown(f"**Export Tariff:**")
                            if schema_name == "Time of Use":
                                st.markdown(f"- Peak: **{t_data.get('exp_peak', 0.15)} AUD**\n- Shoulder: **{t_data.get('exp_shoulder', 0.10)} AUD**\n- Off-Peak: **{t_data.get('exp_offpeak', 0.05)} AUD**")
                            else:
                                st.markdown(f"Flat Rate: **{t_data.get('export_price', 0.08)} AUD/kWh**")
                        with tc2:
                            st.markdown(f"**Import Tariff:**")
                            if schema_name == "Time of Use":
                                st.markdown(f"- Peak: **{t_data.get('peak_price', 0.45)} AUD**\n- Shoulder: **{t_data.get('shoulder_price', 0.25)} AUD**\n- Off-Peak: **{t_data.get('offpeak_price', 0.15)} AUD**")
                            else:
                                st.markdown(f"Flat Rate: **{t_data.get('import_flat', 0.20)} AUD/kWh**")
                
                st.markdown("### 💾 Export Data")
                st.download_button(
                    label=f"Download Dataset (CSV)",
                    data=st.session_state['regen_csv_data'],
                    file_name=f"Data_{st.session_state['regen_nim']}_{st.session_state['regen_reg']}_{st.session_state['regen_pt']}.csv",
                    mime="text/csv",
                    key=f"dl_regen_{st.session_state['regen_nim']}" 
                )
                
        tracker_ui()

else :
    
    active_cfg = st.session_state.get('active_config', 'Default')
    st.info(f"👋 **Welcome!**  \n\nClick the button below to generate your dataset.")
            
    student_nim = st.text_input("Student ID", placeholder="eg: z5593968").strip()
    st.session_state['current_nim'] = student_nim

    st.markdown("---")
    btn_run = st.button("Generate Data", type="primary", width="stretch", key="btn_student")
    res_container = st.container()


if btn_run:
    if st.session_state['role'] == 'student':
        if not st.session_state.get('current_nim'):
            st.warning("⚠️ Please Enter Your Student ID!")
            st.stop()

        df_hist = cfg.load_config_history()
        if not df_hist.empty:
            active_cfg = st.session_state.get('active_config')
            matched = df_hist[df_hist['Config_Name'] == active_cfg] if active_cfg else pd.DataFrame()
            if not matched.empty:
                cfg.apply_row_to_session(matched.iloc[0])
            else:
                cfg.apply_row_to_session(df_hist.iloc[0])
        
        active_cfg_name = st.session_state.get('active_config', 'Default')
        seed_val = s_log.generate_seed(st.session_state['current_nim'], active_cfg_name)
        
        random.seed(seed_val)
        np.random.seed(seed_val)
    else:
        random.seed()
        np.random.seed()

    # AMBIL PARAMETER DARI SESSION STATE
    use_rand_location = st.session_state.get('chk_loc', False)
    use_rand_load = st.session_state.get('chk_load', False)
    use_rand_solar = st.session_state.get('chk_solar', False)
    use_rand_bat = st.session_state.get('chk_bat', False)

    p_solar_min = st.session_state.get('sol_min', 4.0)
    p_solar_max = st.session_state.get('sol_max', 6.0)
    p_solar_fix = st.session_state.get('sol_fix', 5.0)
    p_temp = st.session_state.get('sol_temp', -0.004)
    p_pr = st.session_state.get('sol_pr', 0.8)

    p_bat_min = st.session_state.get('bat_min', 8.0)
    p_bat_max = st.session_state.get('bat_max', 12.0)
    p_bat_fix = st.session_state.get('bat_fix', 10.0)
    p_eff = st.session_state.get('bat_eff', 95) / 100
    p_soc = st.session_state.get('bat_soc_init', 50) / 100
    range_soc = st.session_state.get('bat_soc_range', (10, 90))
    p_min_soc = range_soc[0] / 100
    p_max_soc = range_soc[1] / 100

    vpp_price = st.session_state.get('vpp_threshold', 800)
    tariff_scheme = st.session_state.get('tariff_scheme', 'Flat')
    if tariff_scheme == "Random":
        tariff_scheme = random.choice(["Flat", "Time of Use", "Wholesale Price"])
    exp_price = st.session_state.get('exp_tariff', 0.08)
    p_flat = st.session_state.get('imp_tariff', 0.20)
    p_peak = st.session_state.get('pp', 0.45)
    p_offpeak = st.session_state.get('po', 0.15)
    p_shoulder = st.session_state.get('ps', 0.25)
    e_peak = st.session_state.get('e_peak', 0.15)
    e_offpeak = st.session_state.get('e_offpeak', 0.05)
    e_shoulder = st.session_state.get('e_shoulder', 0.10)

    selected_load_file = st.session_state.get('sel_load_file', None)
    # start_y = st.session_state.get('date_start', 2020)
    # end_y = st.session_state.get('date_end', 2020)

    # --- KALKULASI LOKASI ---
    if use_rand_location: 
        selected_loc = st.session_state.get('loc_region')
        raw_point = st.session_state.get('loc_point')
        
        if raw_point == "Randomize":
            list_titik = loader.get_list_titik(selected_loc)
            selected_point = random.choice(list_titik) if list_titik else None
        else:
            selected_point = raw_point
            
    else: 
        list_lokasi = loader.get_list_lokasi()
        selected_loc = random.choice(list_lokasi)
        list_titik_random = loader.get_list_titik(selected_loc)
        selected_point = random.choice(list_titik_random) if list_titik_random else None
    

    # --- KALKULASI DURASI ---
    use_rand_dur = st.session_state.get('chk_dur', False)
    
    if use_rand_dur: 
        final_start_y = st.session_state.get('date_start', 2020)
        final_end_y = st.session_state.get('date_end', 2020)
    else: 
        actual_years = loader.get_available_years(selected_loc, selected_point)
        
        if actual_years:
            dur_req = st.session_state.get('rand_dur_years', 1)
            dur_req = min(dur_req, len(actual_years)) 
            
            max_start_idx = len(actual_years) - dur_req
            
            rand_idx = random.randint(0, max_start_idx)
            
            final_start_y = actual_years[rand_idx]
            final_end_y = actual_years[rand_idx + dur_req - 1]
        else:
            final_start_y, final_end_y = 2020, 2020


    # --- KALKULASI BEBAN ---
    all_files = loader.get_list_load_profiles()
    
    if use_rand_load: 
        final_load_file = selected_load_file
        final_load_mult = st.session_state.get('load_mult', 15.0)
    else: 
        if all_files:
            final_load_file = random.choice(all_files)
            final_load_mult = round(random.uniform(8.0, 32.0), 1)
        else:
            st.error("❌ No load profile files found!")
            st.stop()

    # --- KALKULASI SOLAR ---
    is_solar_fixed = False 
    if not use_rand_solar:
        segment_solar = 5
        solar_total_range = p_solar_max - p_solar_min
        solar_segment_width = solar_total_range / segment_solar
        
        if final_load_mult < 16.0:
            start_seg_solar = 0
            end_seg_solar = 2
        elif final_load_mult < 24.0:
            start_seg_solar = 1
            end_seg_solar = 3
        else:
            start_seg_solar = 2
            end_seg_solar = 4

        final_solar_min = p_solar_min + (start_seg_solar * solar_segment_width)
        final_solar_max = p_solar_min + ((end_seg_solar + 1) * solar_segment_width)

        raw_solar = random.uniform(final_solar_min, final_solar_max)
        
        final_p_solar = round(raw_solar * 2) / 2
    else:
        final_p_solar = round(p_solar_fix * 2) / 2
        is_solar_fixed = True

    # --- KALKULASI BATERAI ---
    if not use_rand_bat:
        segment = 5
        bat_total_range = p_bat_max - p_bat_min
        bat_segment_width = bat_total_range / segment

        if is_solar_fixed:
            mid = (segment - 1) // 2
            start_seg = max(0, mid - 1)
            end_seg   = min(segment - 1, mid + 1)
        else:
            solar_range = p_solar_max - p_solar_min
            if solar_range <= 0:
                current_segment = (segment - 1) // 2
            else:
                relative_pos = (final_p_solar - p_solar_min) / solar_range
                raw_segment = int(relative_pos * segment)
                current_segment = max(0, min(segment - 1, raw_segment))

            start_seg = max(0, current_segment - 1)
            end_seg   = min(segment - 1, current_segment + 1)

        final_bat_min = p_bat_min + (start_seg * bat_segment_width)
        final_bat_max = p_bat_min + ((end_seg + 1) * bat_segment_width)

        raw_bat = random.uniform(final_bat_min, final_bat_max)
        final_p_bat = round(raw_bat * 2) / 2
    else:
        final_p_bat = p_bat_fix

    bat_total_range = p_bat_max - p_bat_min
    
    if bat_total_range <= 0:
        bat_segment_idx = 2 
    else:
        bat_segment_width = bat_total_range / 5
        bat_segment_idx = int((final_p_bat - p_bat_min) / bat_segment_width)
        bat_segment_idx = max(0, min(4, bat_segment_idx))

    if bat_segment_idx == 0:
        auto_charge_power = 5.0
    elif bat_segment_idx in [1, 2]:
        auto_charge_power = 10.0
    else:
        auto_charge_power = 15.0

    st.toast(f"📄 Load Profile: {final_load_file}")
    with st.spinner(f"Combining data for {selected_loc} ({selected_point}) from {final_start_y}-{final_end_y}..."):
        df_input = loader.load_and_merge_data(
            selected_loc, 
            selected_point, 
            final_start_y, 
            final_end_y, 
            fixed_load_file=final_load_file 
        )
        tm.sleep(0.5) 
    
    if df_input is not None:
        col_load_name = 'load_profile' if 'load_profile' in df_input.columns else 'beban_rumah_kw'
        df_input[col_load_name] = df_input[col_load_name] * final_load_mult
        params = {
            'solar_capacity_kw': final_p_solar, 
            'temp_coeff': p_temp,
            'pr': p_pr,
            'battery_capacity_kwh': final_p_bat, 
            'battery_efficiency': p_eff,
            'battery_initial_soc': p_soc,
            'max_charge_kw': auto_charge_power,
            'max_discharge_kw': auto_charge_power,
            'soc_min_pct': p_min_soc,
            'soc_max_pct': p_max_soc,
            'dispatch_price_threshold': vpp_price, 
            't_offpeak_start': st.session_state.get('t_o_start', time(22,0)),
            't_offpeak_end': st.session_state.get('t_o_end', time(6,0)),
            't_peak_start': st.session_state.get('t_p_start', time(17,0)),
            't_peak_end': st.session_state.get('t_p_end', time(20,0)),
            't_shoulder_start': st.session_state.get('t_s_start', time(14,0)),
            't_shoulder_end': st.session_state.get('t_s_end', time(17,0)),
            'tariff_scheme': tariff_scheme,
            'df_wholesale_fees': loader.get_wholesale_fees(selected_loc),
            'export_price': exp_price,
            'import_flat': p_flat,
            'peak_price': p_peak,
            'offpeak_price': p_offpeak,
            'shoulder_price': p_shoulder,
            'exp_peak': e_peak,
            'exp_offpeak': e_offpeak,
            'exp_shoulder': e_shoulder
        }
        
        with st.spinner("Calculating Energy Flow..."):
            df_result = calculator.run_simulation(df_input, params)
        
        st.session_state['hasil_simulasi'] = df_result
        st.session_state['info_simulasi'] = f"{selected_loc}_{selected_point}_{final_start_y}-{final_end_y}"
        
        # Susun Snapshot Tarif
        tariff_snapshot = {'tariff_scheme': tariff_scheme}
        
        if tariff_scheme == "Time of Use":
            tariff_snapshot.update({
                'peak_price': p_peak, 'exp_peak': e_peak,
                'peak_start': st.session_state.get('t_p_start', time(17,0)).strftime("%H:%M"),
                'peak_end': st.session_state.get('t_p_end', time(20,0)).strftime("%H:%M"),
                'offpeak_price': p_offpeak, 'exp_offpeak': e_offpeak,
                'offpeak_start': st.session_state.get('t_o_start', time(22,0)).strftime("%H:%M"),
                'offpeak_end': st.session_state.get('t_o_end', time(6,0)).strftime("%H:%M"),
                'shoulder_price': p_shoulder, 'exp_shoulder': e_shoulder,
                'shoulder_start': st.session_state.get('t_s_start', time(14,0)).strftime("%H:%M"),
                'shoulder_end': st.session_state.get('t_s_end', time(17,0)).strftime("%H:%M"),
            })
        elif tariff_scheme == "Flat":
            tariff_snapshot['import_flat'] = p_flat
            tariff_snapshot['export_price'] = exp_price

        st.session_state['used_params'] = {
            'solar': final_p_solar,
            'solar_pr': p_pr,
            'solar_temp': p_temp,
            'bat': final_p_bat,
            'bat_eff': p_eff,
            'bat_soc_init': p_soc,
            'bat_charge_kw': auto_charge_power,
            'bat_discharge_kw': auto_charge_power,
            'soc_min': p_min_soc,
            'soc_max': p_max_soc,
            'vpp_thresh': vpp_price,
            'tariff_data': tariff_snapshot,
            'location': f"{selected_loc} - {selected_point}",
            'period': f"{final_start_y}" if final_start_y == final_end_y else f"{final_start_y} to {final_end_y}",
            'load_source': final_load_file,
            'load_multiplier': final_load_mult 
        }
        
        if st.session_state['role'] == 'student':
            active_cfg_name = st.session_state.get('active_config', 'Default')
            s_log.save_log_to_sheets(
                st.session_state['current_nim'], 
                active_cfg_name, 
                st.session_state['used_params']
            )
        
        with res_container: 
            st.success(f"Data Has Been Generated!")
    else:
        with res_container:
            st.error("Failed to Generate the Data")
         

if st.session_state['hasil_simulasi'] is not None:
    
    with res_container : 
        df_result = st.session_state['hasil_simulasi']
        file_name_info = st.session_state['info_simulasi']
        used_p = st.session_state['used_params']
        t_data = used_p['tariff_data']

        st.divider()
        st.markdown("### 📋 Generated Simulation Info")
        
        with st.container(border=True):
            pr_pct = f"{int(used_p['solar_pr'] * 100)}%"
            temp_val = f"{used_p['solar_temp']} / °C"
            
            if st.session_state.get('role', 'student') == 'admin':
                st.markdown(f"**📍 Location:** `{used_p['location']}` | **🗓️ Period:** `{used_p['period']}` | **🏠 Load:** `{used_p['load_source']}` **(x {used_p.get('load_multiplier', 1.0)})**")
                
                st.divider()
                
                c_sys1, c_sys2, c_sys3 = st.columns(3)
                
                with c_sys1:
                    st.markdown("#### ☀️ Solar PV")
                    st.markdown(f"""
                    - Capacity: **{used_p['solar']} kWp**
                    - PR: **{pr_pct}**
                    - Temp Coeff: **{temp_val}**
                    """)
                    
                with c_sys2:
                    st.markdown("#### 🔋 Battery Storage")
                    st.markdown(f"""
                    - Capacity: **{used_p['bat']} kWh**
                    - Power: **-{used_p['bat_charge_kw']} / +{used_p['bat_discharge_kw']} kW**
                    - Efficiency: **{int(used_p['bat_eff']*100)}%**
                    """)
                    
                with c_sys3:
                    st.markdown("#### ⚡ Control Logic")
                    st.markdown(f"""
                    - VPP Threshold: **{used_p['vpp_thresh']} AUD**
                    - SoC Limits: **{int(used_p['soc_min']*100)}% - {int(used_p['soc_max']*100)}%**
                    - Initial SoC: **{int(used_p['bat_soc_init']*100)}%**
                    """)
            else:
                st.markdown(f"**📍 Location:** `{used_p['location']}` | **🗓️ Period:** `{used_p['period']}` | **🏠 Load:** `{used_p['load_source']}` **(x {used_p.get('load_multiplier', 1.0)})** | **☀️ Solar PV:** PR `{pr_pct}` | Temp Coeff `{temp_val}`")

        with st.expander("💲 View Applied Tariff Details", expanded=False):
            schema_name = t_data.get('tariff_scheme', "Flat")
            st.markdown(f"**Scheme:** `{schema_name}`")
            
            if schema_name == "Wholesale Price":
                st.markdown("- **Import:** Spot Price + Market + Network + Other Fees\n- **Export:** Spot Price + Market Fees")
            else:
                tc1, tc2 = st.columns(2)
                with tc1:
                    st.markdown(f"**Export Tariff:**")
                    if schema_name == "Time of Use":
                        st.markdown(f"- Peak: **{t_data.get('exp_peak', 0.15)} AUD**\n- Shoulder: **{t_data.get('exp_shoulder', 0.10)} AUD**\n- Off-Peak: **{t_data.get('exp_offpeak', 0.05)} AUD**")
                    else:
                        st.markdown(f"Flat Rate: **{t_data.get('export_price', 0.08)} AUD/kWh**")
                with tc2:
                    st.markdown(f"**Import Tariff:**")
                    if schema_name == "Time of Use":
                        st.markdown(f"- Peak: **{t_data.get('peak_price', 0.45)} AUD**\n- Shoulder: **{t_data.get('shoulder_price', 0.25)} AUD**\n- Off-Peak: **{t_data.get('offpeak_price', 0.15)} AUD**")
                    else:
                        st.markdown(f"Flat Rate: **{t_data.get('import_flat', 0.20)} AUD/kWh**")

        st.markdown("### 💾 Export Data")
        
        df_export = df_result.copy()

        output_columns = [
            'timestamp',
            'irradiance',
            'temperature', 
            'solar_output_kw', 
            'load_profile',
            'price_profile',       
            'battery_soc_pct',
            'battery_soc_kwh',     
            'battery_power_ac_kw',
            'grid_net_kw',
            'tariff_import_AUD',
            'tariff_export_AUD'
        ]
        final_cols = [c for c in output_columns if c in df_export.columns]
        df_export = df_export[final_cols]

        df_export = df_export.rename(columns={
            'irradiance': 'irradiance_Wh/m^2',
            'temperature': 'temperature_C',
            'load_profile': 'load_kW',
            'price_profile': 'price_AUD/mWh',
            'battery_soc_pct': 'battery_soc_%',
            'tariff_import_AUD': 'tariff_import_AUD/kWh',
            'tariff_export_AUD': 'tariff_export_AUD/kWh'
        })
        
        csv = df_export.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Dataset (CSV)",
            data=csv,
            file_name=f"Data_{file_name_info}.csv",
            mime="text/csv",
            key='download-csv' 
        )

        if st.session_state.get('role', 'student') == 'admin':
            st.divider()
            st.subheader("📊 Detailed Analysis")
            
            df_result['year']  = df_result['timestamp'].dt.year
            df_result['month'] = df_result['timestamp'].dt.month
            
            available_years_vis = sorted(df_result['year'].unique())
            selected_vis_year = st.selectbox("Select Year:", available_years_vis)
            df_vis_year = df_result[df_result['year'] == selected_vis_year].copy()
            
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

            @st.fragment
            def show_monthly_analysis_fragment():
                available_months = sorted(df_vis_year['month'].unique())
                month_map = {m: calendar.month_name[m] for m in available_months}
                
                selected_month_name = st.selectbox("Select Month for Profile:", list(month_map.values()))
                
                selected_vis_month = [k for k, v in month_map.items() if v == selected_month_name][0]
                df_vis_month = df_vis_year[df_vis_year['month'] == selected_vis_month].copy()
                
                visualizer.plot_monthly_analysis(df_vis_month, col_load, selected_month_name, selected_vis_year)

            show_monthly_analysis_fragment()