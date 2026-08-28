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
from modules import assignment as asgn
from modules import ui_helpers as ui_h
from modules import data_noise as d_noise
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

# Inisialisasi active_assignment sebelum app_initialized agar selalu ada
if 'active_assignment' not in st.session_state:
    st.session_state['active_assignment'] = asgn.ASSIGNMENT_1

if 'app_initialized' not in st.session_state:
    st.session_state['app_initialized'] = True
    active_asgn = st.session_state['active_assignment']
    df_hist = cfg.load_config_history(active_asgn)
    if not df_hist.empty:
        latest_config = df_hist.iloc[0]
        cfg.apply_row_to_session(latest_config)
        st.session_state['active_config'] = latest_config['Config_Name']

if 'hasil_simulasi' not in st.session_state:
    st.session_state['hasil_simulasi'] = None
    st.session_state['gen_csv_data']   = None
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
        
        # ── ASSIGNMENT SELECTOR ──────────────────────────────────────
        st.subheader("📋 Assignment Version")
        all_labels = asgn.get_all_labels()
        current_asgn_key   = st.session_state.get('active_assignment', asgn.ASSIGNMENT_1)
        current_asgn_label = asgn.get_label(current_asgn_key)
        idx_asgn = all_labels.index(current_asgn_label) if current_asgn_label in all_labels else 0
        
        selected_asgn_label = st.selectbox(
            "Select Assignment:",
            all_labels,
            index=idx_asgn,
            key="ui_assignment_selector"
        )
        selected_asgn_key = asgn.get_key_from_label(selected_asgn_label)
        
        # Saat assignment berubah: update session state & auto-load config terbaru
        if selected_asgn_key != st.session_state.get('active_assignment'):
            st.session_state['active_assignment'] = selected_asgn_key
            st.session_state['hasil_simulasi'] = None  # reset hasil lama
            st.session_state['gen_csv_data']   = None  # reset CSV bytes lama
            latest_row = cfg.get_latest_config_for_assignment(selected_asgn_key)
            if latest_row is not None:
                cfg.apply_row_to_session(latest_row)
                st.session_state['active_config'] = latest_row.get('Config_Name', 'Default Config')
            else:
                st.session_state['active_config'] = 'Default Config'
            st.rerun()

        active_cfg = st.session_state.get('active_config', 'Default Config')
        st.success(f"**Active Config:** {active_cfg}")
        st.markdown("Save and Load configuration")

        st.divider()
        
        st.subheader("📂 Load History Config")
        df_history = cfg.load_config_history(selected_asgn_key)
        
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

        # Ambil visibilitas parameter & visualisasi berdasarkan assignment aktif
        _asgn_key  = st.session_state.get('active_assignment', asgn.ASSIGNMENT_1)
        _param_vis = asgn.get_params_visibility(_asgn_key)
        _show_battery = _param_vis.get("show_battery", True)
        _show_vpp     = _param_vis.get("show_vpp", True)

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

                        _is_asgn2_dur = (_asgn_key == asgn.ASSIGNMENT_2)

                        # Assignment 2: start year tidak boleh tahun terakhir
                        # (harus ada setidaknya 1 tahun lagi sebagai end year)
                        if _is_asgn2_dur and len(available_years) >= 2:
                            valid_start_years = available_years[:-1]
                        else:
                            valid_start_years = available_years

                        saved_start = st.session_state.get('date_start', valid_start_years[0])
                        idx_start = valid_start_years.index(saved_start) if saved_start in valid_start_years else 0

                        ui_start_y = y1.selectbox(
                            "Start Date",
                            valid_start_years,
                            index=idx_start,
                            key="ui_date_start"
                        )
                        st.session_state['date_start'] = ui_start_y

                        # End year: selalu > start year untuk Asgn 2, >= untuk Asgn 1
                        if _is_asgn2_dur:
                            valid_end_years = [y for y in available_years if y > ui_start_y]
                        else:
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
                        
                        # Assignment 2: minimal 2 tahun
                        _min_dur = 2 if _asgn_key == asgn.ASSIGNMENT_2 else 1
                        saved_rand_dur = st.session_state.get('rand_dur_years', _min_dur)
                        saved_rand_dur = max(_min_dur, min(saved_rand_dur, total_years))

                        # Reset widget cache jika nilainya di bawah minimum (misal saat switch Asgn 1 → Asgn 2)
                        if st.session_state.get('ui_rand_dur_years', _min_dur) < _min_dur:
                            st.session_state['ui_rand_dur_years'] = _min_dur
                        
                        ui_dur = st.number_input(
                            f"Duration (Years)", 
                            min_value=_min_dur, 
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
                if _show_vpp:
                    st.info("⚙️ VPP Settings")
                    vpp_price = st.number_input("Dispatch Price Threshold (AUD/MWh)", 0, 2000, step=10, key="vpp_threshold")

                st.info("💲 Tariff")

                # Deteksi mode tariff berdasarkan assignment aktif
                _tariff_mode = asgn.get_params_visibility(_asgn_key).get("tariff_mode", "single")

                t_utils.initialize_session_state()

                # ── Callbacks waktu ToU (dipakai oleh Asgn 1 dan Asgn 2) ──────
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

                def _sync_pp(): st.session_state['pp'] = st.session_state['ui_pp']
                def _sync_po(): st.session_state['po'] = st.session_state['ui_po']
                def _sync_ps(): st.session_state['ps'] = st.session_state['ui_ps']
                def _sync_ep(): st.session_state['e_peak'] = st.session_state['ui_e_peak']
                def _sync_eo(): st.session_state['e_offpeak'] = st.session_state['ui_e_offpeak']
                def _sync_es(): st.session_state['e_shoulder'] = st.session_state['ui_e_shoulder']
                def _sync_flat_imp(): st.session_state['imp_tariff'] = st.session_state['ui_imp_tariff']
                def _sync_flat_exp(): st.session_state['exp_tariff'] = st.session_state['ui_exp_tariff']

                if _tariff_mode == "combined":
                    # ── ASSIGNMENT 2: Flat + ToU, layout vertikal 1 kolom ──

                    # -- Flat Tariff (baris atas) --
                    st.markdown("**Flat Tariff**")
                    tf1, tf2 = st.columns(2)
                    tf1.number_input("Import (AUD/kWh)", 0.0, 2.0, step=0.01,
                                     key="ui_imp_tariff",
                                     value=float(st.session_state.get('imp_tariff', 0.20)),
                                     on_change=_sync_flat_imp)
                    tf2.number_input("Export (AUD/kWh)", 0.0, 1.0, step=0.01,
                                     key="ui_exp_tariff",
                                     value=float(st.session_state.get('exp_tariff', 0.08)),
                                     on_change=_sync_flat_exp)

                    # -- Time of Use Tariff (baris bawah) --
                    st.markdown("**Time of Use Tariff**")

                    st.markdown("*Time Periods*")

                    # Setiap baris jam punya st.columns(2) sendiri agar sejajar
                    st.caption("Peak")
                    tr1, tr2 = st.columns(2)
                    tr1.time_input("Peak Start", key="ui_t_p_start",
                                   value=st.session_state.get('t_p_start', time(19,0)),
                                   on_change=_sync_t_p_start, label_visibility="collapsed")
                    tr2.time_input("Peak End", key="ui_t_p_end",
                                   value=st.session_state.get('t_p_end', time(23,0)),
                                   on_change=_sync_t_p_end, label_visibility="collapsed")

                    st.caption("Off-Peak")
                    tr1, tr2 = st.columns(2)
                    tr1.time_input("Off-Peak Start", key="ui_t_o_start",
                                   value=st.session_state.get('t_o_start', time(23,0)),
                                   on_change=_sync_t_o_start, label_visibility="collapsed")
                    tr2.time_input("Off-Peak End", key="ui_t_o_end",
                                   value=st.session_state.get('t_o_end', time(7,0)),
                                   on_change=_sync_t_o_end, label_visibility="collapsed")

                    st.caption("Shoulder")
                    tr1, tr2 = st.columns(2)
                    tr1.time_input("Shoulder Start", key="ui_t_s_start",
                                   value=st.session_state.get('t_s_start', time(7,0)),
                                   on_change=_sync_t_s_start, label_visibility="collapsed")
                    tr2.time_input("Shoulder End", key="ui_t_s_end",
                                   value=st.session_state.get('t_s_end', time(19,0)),
                                   on_change=_sync_t_s_end, label_visibility="collapsed")

                    st.markdown("*Prices (AUD/kWh)*")
                    tp1, tp2 = st.columns(2)
                    with tp1:
                        st.markdown("Import")
                        st.number_input("Peak", 0.0, 2.0, step=0.01, key="ui_pp",
                                        value=float(st.session_state.get('pp', 0.45)), on_change=_sync_pp)
                        st.number_input("Off-Peak", 0.0, 2.0, step=0.01, key="ui_po",
                                        value=float(st.session_state.get('po', 0.15)), on_change=_sync_po)
                        st.number_input("Shoulder", 0.0, 2.0, step=0.01, key="ui_ps",
                                        value=float(st.session_state.get('ps', 0.25)), on_change=_sync_ps)
                    with tp2:
                        st.markdown("Export")
                        st.number_input("Peak", 0.0, 2.0, step=0.01, key="ui_e_peak",
                                        value=float(st.session_state.get('e_peak', 0.15)), on_change=_sync_ep)
                        st.number_input("Off-Peak", 0.0, 2.0, step=0.01, key="ui_e_offpeak",
                                        value=float(st.session_state.get('e_offpeak', 0.05)), on_change=_sync_eo)
                        st.number_input("Shoulder", 0.0, 2.0, step=0.01, key="ui_e_shoulder",
                                        value=float(st.session_state.get('e_shoulder', 0.10)), on_change=_sync_es)

                else:
                    # ── ASSIGNMENT 1 (dan default): dropdown scheme seperti sebelumnya ──
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

                    if ui_scheme == "Flat":
                        st.markdown("**💲 Set Prices (AUD/kWh)**")
                        c1, c2 = st.columns(2)
                        c1.number_input("Import", 0.0, 2.0, step=0.01, key="ui_imp_tariff",
                                        value=float(st.session_state.get('imp_tariff', 0.20)), on_change=_sync_flat_imp)
                        c2.number_input("Export", 0.0, 1.0, step=0.01, key="ui_exp_tariff",
                                        value=float(st.session_state.get('exp_tariff', 0.08)), on_change=_sync_flat_exp)

                    elif ui_scheme == "Time of Use":
                        st.markdown("**🕒 Set Time Periods**")

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
                if _show_battery:
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
                else:
                    st.info("🔋 Battery — *Not used in this version*")

        st.markdown("---")
        btn_run = st.button("Generate Data", type="primary", width="stretch", key="btn_admin")
        res_container = st.container()

    with tab_tracker:
        
        # Membungkus UI Tracker ke dalam Fragment agar tidak refresh 1 halaman penuh
        @st.fragment
        def tracker_ui():
            # Filter per assignment
            tracker_asgn_labels = asgn.get_all_labels()
            default_asgn_label  = asgn.get_label(st.session_state.get('active_assignment', asgn.ASSIGNMENT_1))
            default_idx = tracker_asgn_labels.index(default_asgn_label) if default_asgn_label in tracker_asgn_labels else 0

            selected_tracker_label = st.selectbox(
                "Filter by Assignment:",
                tracker_asgn_labels,
                index=default_idx,
                key="tracker_asgn_filter"
            )
            selected_tracker_asgn = asgn.get_key_from_label(selected_tracker_label)

            df_logs = s_log.get_student_logs(assignment_type=selected_tracker_asgn)
            
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
                    base = f"Loc: {p.get('location','')} | PV: {p.get('solar','')}kWp | Load: {p.get('load_source','')}"
                    if p.get('bat') is not None:
                        base += f" | Bat: {p.get('bat','')}kWh"
                    return base
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
                            
                            # --- Ambil assignment_type dari snapshot ---
                            regen_asgn_type = saved_params.get('assignment_type', 'assignment_1')

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

                                sim_params = {
                                    'solar_capacity_kw': saved_params['solar'], 
                                    'temp_coeff': saved_params['solar_temp'],
                                    'pr': saved_params['solar_pr'],
                                    'df_wholesale_fees': loader.get_wholesale_fees(reg),
                                }

                                if regen_asgn_type == 'assignment_1':
                                    sim_params.update({
                                        'battery_capacity_kwh': saved_params.get('bat', 10.0), 
                                        'battery_efficiency': saved_params.get('bat_eff', 0.95),
                                        'battery_initial_soc': saved_params.get('bat_soc_init', 0.5),
                                        'max_charge_kw': saved_params.get('bat_charge_kw', 10.0),
                                        'max_discharge_kw': saved_params.get('bat_discharge_kw', 10.0),
                                        'soc_min_pct': saved_params.get('soc_min', 0.1),
                                        'soc_max_pct': saved_params.get('soc_max', 0.9),
                                        'dispatch_price_threshold': saved_params.get('vpp_thresh', 800),
                                    })
                                elif regen_asgn_type == 'assignment_2':
                                    r_solar = saved_params.get('solar', 5.0)
                                    r_load_mult = saved_params.get('load_multiplier', 15.0)
                                    r_bat_cap = saved_params.get('bat')
                                    if r_bat_cap is None:
                                        r_bat_cap = round(((1.5 * r_solar) + (0.3 * r_load_mult)) * 2) / 2
                                    r_bat_power = saved_params.get('bat_charge_kw')
                                    if r_bat_power is None:
                                        r_bat_power = min(r_solar, round((0.5 * r_bat_cap) * 2) / 2)
                                    sim_params.update({
                                        'battery_capacity_kwh': r_bat_cap,
                                        'battery_efficiency': 0.95,
                                        'battery_initial_soc': 0.5,
                                        'max_charge_kw': r_bat_power,
                                        'max_discharge_kw': r_bat_power,
                                        'soc_min_pct': 0.1,
                                        'soc_max_pct': 0.9,
                                        'dispatch_price_threshold': 800,
                                    })
                                
                                t_data = saved_params['tariff_data']
                                regen_scheme = t_data.get('tariff_scheme', 'Flat')
                                sim_params['tariff_scheme'] = regen_scheme

                                # Time defaults (bisa di-override oleh snapshot)
                                sim_params.update({
                                    't_peak_start':    time(17, 0),
                                    't_peak_end':      time(20, 0),
                                    't_offpeak_start': time(22, 0),
                                    't_offpeak_end':   time(6, 0),
                                    't_shoulder_start':time(14, 0),
                                    't_shoulder_end':  time(17, 0)
                                })

                                # Restore tariff params dari snapshot
                                if regen_scheme in ("Time of Use", "Assignment2"):
                                    sim_params.update({
                                        'peak_price':    t_data.get('peak_price', 0.45),
                                        'exp_peak':      t_data.get('exp_peak', 0.15),
                                        't_peak_start':  datetime.strptime(t_data['peak_start'],    "%H:%M").time(),
                                        't_peak_end':    datetime.strptime(t_data['peak_end'],      "%H:%M").time(),
                                        'offpeak_price': t_data.get('offpeak_price', 0.15),
                                        'exp_offpeak':   t_data.get('exp_offpeak', 0.05),
                                        't_offpeak_start': datetime.strptime(t_data['offpeak_start'], "%H:%M").time(),
                                        't_offpeak_end':   datetime.strptime(t_data['offpeak_end'],   "%H:%M").time(),
                                        'shoulder_price':t_data.get('shoulder_price', 0.25),
                                        'exp_shoulder':  t_data.get('exp_shoulder', 0.10),
                                        't_shoulder_start': datetime.strptime(t_data['shoulder_start'], "%H:%M").time(),
                                        't_shoulder_end':   datetime.strptime(t_data['shoulder_end'],   "%H:%M").time(),
                                    })
                                if regen_scheme in ("Flat", "Assignment2"):
                                    sim_params['import_flat']  = t_data.get('import_flat', 0.20)
                                    sim_params['export_price'] = t_data.get('export_price', 0.08)

                                df_result_regen = calculator.run_simulation(df_input_regen, sim_params, regen_asgn_type)

                                # Rename map kondisional per assignment
                                if regen_asgn_type == asgn.ASSIGNMENT_2:
                                    df_export = df_result_regen.rename(columns={
                                        'irradiance':          'irradiance_W/m^2',
                                        'temperature':         'temperature_C',
                                        'load_profile':        'load_kW',
                                        'price_profile':       'price_AUD/MWh',
                                        'solar_output_kw':     'solar_output_kW',
                                        'battery_soc_pct':     'battery_soc_%',
                                        'battery_soc_kwh':     'battery_soc_kwh',
                                        'battery_power_ac_kw': 'battery_power_ac_kW',
                                        'grid_net_kw':         'grid_net_kW',
                                    })
                                else:
                                    df_export = df_result_regen.round(2).rename(columns={
                                        'irradiance':          'irradiance_W/m^2',
                                        'temperature':         'temperature_C',
                                        'load_profile':        'load_kW',
                                        'price_profile':       'price_AUD/MWh',
                                        'solar_output_kw':     'solar_output_kW',
                                        'battery_soc_pct':     'battery_soc_%',
                                        'battery_soc_kwh':     'battery_soc_kwh',
                                        'battery_power_ac_kw': 'battery_power_ac_kW',
                                        'tariff_import_AUD':   'tariff_import_AUD/kWh',
                                        'tariff_export_AUD':   'tariff_export_AUD/kWh',
                                        'grid_net_kw':         'grid_net_kW',
                                    })

                                _out_cols_full = asgn.get_output_columns(regen_asgn_type, is_admin_full=True)
                                df_export_full = df_export[[c for c in _out_cols_full if c in df_export.columns]]

                                # Full CSV
                                st.session_state['regen_csv_data'] = df_export_full.to_csv(index=False).encode('utf-8')

                                # Partial CSV untuk Asgn 2
                                if regen_asgn_type == asgn.ASSIGNMENT_2:
                                    _out_cols_student = asgn.get_output_columns(regen_asgn_type, is_admin_full=False)
                                    _df_regen_partial = df_export[[c for c in _out_cols_student if c in df_export.columns]].copy()
                                    _df_regen_partial = d_noise.apply_assignment2_missing_values(_df_regen_partial, nim_target)
                                    _df_regen_partial = _df_regen_partial[[c for c in _out_cols_student if c in _df_regen_partial.columns]]
                                    st.session_state['regen_csv_data_partial'] = _df_regen_partial.to_csv(index=False).encode('utf-8')
                                    del _df_regen_partial
                                else:
                                    st.session_state['regen_csv_data_partial'] = None

                                st.session_state['regen_nim'] = nim_target
                                st.session_state['regen_reg'] = reg
                                st.session_state['regen_pt'] = pt
                                st.session_state['regen_params'] = saved_params
                                st.session_state['regen_df_result'] = df_result_regen
                                st.session_state['regen_assignment_type'] = regen_asgn_type

                    except Exception as e:
                        st.error(f"Failed to process data: {e}")
            else:
                st.info("Select one of the rows to re-generate the data.")
                
            if st.session_state.get('regen_csv_data') is not None:
                st.success("✅ Data has been re-generated!")

                _regen_asgn = st.session_state.get('regen_assignment_type', asgn.ASSIGNMENT_1)
                _regen_vc   = asgn.get_vis_config(_regen_asgn)
                _regen_nim  = st.session_state['regen_nim']
                _regen_reg  = st.session_state['regen_reg']
                _regen_pt   = st.session_state['regen_pt']

                ui_h.render_result_panel(
                    df_result            = st.session_state.get('regen_df_result'),
                    used_p               = st.session_state['regen_params'],
                    vc                   = _regen_vc,
                    csv_bytes            = st.session_state['regen_csv_data'],
                    csv_bytes_partial    = st.session_state.get('regen_csv_data_partial'),
                    download_label       = "Download Dataset (CSV)",
                    download_filename    = f"Data_{_regen_nim}_{_regen_reg}_{_regen_pt}.csv",
                    download_key         = f"dl_regen_{_regen_nim}",
                    download_key_partial = f"dl_regen_partial_{_regen_nim}",
                    year_key             = "sb_year_regen",
                    month_key            = "sb_month_regen",
                    show_analysis        = True,
                )

        tracker_ui()

else:
    active_cfg = st.session_state.get('active_config', 'Default')
    st.info(f"👋 **Welcome!**  \n\nSelect your assignment and enter your Student ID to generate your dataset.")

    # ── ASSIGNMENT SELECTOR (Student) ──────────────────────────────
    student_asgn_labels = asgn.get_all_labels()
    current_student_asgn_key   = st.session_state.get('active_assignment', asgn.ASSIGNMENT_1)
    current_student_asgn_label = asgn.get_label(current_student_asgn_key)
    idx_student_asgn = student_asgn_labels.index(current_student_asgn_label) if current_student_asgn_label in student_asgn_labels else 0

    selected_student_asgn_label = st.selectbox(
        "Select Assignment:",
        student_asgn_labels,
        index=idx_student_asgn,
        key="ui_student_assignment"
    )
    selected_student_asgn_key = asgn.get_key_from_label(selected_student_asgn_label)
    st.session_state['active_assignment'] = selected_student_asgn_key

    student_nim = st.text_input("Student ID", placeholder="e.g.: z5593968").strip()
    st.session_state['current_nim'] = student_nim

    st.markdown("---")
    btn_run = st.button("Generate Data", type="primary", width="stretch", key="btn_student")
    res_container = st.container()



if btn_run:
    # Ambil assignment aktif untuk generate ini
    active_asgn_type = st.session_state.get('active_assignment', asgn.ASSIGNMENT_1)

    if st.session_state['role'] == 'student':
        if not st.session_state.get('current_nim'):
            st.warning("⚠️ Please Enter Your Student ID!")
            st.stop()

        df_hist = cfg.load_config_history(active_asgn_type)
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

    # --- KALKULASI BATERAI (hanya untuk Assignment 1) ---
    final_p_bat = None
    auto_charge_power = None

    if active_asgn_type == asgn.ASSIGNMENT_1:
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

        if active_asgn_type == asgn.ASSIGNMENT_1 and final_p_bat is not None:
            params.update({
                'battery_capacity_kwh': final_p_bat, 
                'battery_efficiency': p_eff,
                'battery_initial_soc': p_soc,
                'max_charge_kw': auto_charge_power,
                'max_discharge_kw': auto_charge_power,
                'soc_min_pct': p_min_soc,
                'soc_max_pct': p_max_soc,
                'dispatch_price_threshold': vpp_price,
            })
        elif active_asgn_type == asgn.ASSIGNMENT_2:
            auto_bat_cap = round(((1.5 * final_p_solar) + (0.3 * final_load_mult)) * 2) / 2
            auto_bat_power = min(final_p_solar, round((0.5 * auto_bat_cap) * 2) / 2)
            final_p_bat = auto_bat_cap
            auto_charge_power = auto_bat_power
            params.update({
                'battery_capacity_kwh': auto_bat_cap,
                'battery_efficiency': 0.95,
                'battery_initial_soc': 0.5,
                'max_charge_kw': auto_bat_power,
                'max_discharge_kw': auto_bat_power,
                'soc_min_pct': 0.1,
                'soc_max_pct': 0.9,
                'dispatch_price_threshold': 800,
            })
        
        with st.spinner("Calculating Energy Flow..."):
            df_result = calculator.run_simulation(df_input, params, active_asgn_type)

        
        st.session_state['hasil_simulasi'] = df_result
        st.session_state['info_simulasi'] = f"{selected_loc}_{selected_point}_{final_start_y}-{final_end_y}"

        # ── Buat CSV bytes sekali saat generate ──────────────────────────────
        _asgn_for_csv = active_asgn_type

        if _asgn_for_csv == asgn.ASSIGNMENT_2:
            # Asgn 2: price_profile → price_AUD/MWh; spot_price_AUD/kWh & battery cols dari calculator
            _df_csv = df_result.copy().rename(columns={
                'irradiance':          'irradiance_W/m^2',
                'temperature':         'temperature_C',
                'load_profile':        'load_kW',
                'price_profile':       'price_AUD/MWh',
                'solar_output_kw':     'solar_output_kW',
                'battery_soc_pct':     'battery_soc_%',
                'battery_soc_kwh':     'battery_soc_kwh',
                'battery_power_ac_kw': 'battery_power_ac_kW',
                'grid_net_kw':         'grid_net_kW',
            })
        else:
            # Asgn 1: rename map lama (termasuk tariff_import/export_AUD)
            _df_csv = df_result.copy().rename(columns={
                'irradiance':          'irradiance_W/m^2',
                'temperature':         'temperature_C',
                'load_profile':        'load_kW',
                'price_profile':       'price_AUD/MWh',
                'solar_output_kw':     'solar_output_kW',
                'battery_soc_pct':     'battery_soc_%',
                'battery_soc_kwh':     'battery_soc_kwh',
                'battery_power_ac_kw': 'battery_power_ac_kW',
                'tariff_import_AUD':   'tariff_import_AUD/kWh',
                'tariff_export_AUD':   'tariff_export_AUD/kWh',
                'grid_net_kw':         'grid_net_kW',
            })
            for _c in ['tariff_import_AUD/kWh', 'tariff_export_AUD/kWh']:
                if _c in _df_csv.columns:
                    _df_csv[_c] = _df_csv[_c].round(5)

        _desired_full = asgn.get_output_columns(_asgn_for_csv, is_admin_full=True)
        _df_full_csv  = _df_csv[[c for c in _desired_full if c in _df_csv.columns]]

        # Full CSV (semua data lengkap termasuk kolom baterai untuk Admin)
        st.session_state['gen_csv_data'] = _df_full_csv.to_csv(index=False).encode('utf-8')

        # Partial CSV (hanya untuk Asgn 2): tahun ke-2+ dikosongkan di kolom forecast
        if _asgn_for_csv == asgn.ASSIGNMENT_2:
            _desired_student = asgn.get_output_columns(_asgn_for_csv, is_admin_full=False)
            _df_partial = _df_csv[[c for c in _desired_student if c in _df_csv.columns]].copy()
            _nim_for_noise = student_nim if st.session_state.get('role') == 'student' else 'student'
            _df_partial = d_noise.apply_assignment2_missing_values(_df_partial, _nim_for_noise)
            _df_partial = _df_partial[[c for c in _desired_student if c in _df_partial.columns]]
            st.session_state['gen_csv_data_partial'] = _df_partial.to_csv(index=False).encode('utf-8')
            del _df_partial
        else:
            st.session_state['gen_csv_data_partial'] = None

        del _df_csv

        # ── Susun Snapshot Tarif ──────────────────────────────────────────────
        if active_asgn_type == asgn.ASSIGNMENT_2:
            # Asgn 2: selalu simpan Flat + ToU sekaligus
            tariff_snapshot = {
                'tariff_scheme': 'Assignment2',
                'import_flat': p_flat,
                'export_price': exp_price,
                'peak_price': p_peak, 'exp_peak': e_peak,
                'peak_start': st.session_state.get('t_p_start', time(19,0)).strftime("%H:%M"),
                'peak_end': st.session_state.get('t_p_end', time(23,0)).strftime("%H:%M"),
                'offpeak_price': p_offpeak, 'exp_offpeak': e_offpeak,
                'offpeak_start': st.session_state.get('t_o_start', time(23,0)).strftime("%H:%M"),
                'offpeak_end': st.session_state.get('t_o_end', time(7,0)).strftime("%H:%M"),
                'shoulder_price': p_shoulder, 'exp_shoulder': e_shoulder,
                'shoulder_start': st.session_state.get('t_s_start', time(7,0)).strftime("%H:%M"),
                'shoulder_end': st.session_state.get('t_s_end', time(19,0)).strftime("%H:%M"),
            }
        else:
            # Asgn 1: snapshot per skema yang dipilih
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
            'assignment_type': active_asgn_type,
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
                st.session_state['used_params'],
                assignment_type=active_asgn_type
            )

        
        with res_container: 
            st.success(f"Data has been generated!")
    else:
        with res_container:
            st.error("Failed to generate the data")
         

if st.session_state['hasil_simulasi'] is not None:

    with res_container:
        df_result      = st.session_state['hasil_simulasi']
        file_name_info = st.session_state['info_simulasi']
        used_p         = st.session_state['used_params']
        csv_bytes      = st.session_state.get('gen_csv_data', b'')

        _gen_asgn = used_p.get('assignment_type', asgn.ASSIGNMENT_1)
        _gen_vc   = asgn.get_vis_config(_gen_asgn)

        ui_h.render_result_panel(
            df_result            = df_result,
            used_p               = used_p,
            vc                   = _gen_vc,
            csv_bytes            = csv_bytes,
            csv_bytes_partial    = st.session_state.get('gen_csv_data_partial'),
            download_label       = "Download Dataset (CSV)",
            download_filename    = f"Data_{file_name_info}.csv",
            download_key         = "download-csv",
            download_key_partial = "download-csv-partial",
            year_key             = "sb_year",
            month_key            = "sb_month",
            show_analysis        = True,
        )
