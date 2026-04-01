import streamlit as st
import pandas as pd
from datetime import time, datetime
from streamlit_gsheets import GSheetsConnection

TAB_CONFIG = "Config_History"

def init_default_states():
    """Mengisi nilai default ke dalam memori agar widget tidak bentrok (tanpa warning)"""
    defaults = {
        "chk_dur": False,
        "rand_dur_years": 1,
        "chk_loc": False,
        "chk_load": False,
        "load_mult": 15.0,
        "chk_solar": False,
        "chk_bat": False,
        "chk_tou": False,
        "vpp_threshold": 800,
        "sol_min": 4.0,
        "sol_max": 6.0,
        "sol_fix": 5.0,
        "sol_temp": -0.004,
        "sol_pr": 0.8,
        "bat_min": 8.0,
        "bat_max": 12.0,
        "bat_fix": 10.0,
        "bat_eff": 95,
        "bat_soc_init": 50,
        "bat_soc_range": (10, 90),
        "exp_tariff": 0.08, "imp_tariff": 0.20, 
        "pp": 0.45, "po": 0.15, "ps": 0.25
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def time_encoder(obj):
    """Mengubah object datetime.time menjadi string 'HH:MM' untuk database"""
    if isinstance(obj, time):
        return obj.strftime("%H:%M")
    raise TypeError("Type not serializable")

def get_gsheets_connection():
    return st.connection("gsheets", type=GSheetsConnection)

def load_config_history():
    try:
        conn = get_gsheets_connection()
        df_history = conn.read(worksheet=TAB_CONFIG, ttl=30) 
        if df_history is None or df_history.empty:
            return pd.DataFrame()
        if 'Config_Name' not in df_history.columns:
            return pd.DataFrame()
        df_history = df_history.dropna(subset=['Config_Name'])
        df_history = df_history[df_history['Config_Name'].astype(str).str.strip() != '']
        
        return df_history.tail(10).iloc[::-1]
    except Exception as e:
        st.error(f"⚠️ Failed to Read Config History: {e}")
        return pd.DataFrame()

def save_config_to_sheets(config_name, current_state):
    try:
        conn = get_gsheets_connection()
        df_existing = conn.read(worksheet=TAB_CONFIG, ttl=0)
        df_existing = df_existing.dropna(subset=['Config_Name'])
        
        new_row = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Config_Name": config_name,
            "use_rand_duration": current_state.get("chk_dur", False),
            "rand_dur_years": current_state.get("rand_dur_years", 1),
            "start_year": current_state.get("date_start", 2020),
            "end_year": current_state.get("date_end", 2020),
            "use_rand_location": current_state.get("chk_loc", True),
            "region_fix": current_state.get("loc_region", ""),
            "point_fix": current_state.get("loc_point", ""),
            "use_rand_load_profile": current_state.get("chk_load", False),
            "load_profile_fix": current_state.get("sel_load_file", ""),
            "load_mult": current_state.get("load_mult", 15.0),
            "use_rand_solar": current_state.get("chk_solar", False),
            "solar_min": current_state.get("sol_min", 4.0),
            "solar_max": current_state.get("sol_max", 6.0),
            "solar_fix": current_state.get("sol_fix", 5.0),
            "temp_coeff": current_state.get("sol_temp", -0.004),
            "pr": current_state.get("sol_pr", 0.8),
            "use_rand_bat": current_state.get("chk_bat", False),
            "bat_min": current_state.get("bat_min", 8.0),
            "bat_max": current_state.get("bat_max", 12.0),
            "bat_fix": current_state.get("bat_fix", 10.0),
            "bat_eff": current_state.get("bat_eff", 95),
            "bat_init_soc": current_state.get("bat_soc_init", 50) / 100,
            "soc_min": current_state.get("bat_soc_range", (10, 90))[0] / 100,
            "soc_max": current_state.get("bat_soc_range", (10, 90))[1] / 100,
            "vpp_thresh": current_state.get("vpp_threshold", 800),
            "t_peak_start": time_encoder(current_state.get("t_p_start", time(17,0))),
            "t_peak_end": time_encoder(current_state.get("t_p_end", time(20,0))),
            "t_offpeak_start": time_encoder(current_state.get("t_o_start", time(22,0))),
            "t_offpeak_end": time_encoder(current_state.get("t_o_end", time(6,0))),
            "t_shoulder_start": time_encoder(current_state.get("t_s_start", time(14,0))),
            "t_shoulder_end": time_encoder(current_state.get("t_s_end", time(17,0))),
            "exp_tariff": current_state.get("exp_tariff", 0.08),
            "use_tou": current_state.get("chk_tou", False),
            "imp_tariff": current_state.get("imp_tariff", 0.20),
            "p_peak": current_state.get("pp", 0.45),
            "p_offpeak": current_state.get("po", 0.15),
            "p_shoulder": current_state.get("ps", 0.25)
        }
        
        df_new = pd.DataFrame([new_row])
        df_updated = pd.concat([df_existing, df_new], ignore_index=True)
        conn.update(worksheet=TAB_CONFIG, data=df_updated)
        
        st.cache_data.clear()
        return True
    except Exception as e:
        st.error(f"⚠️ Failed to Save Config: {e}")
        return False
    
def apply_row_to_session(selected_row):
    mapping = {
        "use_rand_duration": "chk_dur",
        "rand_dur_years": "rand_dur_years",
        "use_rand_location": "chk_loc", "region_fix": "loc_region", "point_fix": "loc_point",
        "use_rand_load_profile": "chk_load", "load_profile_fix": "sel_load_file",
        "load_mult": "load_mult",
        "use_rand_solar": "chk_solar", "solar_min": "sol_min", "solar_max": "sol_max",
        "solar_fix": "sol_fix", "temp_coeff": "sol_temp", "pr": "sol_pr",
        "use_rand_bat": "chk_bat", "bat_min": "bat_min", "bat_max": "bat_max",
        "bat_fix": "bat_fix", "bat_eff": "bat_eff", "bat_init_soc": "bat_soc_init",
        "vpp_thresh": "vpp_threshold", "t_peak_start": "t_p_start", "t_peak_end": "t_p_end",
        "t_offpeak_start": "t_o_start", "t_offpeak_end": "t_o_end",
        "t_shoulder_start": "t_s_start", "t_shoulder_end": "t_s_end",
        "exp_tariff": "exp_tariff", "use_tou": "chk_tou", "imp_tariff": "imp_tariff",
        "p_peak": "pp", "p_offpeak": "po", "p_shoulder": "ps"
    }
    for db_col, widget_key in mapping.items():
        if db_col in selected_row:
            val = selected_row[db_col]
            if db_col.startswith("t_") and isinstance(val, str):
                try:
                    h, m = map(int, val.split(':'))
                    st.session_state[widget_key] = time(h, m)
                except: pass
            elif widget_key.startswith("chk_"):
                if pd.isna(val): 
                    st.session_state[widget_key] = False
                else:
                    teks_val = str(val).strip().upper()
                    if teks_val in ["TRUE", "1", "1.0"]:
                        st.session_state[widget_key] = True
                    else:
                        st.session_state[widget_key] = False
            else:
                if not pd.isna(val):
                    if db_col == "bat_init_soc":
                        st.session_state[widget_key] = int(float(val) * 100)
                    elif widget_key in ["vpp_threshold", "bat_eff", "date_start", "date_end", "rand_dur_years"]:
                        st.session_state[widget_key] = int(float(val))
                    elif widget_key in ["load_mult","sol_min", "sol_max", "sol_fix", "sol_temp", "sol_pr", "bat_min", "bat_max", "bat_fix", "exp_tariff", "imp_tariff", "pp", "po", "ps"]:
                        st.session_state[widget_key] = float(val)
                    else:
                        st.session_state[widget_key] = val
                
    if "soc_min" in selected_row and "soc_max" in selected_row:
        val_min = selected_row["soc_min"]
        val_max = selected_row["soc_max"]
        if not pd.isna(val_min) and not pd.isna(val_max):
            st.session_state["bat_soc_range"] = (int(float(val_min)*100), int(float(val_max)*100))
            
    if "start_year" in selected_row and not pd.isna(selected_row["start_year"]): 
        st.session_state["date_start"] = int(float(selected_row["start_year"]))
    if "end_year" in selected_row and not pd.isna(selected_row["end_year"]): 
        st.session_state["date_end"] = int(float(selected_row["end_year"]))