import streamlit as st
import pandas as pd
import json
import zlib
from datetime import datetime
from streamlit_gsheets import GSheetsConnection

TAB_LOGS = "Student_Logs"

def get_gsheets_connection():
    return st.connection("gsheets", type=GSheetsConnection)

def generate_seed(nim, config_name=""):
    """
    Menggabungkan NIM dan Nama Config agar seed-nya unik 
    untuk setiap tugas, meskipun NIM-nya sama.
    """
    nim_clean = str(nim).strip().upper()
    config_clean = str(config_name).strip()
    
    gabungan = f"{nim_clean}_{config_clean}"
    return zlib.crc32(gabungan.encode('utf-8'))

def save_log_to_sheets(nim, config_name, used_params_dict):
    """Menyimpan riwayat generate siswa ke database (Metadata JSON)"""
    try:
        conn = get_gsheets_connection()
        df_existing = conn.read(worksheet=TAB_LOGS, ttl=0)
        
        json_snapshot = json.dumps(used_params_dict)
        
        new_row = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "NIM": str(nim).strip().upper(),
            "Config_Name": config_name,
            "Parameter_Snapshot": json_snapshot
        }
        
        df_new = pd.DataFrame([new_row])
        df_updated = pd.concat([df_existing, df_new], ignore_index=True)
        conn.update(worksheet=TAB_LOGS, data=df_updated)
        
        st.cache_data.clear()
        return True
    except Exception as e:
        st.error(f"⚠️ Failed to Save Student Log: {e}")
        return False

def get_student_logs():
    """Mengambil riwayat log mahasiswa dari Google Sheets"""
    try:
        conn = get_gsheets_connection()
        df = conn.read(worksheet=TAB_LOGS, ttl=180)
        df = df.dropna(subset=['NIM', 'Timestamp'])
        return df
    except Exception as e:
        st.error(f"⚠️ Failed to fetch log data: {e}")
        return pd.DataFrame()