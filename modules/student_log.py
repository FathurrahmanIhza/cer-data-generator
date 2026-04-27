import streamlit as st
import pandas as pd
import json
import zlib
from datetime import datetime
from modules.config import supabase

TAB_LOGS = "student_logs"

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
    """
    Sekarang menyimpan ke Supabase. 
    Nama fungsi tetap sama agar tidak perlu banyak mengubah main.py, 
    tapi logic di dalamnya sudah beralih ke Supabase.
    """
    try:
        new_row = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "NIM": str(nim).strip().upper(),
            "Config_Name": config_name,
            "Parameter_Snapshot": used_params_dict 
        }
        
        supabase.table(TAB_LOGS).insert(new_row).execute()
        
        st.cache_data.clear()
        return True
    except Exception as e:
        st.error(f"⚠️ Gagal menyimpan Log Mahasiswa ke Supabase: {e}")
        return False

@st.cache_data(ttl=60)
def get_student_logs():
    """Mengambil riwayat log mahasiswa dari Supabase"""
    try:
        response = supabase.table(TAB_LOGS).select("*").execute()
        df = pd.DataFrame(response.data)
        
        if not df.empty:
            df = df.dropna(subset=['NIM', 'Timestamp'])

            df['Parameter_Snapshot'] = df['Parameter_Snapshot'].apply(
                lambda x: json.dumps(x) if isinstance(x, dict) else x
            )
        return df
    except Exception as e:
        st.error(f"⚠️ Gagal mengambil data log dari Supabase: {e}")
        return pd.DataFrame()