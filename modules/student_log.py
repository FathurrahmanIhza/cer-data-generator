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

def save_log_to_sheets(nim, config_name, used_params_dict, assignment_type="assignment_1"):
    """
    Menyimpan log generate mahasiswa ke Supabase.
    assignment_type ikut disimpan untuk keperluan filter tracker & reproducibility regenerate.
    """
    try:
        new_row = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "NIM": str(nim).strip().upper(),
            "Config_Name": config_name,
            "Parameter_Snapshot": used_params_dict,
            "assignment_type": assignment_type,
        }
        
        supabase.table(TAB_LOGS).insert(new_row).execute()
        
        st.cache_data.clear()
        return True
    except Exception as e:
        st.error(f"⚠️ Gagal menyimpan Log Mahasiswa ke Supabase: {e}")
        return False

@st.cache_data(ttl=60)
def get_student_logs(assignment_type=None):
    """
    Mengambil riwayat log mahasiswa dari Supabase.
    Jika assignment_type diberikan, filter hanya log untuk assignment tersebut.
    """
    try:
        query = supabase.table(TAB_LOGS).select("*")
        if assignment_type:
            query = query.eq("assignment_type", assignment_type)
        
        response = query.execute()
        df = pd.DataFrame(response.data)
        
        if not df.empty:
            df = df.dropna(subset=['NIM', 'Timestamp'])
            df['Parameter_Snapshot'] = df['Parameter_Snapshot'].apply(
                lambda x: json.dumps(x) if isinstance(x, dict) else x
            )
            if 'assignment_type' not in df.columns:
                df['assignment_type'] = 'assignment_1'
        return df
    except Exception as e:
        st.error(f"⚠️ Gagal mengambil data log dari Supabase: {e}")
        return pd.DataFrame()