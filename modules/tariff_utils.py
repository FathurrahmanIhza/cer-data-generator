import streamlit as st
from datetime import time

# ==========================================
# 1. SETTING ANGKA AWAL (DEFAULT VALUES)
# ==========================================
# Di sini kita tentukan jam start awalnya.
# Sesuai request Anda:
# - Peak: 19:00 s/d 23:00
# - Off-Peak: 23:00 s/d 07:00
# - Shoulder: 07:00 s/d 19:00

DEFAULT_PEAK_START = time(19, 0)  # Jam 19:00
DEFAULT_PEAK_END   = time(23, 0)  # Jam 23:00
DEFAULT_OFFPEAK_END= time(7, 0)   # Jam 07:00

# ==========================================
# 2. INISIALISASI STATE (MEMORY)
# ==========================================
def initialize_session_state():
    """
    Fungsi ini dipanggil SEKALI saja saat aplikasi loading.
    Tugasnya memasukkan 'Angka Awal' di atas ke dalam memori Streamlit.
    """
    
    # Cek apakah memori 't_p_start' sudah ada? Jika belum, isi dengan DEFAULT
    if "t_p_start" not in st.session_state:
        st.session_state.t_p_start = DEFAULT_PEAK_START
    
    if "t_p_end" not in st.session_state:
        st.session_state.t_p_end = DEFAULT_PEAK_END
        
    # LOGIKA LINKING (Agar otomatis nyambung)
    # Start Off-Peak pasti sama dengan End Peak
    if "t_o_start" not in st.session_state:
        st.session_state.t_o_start = DEFAULT_PEAK_END 
        
    if "t_o_end" not in st.session_state:
        st.session_state.t_o_end = DEFAULT_OFFPEAK_END
        
    # Start Shoulder pasti sama dengan End Off-Peak
    if "t_s_start" not in st.session_state:
        st.session_state.t_s_start = DEFAULT_OFFPEAK_END
        
    # End Shoulder pasti sama dengan Start Peak
    if "t_s_end" not in st.session_state:
        st.session_state.t_s_end = DEFAULT_PEAK_START

# ==========================================
# 3. FUNGSI SINKRONISASI (CALLBACKS)
# ==========================================
# Bagian ini menjaga agar kalau satu diganti, temannya ikut terganti.

def sync_peak_end():
    st.session_state.t_o_start = st.session_state.t_p_end

def sync_offpeak_start():
    st.session_state.t_p_end = st.session_state.t_o_start

def sync_offpeak_end():
    st.session_state.t_s_start = st.session_state.t_o_end

def sync_shoulder_start():
    st.session_state.t_o_end = st.session_state.t_s_start

def sync_shoulder_end():
    st.session_state.t_p_start = st.session_state.t_s_end

def sync_peak_start():
    st.session_state.t_s_end = st.session_state.t_p_start

# ==========================================
# 4. FUNGSI LOGIKA ARRAY HARGA
# ==========================================
def generate_hourly_prices(p_peak, p_offpeak, p_shoulder):
    # Ambil nilai JAM (hour) saja dari memori
    h_peak_start = st.session_state.t_p_start.hour
    h_peak_end   = st.session_state.t_p_end.hour
    h_sh_start   = st.session_state.t_s_start.hour
    h_sh_end     = st.session_state.t_s_end.hour
    
    # Default array (isi semua dengan Offpeak dulu)
    prices = [p_offpeak] * 24
    
    # Fungsi pembantu untuk mengisi range jam
    def fill(start, end, val, arr):
        if start < end:
            for h in range(start, end): arr[h] = val
        else: # Kasus lintas hari (misal 23 ke 07)
            for h in range(start, 24): arr[h] = val
            for h in range(0, end):    arr[h] = val
        return arr

    # Timpa sesuai prioritas:
    # 1. Shoulder
    prices = fill(h_sh_start, h_sh_end, p_shoulder, prices)
    # 2. Peak (Paling kuat, menimpa yang lain)
    prices = fill(h_peak_start, h_peak_end, p_peak, prices)
    
    return prices