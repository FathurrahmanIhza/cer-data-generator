import os
import pandas as pd
import streamlit as st
import random 

DATASET_DIR = "dataset"
LOAD_PROFILE_DIR = os.path.join(DATASET_DIR, "load_profile")

def get_list_lokasi():
    if not os.path.exists(DATASET_DIR): return []
    items = os.listdir(DATASET_DIR)
    return sorted([item for item in items if os.path.isdir(os.path.join(DATASET_DIR, item)) and item != "load_profile"])

def get_list_titik(nama_lokasi):
    path_lokasi = os.path.join(DATASET_DIR, nama_lokasi)
    if not os.path.exists(path_lokasi): return []
    items = os.listdir(path_lokasi)
    return sorted([item for item in items if os.path.isdir(os.path.join(path_lokasi, item)) and item != "Price"])

def get_available_years(nama_lokasi, nama_titik):
    path_titik = os.path.join(DATASET_DIR, nama_lokasi, nama_titik)
    if not os.path.exists(path_titik): return []
    files = os.listdir(path_titik)
    return sorted([int(f.replace('.csv', '')) for f in files if f.endswith('.csv')])

def load_and_merge_data(nama_lokasi, nama_titik, start_year, end_year):
    list_df = []
    
    path_titik = os.path.join(DATASET_DIR, nama_lokasi, nama_titik) 
    path_price = os.path.join(DATASET_DIR, nama_lokasi, "Price")     
    
    if os.path.exists(LOAD_PROFILE_DIR):
        files = [f for f in os.listdir(LOAD_PROFILE_DIR) if f.endswith('.csv')]
        if files:
            selected_load_file = random.choice(files)
            path_selected_load = os.path.join(LOAD_PROFILE_DIR, selected_load_file)
            
            st.toast(f"🎲 Random Load Profile Selected: {selected_load_file}")
            
            try:
                df_load = pd.read_csv(path_selected_load)
                df_load['timestamp'] = pd.to_datetime(df_load['timestamp'])
                df_load = df_load.drop_duplicates(subset=['timestamp'])
                
                if 'beban_rumah_kw' in df_load.columns:
                    df_load.rename(columns={'beban_rumah_kw': 'load_profile'}, inplace=True)
                    
            except Exception as e:
                st.error(f"Gagal membaca file load profile: {e}")
                return None
        else:
            st.error("Folder load_profile kosong!")
            return None
    else:
        st.error("Folder load_profile tidak ditemukan.")
        return None

    for year in range(start_year, end_year + 1):
        file_weather = os.path.join(path_titik, f"{year}.csv")
        file_price = os.path.join(path_price, f"{year}.csv")
        
        if os.path.exists(file_weather) and os.path.exists(file_price):
            try:
                df_main = pd.read_csv(file_weather)
                df_main['timestamp'] = pd.to_datetime(df_main['timestamp'])
                df_main = df_main.drop_duplicates(subset=['timestamp'])
                
                if 'suhu' in df_main.columns:
                    df_main.rename(columns={'suhu': 'temperature'}, inplace=True)
                
                df_price = pd.read_csv(file_price)
                df_price['timestamp'] = pd.to_datetime(df_price['timestamp'])
                df_price = df_price.drop_duplicates(subset=['timestamp'])
                
                if 'harga_listrik' in df_price.columns:
                    df_price.rename(columns={'harga_listrik': 'price_import'}, inplace=True)
                
                df_merged = pd.merge(df_main, df_price, on='timestamp', how='left')
                
                df_merged = pd.merge(df_merged, df_load, on='timestamp', how='left')
                
                if 'load_profile' in df_merged.columns:
                    df_merged['load_profile'] = df_merged['load_profile'].fillna(0)
                
                if 'price_import' in df_merged.columns:
                    df_merged['price_import'] = df_merged['price_import'].ffill().fillna(0)
                
                list_df.append(df_merged)
                
            except Exception as e:
                st.error(f"Error tahun {year}: {e}")
        else:
            st.warning(f"Data tahun {year} tidak lengkap.")
            
    if not list_df:
        return None
        
    full_df = pd.concat(list_df, ignore_index=True)
    full_df = full_df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)

    return full_df