import os
import pandas as pd
import numpy as np

ROOT_DIR = "dataset"

LOKASI_LIST = ["Aceh", "Medan", "Padang"]

TITIK_LIST = ["Lokasi_1", "Lokasi_2", "Lokasi_3"]

YEARS = [2023, 2024, 2025]

def generate_yearly_data(year):
    start = f"{year}-01-01 00:00:00"
    end = f"{year}-12-31 23:55:00"
    timestamps = pd.date_range(start=start, end=end, freq='5T')
    
    df = pd.DataFrame({'timestamp': timestamps})
    df['jam'] = df['timestamp'].dt.hour
    
    rand_factor = np.random.uniform(0.8, 1.2) 
    
    df['irradiance'] = df['jam'].apply(
        lambda x: max(0, np.sin((x - 6) * np.pi / 12) * 1000 * rand_factor) if 6 <= x <= 18 else 0
    ).round(2)
    
    df['suhu'] = (24 + (df['irradiance'] / 85) + np.random.uniform(-0.5, 0.5)).round(1)
    
    
    return df.drop(columns=['jam'])

def generate_price_profile():
    hours = list(range(24))
    prices = [2500 if 17 <= h <= 22 else 1440 for h in hours]
    return pd.DataFrame({'jam': hours, 'harga_per_kwh': prices})

print(f"Mulai membuat database terstruktur di folder '{ROOT_DIR}'...")

if not os.path.exists(ROOT_DIR):
    os.makedirs(ROOT_DIR)

for loc in LOKASI_LIST:
    loc_path = os.path.join(ROOT_DIR, loc)
    if not os.path.exists(loc_path):
        os.makedirs(loc_path)
    
    print(f"📂 Lokasi: {loc}")

    price_path = os.path.join(loc_path, "price_profile.csv")
    if not os.path.exists(price_path):
        generate_price_profile().to_csv(price_path, index=False)
        print(f"   ├── 💲 Dibuat: price_profile.csv")

    for titik in TITIK_LIST:
        titik_path = os.path.join(loc_path, titik)
        if not os.path.exists(titik_path):
            os.makedirs(titik_path)
        
        print(f"   ├── 📍 Titik: {titik}")
        
        for year in YEARS:
            file_path = os.path.join(titik_path, f"{year}.csv")
            if not os.path.exists(file_path):
                generate_yearly_data(year).to_csv(file_path, index=False)

print("\n✅ SELESAI! Struktur database Lokasi -> Titik -> Tahun siap.")