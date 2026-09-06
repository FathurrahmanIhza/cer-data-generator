"""
modules/data_noise.py
Modul untuk menginjeksi missing values (data gaps) pada dataset Assignment 2 Partial CSV.
Aturan:
- Missing values disisipkan beruntun (contiguous gaps): Small (1-2 jam / 12-24 baris), Medium (4-6 jam / 48-72 baris), High (24 jam / 288 baris).
- Total missing values disetting ~0,5% dari total baris data.
- Generator di-seed berdasarkan Student ID (NIM) agar hasil 100% deterministik & konsisten per mahasiswa.
- Kolom yang terdampak missing value:
  - Tahun 1: irradiance_W/m^2, temperature_C, solar_output_kW, load_kW.
  - Tahun 2+: irradiance_W/m^2, temperature_C, load_kW.
- Kolom commercial_forecasted_solar_output_kW:
  - Tahun 1: Selalu NaN.
  - Tahun 2+: Selalu berisi data forecast komersial (TIDAK BOLEH kena missing values).
"""

import hashlib
import numpy as np
import pandas as pd


def add_realistic_pv_noise(pv_pure: np.ndarray, irradiance: np.ndarray, noise_level: float = 0.08, cloud_prob: float = 0.05, seed: int = None) -> np.ndarray:
    """
    Menambahkan Multiplicative Gaussian noise dan cloud transient drop-out pada PV Output.
    
    Parameters
    ----------
    pv_pure : np.ndarray
        Array solar output murni dari rumus deterministik.
    irradiance : np.ndarray
        Array irradiance (W/m^2).
    noise_level : float
        Level Gaussian noise (default 0.08 = 8%).
    cloud_prob : float
        Probabilitas drop-out awan saat siang hari (default 0.05 = 5%).
    seed : int, optional
        Random seed untuk reproduksibilitas.
        
    Returns
    -------
    np.ndarray
        Array solar output yang sudah ditambah noise realistis.
    """
    if seed is not None:
        rng = np.random.RandomState(int(seed) % (2**32))
    else:
        rng = np.random.RandomState(42)

    # 1. Multiplicative Gaussian Noise (cth: variasi 8%)
    gaussian_noise = rng.normal(1.0, noise_level, size=len(pv_pure))
    pv_noisy = pv_pure * gaussian_noise

    # 2. Cloud Intermittency (Drop-out acak saat ada iradiasi matahari > 50 W/m^2)
    is_daytime = irradiance > 50
    cloud_drops = rng.binomial(1, cloud_prob, size=len(pv_pure))
    drop_severity = rng.uniform(0.4, 0.7, size=len(pv_pure))  # Drop 40% - 70%

    # Aplikasikan drop awan hanya di siang hari
    cloud_impact = 1.0 - (cloud_drops * drop_severity * is_daytime)
    pv_final = pv_noisy * cloud_impact

    # Pastikan tidak ada nilai di bawah 0
    return np.maximum(pv_final, 0.0)


def apply_assignment2_missing_values(df: pd.DataFrame, student_nim: str) -> pd.DataFrame:
    """
    Menginjeksi missing values pada DataFrame Partial CSV Assignment 2.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame hasil simulasi dengan kolom-kolom standar.
    student_nim : str
        ID / NIM Mahasiswa yang digunakan sebagai seed generator.
        
    Returns
    -------
    pd.DataFrame
        DataFrame turunan yang sudah diinjeksi missing values & kolom commercial_forecasted_solar_output_kW.
    """
    df_out = df.copy()

    # 1. Pastikan kolom timestamp berupa datetime
    if not pd.api.types.is_datetime64_any_dtype(df_out['timestamp']):
        df_out['timestamp'] = pd.to_datetime(df_out['timestamp'])

    # 2. Inisialisasi Seed dari NIM
    clean_nim = str(student_nim).strip().lower() if student_nim else "student"
    seed_hash = hashlib.md5(clean_nim.encode('utf-8')).hexdigest()
    seed_val = int(seed_hash[:8], 16) % (2**32)
    rng = np.random.RandomState(seed_val)

    total_rows = len(df_out)
    target_missing_budget = int(round(total_rows * 0.005))

    # 3. Tentukan Rentang Gap (Jumlah Baris per 5-Menit Interval)
    # Small Gap: 12 s.d. 24 baris (1-2 jam)
    # Medium Gap: 48 s.d. 72 baris (4-6 jam)
    # High Gap: 288 baris (24 jam)
    
    gap_lengths = []
    
    # Syarat Minimum Guarantees:
    # 1x High Gap (288 baris)
    gap_lengths.append(288)
    
    # 2x Medium Gaps (48-72 baris)
    for _ in range(2):
        gap_lengths.append(int(rng.randint(48, 73)))
        
    # 3x Small Gaps (12-24 baris)
    for _ in range(3):
        gap_lengths.append(int(rng.randint(12, 25)))

    current_missing = sum(gap_lengths)

    # Isi sisa budget 0.5% (jika budget masih tersisa) dengan kombinasi Small / Medium Gaps
    while current_missing < target_missing_budget:
        remaining = target_missing_budget - current_missing
        if remaining < 12:
            # Sisa budget terlalu kecil untuk gap baru, hentikan
            break
            
        # Pilih tipe gap secara acak (70% Small, 30% Medium)
        if remaining >= 48 and rng.rand() > 0.7:
            g_len = int(rng.randint(48, min(73, remaining + 1)))
        else:
            max_small = min(25, remaining + 1)
            if max_small <= 12:
                g_len = remaining
            else:
                g_len = int(rng.randint(12, max_small))
                
        gap_lengths.append(g_len)
        current_missing += g_len

    # 4. Cari Posisi Acak Tanpa Overlap
    # Gunakan buffer minimal 288 baris (1 hari) antar gap agar tidak menumpuk
    buffer = 288
    occupied_mask = np.zeros(total_rows, dtype=bool)
    placed_gaps = []

    # Urutkan gap dari terbesar ke terkecil agar penempatan gap besar lebih mudah
    gap_lengths.sort(reverse=True)

    for g_len in gap_lengths:
        placed = False
        attempts = 0
        max_attempts = 1000
        
        while not placed and attempts < max_attempts:
            attempts += 1
            # Pilih calon indeks start acak
            max_start = total_rows - g_len
            if max_start <= 0:
                break
                
            start_idx = rng.randint(0, max_start)
            end_idx = start_idx + g_len
            
            # Cek daerah buffer
            check_start = max(0, start_idx - buffer)
            check_end = min(total_rows, end_idx + buffer)
            
            if not np.any(occupied_mask[check_start:check_end]):
                # Tempat valid ditemukan!
                occupied_mask[start_idx:end_idx] = True
                placed_gaps.append((start_idx, end_idx))
                placed = True
                
        if not placed:
            # Jika buffer ketat gagal, coba lagi dengan buffer lebih kecil (12 baris)
            attempts = 0
            while not placed and attempts < max_attempts:
                attempts += 1
                max_start = total_rows - g_len
                if max_start <= 0:
                    break
                start_idx = rng.randint(0, max_start)
                end_idx = start_idx + g_len
                if not np.any(occupied_mask[start_idx:end_idx]):
                    occupied_mask[start_idx:end_idx] = True
                    placed_gaps.append((start_idx, end_idx))
                    placed = True

    # 5. Susun Kolom commercial_forecasted_solar_output_kW & Terapkan Missing Values
    first_year = df_out['timestamp'].dt.year.min()
    mask_year_1 = (df_out['timestamp'].dt.year == first_year)
    mask_year_2_plus = (df_out['timestamp'].dt.year > first_year)

    # Simpan data solar_output murni (tanpa noise) untuk commercial_forecasted_solar_output_kW sebelum di-NaN
    if 'solar_output_pure_kW' in df_out.columns:
        raw_solar_output = df_out['solar_output_pure_kW'].copy()
    elif 'solar_output_pure_kw' in df_out.columns:
        raw_solar_output = df_out['solar_output_pure_kw'].copy()
    elif 'solar_output_kW' in df_out.columns:
        raw_solar_output = df_out['solar_output_kW'].copy()
    else:
        raw_solar_output = None

    # Terapkan NaN pada baris gap yang berhasil ditaruh
    for start_idx, end_idx in placed_gaps:
        # Untuk baris Tahun 1: kosongkan irradiance, temperature, solar_output, load
        y1_slice = df_out.index[start_idx:end_idx][mask_year_1.iloc[start_idx:end_idx]]
        if len(y1_slice) > 0:
            for col in ['irradiance_W/m^2', 'temperature_C', 'solar_output_kW', 'load_kW']:
                if col in df_out.columns:
                    df_out.loc[y1_slice, col] = np.nan

        # Untuk baris Tahun 2+: kosongkan irradiance, temperature, load (solar_output sudah di-NaN nanti)
        y2_slice = df_out.index[start_idx:end_idx][mask_year_2_plus.iloc[start_idx:end_idx]]
        if len(y2_slice) > 0:
            for col in ['irradiance_W/m^2', 'temperature_C', 'load_kW']:
                if col in df_out.columns:
                    df_out.loc[y2_slice, col] = np.nan

    # Kosongkan solar_output_kW & spot_price di Tahun 2+ untuk Partial CSV (tugas mahasiswa)
    blank_cols_y2 = [
        'solar_output_kW',
        'spot_price_AUD/kWh',
    ]
    for col in blank_cols_y2:
        if col in df_out.columns:
            df_out.loc[mask_year_2_plus, col] = np.nan

    # Kolom commercial_forecasted_solar_output_kW:
    # Tahun 1: NaN
    # Tahun 2+: Nilai data solar output komersial asli (TIDAK kena missing values)
    if raw_solar_output is not None:
        comm_solar = pd.Series(np.nan, index=df_out.index)
        comm_solar.loc[mask_year_2_plus] = raw_solar_output.loc[mask_year_2_plus]
        df_out['commercial_forecasted_solar_output_kW'] = comm_solar
    else:
        df_out['commercial_forecasted_solar_output_kW'] = np.nan

    # Pastikan urutan kolom sesuai skema ASSIGNMENT_2 (commercial_forecasted_solar_output_kW tepat setelah solar_output_kW)
    from modules import assignment as asgn
    desired_order = asgn.get_output_columns(asgn.ASSIGNMENT_2, is_admin_full=False)
    final_cols = [c for c in desired_order if c in df_out.columns] + [c for c in df_out.columns if c not in desired_order]
    df_out = df_out[final_cols]

    return df_out
