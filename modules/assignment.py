"""
modules/assignment.py
Registry sentral untuk semua definisi Assignment/Version simulasi.
Tambahkan assignment baru di sini tanpa perlu mengubah main.py atau calculator.py.
"""

ASSIGNMENT_1 = "assignment_1"
ASSIGNMENT_2 = "assignment_2"

# Daftar urutan tampil di UI (dropdown)
ALL_ASSIGNMENTS = [ASSIGNMENT_1, ASSIGNMENT_2]
ASSIGNMENT_LABELS = {
    ASSIGNMENT_1: "Assignment 1",
    ASSIGNMENT_2: "Assignment 2",
}

LABEL_TO_KEY = {v: k for k, v in ASSIGNMENT_LABELS.items()}

# =====================================================================
# VISIBILITAS PARAMETER UI (Config Manager admin)
# =====================================================================
ASSIGNMENT_PARAMS = {
    ASSIGNMENT_1: {
        "show_battery": True,
        "show_vpp":     True,
        "show_tariff":  True,
    },
    ASSIGNMENT_2: {
        "show_battery":  False,
        "show_vpp":      False,
        "show_tariff":   True,
        "tariff_mode":   "combined",  # Asgn 2: tampilkan Flat+ToU sekaligus, tanpa dropdown scheme
    },
}

# =====================================================================
# VISIBILITAS VISUALISASI (chart sections di admin result panel)
# Tambahkan key baru di sini untuk mengontrol chart per assignment,
# tanpa perlu menyentuh main.py atau visualizer.py secara manual.
# =====================================================================
VIS_CONFIG = {
    ASSIGNMENT_1: {
        "show_battery_charts":   True,   # ROW 1: battery discharge bar di stacked chart
        "show_vpp_charts":       True,   # ROW 2-4: heatmap & VPP cumulative sections
        "show_monthly_analysis": True,   # Monthly battery SoC chart (fragment)
        "show_row5":             True,   # Self-consumption, sufficiency, VPP dispatch
        "show_battery_logic":    True,   # Battery Logic Flow expander
        "show_grid_metric":      True,   # Grid Import metric di summary panel
    },
    ASSIGNMENT_2: {
        "show_battery_charts":   False,
        "show_vpp_charts":       False,
        "show_monthly_analysis": False,
        "show_row5":             False,
        "show_battery_logic":    False,
        "show_grid_metric":      False,
    },
}

# =====================================================================
# OUTPUT COLUMNS (kolom CSV yang di-download mahasiswa)
# =====================================================================
OUTPUT_COLUMNS = {
    ASSIGNMENT_1: [
        'timestamp',
        'irradiance_W/m^2',
        'temperature_C',
        'solar_output_kW',
        'load_kW',
        'battery_soc_%',
        'battery_soc_kwh',
        'battery_power_ac_kW',
        'grid_net_kW',
        'tariff_import_AUD/kWh',
        'tariff_export_AUD/kWh',
    ],
    ASSIGNMENT_2: [
        'timestamp',
        'irradiance_W/m^2',
        'temperature_C',
        'solar_output_kW',
        'load_kW',
        'spot_price_AUD/kWh',
        'tariff_import_flat_AUD/kWh',
        'tariff_export_flat_AUD/kWh',
        'tariff_import_tou_AUD/kWh',
        'tariff_export_tou_AUD/kWh',
    ],
}


# =====================================================================
# HELPERS
# =====================================================================

def get_label(assignment_type: str) -> str:
    """Kembalikan label UI untuk assignment_type tertentu."""
    return ASSIGNMENT_LABELS.get(assignment_type, assignment_type)

def get_all_labels() -> list:
    """Kembalikan list semua label UI (untuk dropdown)."""
    return [ASSIGNMENT_LABELS[k] for k in ALL_ASSIGNMENTS]

def get_key_from_label(label: str) -> str:
    """Kembalikan assignment_type dari label UI yang dipilih user."""
    return LABEL_TO_KEY.get(label, ASSIGNMENT_1)

def get_params_visibility(assignment_type: str) -> dict:
    """Kembalikan dict visibilitas parameter UI untuk assignment tertentu."""
    return ASSIGNMENT_PARAMS.get(assignment_type, ASSIGNMENT_PARAMS[ASSIGNMENT_1])

def get_vis_config(assignment_type: str) -> dict:
    """
    Kembalikan dict konfigurasi visualisasi untuk assignment tertentu.
    Digunakan oleh main.py dan visualizer.py untuk render chart secara kondisional.
    Fallback ke Assignment 1 (tampilkan semua) jika assignment tidak dikenal.

    Contoh penggunaan:
        vc = asgn.get_vis_config("assignment_2")
        if vc["show_battery_charts"]: ...
    """
    return VIS_CONFIG.get(assignment_type, VIS_CONFIG[ASSIGNMENT_1])

def get_output_columns(assignment_type: str) -> list:
    """Kembalikan list kolom output CSV untuk assignment tertentu."""
    return OUTPUT_COLUMNS.get(assignment_type, OUTPUT_COLUMNS[ASSIGNMENT_1])

def show_battery(assignment_type: str) -> bool:
    return get_params_visibility(assignment_type).get("show_battery", True)

def show_vpp(assignment_type: str) -> bool:
    return get_params_visibility(assignment_type).get("show_vpp", True)

def show_tariff(assignment_type: str) -> bool:
    return get_params_visibility(assignment_type).get("show_tariff", True)
