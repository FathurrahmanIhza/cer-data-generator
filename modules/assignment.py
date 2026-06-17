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
ASSIGNMENT_PARAMS = {
    ASSIGNMENT_1: {
        "show_battery": True,
        "show_vpp":     True,
        "show_tariff":  True,
    },
    ASSIGNMENT_2: {
        "show_battery": False,
        "show_vpp":     False,
        "show_tariff":  True,
    },
}

OUTPUT_COLUMNS = {
    ASSIGNMENT_1: [
        'timestamp',
        'irradiance_W/m^2',
        'temperature_C',
        'solar_output_kw',
        'load_kW',
        'price_AUD/MWh',
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
        'solar_output_kw',
        'load_kW',
        'price_AUD/MWh',
        'tariff_import_AUD/kWh',
        'tariff_export_AUD/kWh',
    ],
}



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
    """Kembalikan dict visibilitas parameter untuk assignment tertentu."""
    return ASSIGNMENT_PARAMS.get(assignment_type, ASSIGNMENT_PARAMS[ASSIGNMENT_1])

def get_output_columns(assignment_type: str) -> list:
    """Kembalikan list kolom output CSV untuk assignment tertentu."""
    return OUTPUT_COLUMNS.get(assignment_type, OUTPUT_COLUMNS[ASSIGNMENT_1])

def show_battery(assignment_type: str) -> bool:
    return get_params_visibility(assignment_type).get("show_battery", True)

def show_vpp(assignment_type: str) -> bool:
    return get_params_visibility(assignment_type).get("show_vpp", True)

def show_tariff(assignment_type: str) -> bool:
    return get_params_visibility(assignment_type).get("show_tariff", True)
