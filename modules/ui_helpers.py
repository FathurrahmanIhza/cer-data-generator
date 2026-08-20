"""
modules/ui_helpers.py
Fungsi helper untuk rendering UI yang digunakan bersama oleh
generate flow dan regenerate flow di main.py.

Penambahan section baru: cukup edit di satu tempat — berlaku untuk keduanya.
"""

import calendar
import streamlit as st
from modules import assignment as asgn
from modules import visualizer


# ─────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────
DT_HOURS = 5.0 / 60.0   # 5-menit interval → jam


# ─────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────

def _render_sim_info(used_p: dict, vc: dict, role: str) -> None:
    """Render Generated Simulation Info panel (system specs + location)."""
    pr_pct   = f"{int(used_p.get('solar_pr', 0) * 100)}%"
    temp_val = f"{used_p.get('solar_temp', 'N/A')} / °C"

    with st.container(border=True):
        if role == 'admin':
            st.markdown(
                f"**📍 Location:** `{used_p.get('location','N/A')}` | "
                f"**🗓️ Period:** `{used_p.get('period','N/A')}` | "
                f"**🏠 Load:** `{used_p.get('load_source','N/A')}` "
                f"**(x {used_p.get('load_multiplier', 1.0)})**"
            )
            st.divider()

            if vc.get("show_battery_charts", True):
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown("#### ☀️ Solar PV")
                    st.markdown(
                        f"- Capacity: **{used_p.get('solar','N/A')} kWp**\n"
                        f"- PR: **{pr_pct}**\n"
                        f"- Temp Coeff: **{temp_val}**"
                    )
                with c2:
                    st.markdown("#### 🔋 Battery Storage")
                    st.markdown(
                        f"- Capacity: **{used_p.get('bat','N/A')} kWh**\n"
                        f"- Power: **-{used_p.get('bat_charge_kw','N/A')} / +{used_p.get('bat_discharge_kw','N/A')} kW**\n"
                        f"- Efficiency: **{int(used_p.get('bat_eff', 0)*100)}%**"
                    )
                with c3:
                    st.markdown("#### ⚡ Control Logic")
                    st.markdown(
                        f"- VPP Threshold: **{used_p.get('vpp_thresh','N/A')} AUD/MWh**\n"
                        f"- SoC Limits: **{int(used_p.get('soc_min',0)*100)}% - {int(used_p.get('soc_max',0)*100)}%**\n"
                        f"- Initial SoC: **{int(used_p.get('bat_soc_init',0)*100)}%**"
                    )
            else:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("#### ☀️ Solar PV")
                    st.markdown(
                        f"- Capacity: **{used_p.get('solar','N/A')} kWp**\n"
                        f"- PR: **{pr_pct}**\n"
                        f"- Temp Coeff: **{temp_val}**"
                    )
                with c2:
                    st.markdown("#### 🔋 Battery Storage (Auto-Sized)")
                    st.markdown(
                        f"- Capacity: **{used_p.get('bat','N/A')} kWh**\n"
                        f"- Power: **-{used_p.get('bat_charge_kw','N/A')} / +{used_p.get('bat_discharge_kw','N/A')} kW**\n"
                        f"- Mode: *Auto-sized*"
                    )
        else:
            st.markdown(
                f"**📍 Location:** `{used_p.get('location','N/A')}` | "
                f"**🗓️ Period:** `{used_p.get('period','N/A')}` | "
                f"**🏠 Load:** `{used_p.get('load_source','N/A')}` "
                f"**(x {used_p.get('load_multiplier', 1.0)})** | "
                f"**☀️ Solar PV:** PR `{pr_pct}` | Temp Coeff `{temp_val}`"
            )


def _render_tariff_details(t_data: dict) -> None:
    """Render expander tariff details."""
    with st.expander("💲 View Applied Tariff Details", expanded=False):
        schema_name = t_data.get('tariff_scheme', "Flat")

        if schema_name == "Assignment2":
            # Asgn 2: tampilkan semua 3 skema sekaligus
            st.markdown("**Flat Tariff**")
            fc1, fc2 = st.columns(2)
            fc1.markdown(f"Import: **{t_data.get('import_flat', 0.20)} AUD/kWh**")
            fc2.markdown(f"Export: **{t_data.get('export_price', 0.08)} AUD/kWh**")

            st.markdown("**Time of Use Tariff**")
            tc1, tc2 = st.columns(2)
            with tc1:
                st.markdown("Import")
                st.markdown(
                    f"- Peak: **{t_data.get('peak_price', 0.45)} AUD/kWh**\n"
                    f"- Shoulder: **{t_data.get('shoulder_price', 0.25)} AUD/kWh**\n"
                    f"- Off-Peak: **{t_data.get('offpeak_price', 0.15)} AUD/kWh**"
                )
            with tc2:
                st.markdown("Export")
                st.markdown(
                    f"- Peak: **{t_data.get('exp_peak', 0.15)} AUD/kWh**\n"
                    f"- Shoulder: **{t_data.get('exp_shoulder', 0.10)} AUD/kWh**\n"
                    f"- Off-Peak: **{t_data.get('exp_offpeak', 0.05)} AUD/kWh**"
                )

            st.markdown("**Wholesale / Spot Price**")
            st.markdown("- Import / Export: taken directly from market spot price data.")

        elif schema_name == "Wholesale Price":
            st.markdown(f"**Scheme:** `Wholesale Passthrough Price`")
            st.markdown("- **Import:** Spot Price + Market + Network + Other Fees\n- **Export:** Spot Price + Market Fees")

        else:
            display_name = schema_name
            st.markdown(f"**Scheme:** `{display_name}`")
            tc1, tc2 = st.columns(2)
            with tc1:
                st.markdown("**Export Tariff:**")
                if schema_name == "Time of Use":
                    st.markdown(
                        f"- Peak: **{t_data.get('exp_peak', 0.15)} AUD/kWh**\n"
                        f"- Shoulder: **{t_data.get('exp_shoulder', 0.10)} AUD/kWh**\n"
                        f"- Off-Peak: **{t_data.get('exp_offpeak', 0.05)} AUD/kWh**"
                    )
                else:
                    st.markdown(f"Flat Rate: **{t_data.get('export_price', 0.08)} AUD/kWh**")
            with tc2:
                st.markdown("**Import Tariff:**")
                if schema_name == "Time of Use":
                    st.markdown(
                        f"- Peak: **{t_data.get('peak_price', 0.45)} AUD/kWh**\n"
                        f"- Shoulder: **{t_data.get('shoulder_price', 0.25)} AUD/kWh**\n"
                        f"- Off-Peak: **{t_data.get('offpeak_price', 0.15)} AUD/kWh**"
                    )
                else:
                    st.markdown(f"Flat Rate: **{t_data.get('import_flat', 0.20)} AUD/kWh**")


def _render_battery_logic(used_p: dict, vc: dict, t_data: dict) -> None:
    """Render Battery Logic Flow expander (admin only, gated by vis_config)."""
    if not vc.get("show_battery_logic", True):
        return

    schema_name  = t_data.get('tariff_scheme', "Flat")
    display_name = "Wholesale Passthrough Price" if schema_name == "Wholesale Price" else schema_name
    vpp          = used_p.get('vpp_thresh', 'N/A')

    with st.expander("⚙️ View Battery Logic Flow", expanded=False):
        st.markdown(f"**Active Ruleset:** `{display_name}`\n")
        col1, col2, col3, col4 = st.columns(4)

        if schema_name == "Time of Use":
            with col1:
                st.markdown("""
                **1. Self-Consumption**
                - *Status:* Enabled (`Yes`)
                - *Conditions:* When time falls within **Peak** or **Shoulder** periods, there is a solar deficit, and battery has capacity above minimum SoC.
                """)
            with col2:
                st.markdown("""
                **2. Charge from Grid**
                - *Status:* Enabled (`Yes`)
                - *Conditions:* During **Off-peak** period, if battery SoC drops **below 30%**.
                - *Limit:* Automatically stops at exactly **30%** SoC.
                """)
            with col3:
                st.markdown("""
                **3. Hold Scenarios**
                - *Status:* Active *(Standby/Idle)*
                - *Conditions:* Solar alone can supply the load, OR during **Off-peak** periods when battery is capped at **30%**.
                """)
            with col4:
                st.markdown(f"""
                **4. VPP Dispatch Override**
                - *Status:* Emergency Override
                - *Force Discharge:* **Spot Market Price** hits VPP Threshold (**{vpp} AUD/MWh**).
                - *Force Charge:* **Spot Market Price** goes negative (**< 0 AUD/MWh**).
                """)

        elif schema_name == "Wholesale Price":
            with col1:
                st.markdown("""
                **1. Self-Consumption**
                - *Status:* Enabled (`Yes`)
                - *Conditions:* When **Export Tariff** exceeds **10 c/kWh**, there is a solar deficit, and battery has capacity above minimum SoC.
                """)
            with col2:
                st.markdown("""
                **2. Charge from Grid**
                - *Status:* Enabled (`Yes`)
                - *Conditions:* When **Export Tariff** drops below **5 c/kWh**, battery SoC is **below 30%**, and no excess solar.
                - *Limit:* Automatically stops at exactly **30%** SoC.
                """)
            with col3:
                st.markdown("""
                **3. Hold Scenarios**
                - *Status:* Active *(Standby/Idle)*
                - *Conditions:* Solar alone can supply the load, Mid-Price Zone (**5 - 10 c/kWh** of **Export Tariff**), OR Low **Export Tariff** (**< 5 c/kWh**) when battery is capped at **30%**.
                """)
            with col4:
                st.markdown(f"""
                **4. VPP Override**
                - *Status:* Emergency Override
                - *Force Discharge:* **Spot Market Price** hits VPP Threshold (**{vpp} AUD/MWh**).
                - *Force Charge:* **Spot Market Price** goes negative (**< 0 AUD/MWh**).
                """)

        else:  # Flat
            with col1:
                st.markdown("""
                **1. Self-Consumption**
                - *Status:* Enabled (`Yes`)
                - *Conditions:* Whenever load needs supply (solar deficit) and battery has capacity above minimum SoC.
                """)
            with col2:
                st.markdown("""
                **2. Charge from Grid**
                - *Status:* Disabled (`No`)
                - *Conditions:* Battery will never charge from the grid under normal baseline operations.
                """)
            with col3:
                st.markdown("""
                **3. Hold Scenarios**
                - *Status:* Active *(Standby/Idle)*
                - *Conditions:* Solar alone is completely sufficient to cover the household load.
                """)
            with col4:
                st.markdown(f"""
                **4. VPP Dispatch Override**
                - *Status:* Emergency Override
                - *Force Discharge:* **Spot Market Price** hits VPP Threshold (**{vpp} AUD/MWh**).
                - *Force Charge:* **Spot Market Price** goes negative (**< 0 AUD/MWh**).
                """)


def _render_analysis(df_result, vc: dict, year_selectbox_key: str, month_selectbox_key: str) -> None:
    """Render Detailed Analysis section: metrics, annual overview, monthly profile."""
    ts       = df_result['timestamp']
    yr_arr   = ts.dt.year
    mo_arr   = ts.dt.month

    col_load = 'load_profile' if 'load_profile' in df_result.columns else 'beban_rumah_kw'
    col_bat  = 'battery_power_ac_kw' if 'battery_power_ac_kw' in df_result.columns else 'battery_power_kw'

    @st.fragment
    def _analysis_fragment():
        available_years = sorted(yr_arr.unique())
        selected_year   = st.selectbox("Select Year:", available_years, key=year_selectbox_key)

        mask_year = yr_arr == selected_year
        df_year   = df_result[mask_year] 

        total_solar = df_year['solar_output_kw'].sum() * DT_HOURS
        total_load  = df_year[col_load].sum() * DT_HOURS

        if vc.get("show_grid_metric", True) and 'grid_net_kw' in df_year.columns:
            total_import = df_year['grid_net_kw'].clip(lower=0).sum() * DT_HOURS
            m1, m2, m3 = st.columns(3)
            m1.metric(f"Total Solar ({selected_year})", f"{total_solar:,.2f} kWh")
            m2.metric(f"Total Load ({selected_year})",  f"{total_load:,.2f} kWh")
            m3.metric(f"Grid Import ({selected_year})", f"{total_import:,.2f} kWh", delta_color="inverse")
        else:
            m1, m2 = st.columns(2)
            m1.metric(f"Total Solar ({selected_year})", f"{total_solar:,.2f} kWh")
            m2.metric(f"Total Load ({selected_year})",  f"{total_load:,.2f} kWh")

        visualizer.plot_annual_overview(df_year, col_bat, selected_year, vis_config=vc)

        st.divider()

        if vc.get("show_monthly_analysis", True):
            mo_arr_year = mo_arr[mask_year]

            @st.fragment
            def _monthly_fragment():
                available_months = sorted(mo_arr_year.unique())
                month_map        = {m: calendar.month_name[m] for m in available_months}

                selected_month_name = st.selectbox(
                    "Select Month for Profile:", list(month_map.values()), key=month_selectbox_key
                )
                selected_month = [k for k, v in month_map.items() if v == selected_month_name][0]
                df_month = df_year[mo_arr_year == selected_month]

                visualizer.plot_monthly_analysis(df_month, col_load, selected_month_name, selected_year)

            _monthly_fragment()

    _analysis_fragment()


def render_result_panel(
    df_result,
    used_p:                dict,
    vc:                    dict,
    csv_bytes:             bytes,
    csv_bytes_partial:     bytes = None,           # NEW: Partial CSV (tahun 2+ dikosongkan)
    download_label:        str   = "Download Dataset (CSV)",
    download_filename:     str   = "data.csv",
    download_key:          str   = "dl_result",
    download_key_partial:  str   = "dl_result_partial",  # NEW
    year_key:              str   = "sb_year",
    month_key:             str   = "sb_month",
    show_analysis:         bool  = True,
) -> None:
    """
    Render panel hasil simulasi secara lengkap.
    Dipanggil dari generate flow maupun regen flow — logika identik.

    Parameters
    ----------
    df_result            : DataFrame hasil simulasi (kolom internal, bukan renamed)
    used_p               : dict parameter simulasi (dari session_state['used_params'])
    vc                   : dict vis_config dari asgn.get_vis_config(assignment_type)
    csv_bytes            : bytes CSV full (semua baris berisi data)
    csv_bytes_partial    : bytes CSV partial untuk Assignment 2 (tahun ke-2+ dikosongkan);
                           None berarti tidak ditampilkan (Assignment 1 / backward compat)
    download_label       : label tombol download (Full)
    download_filename    : nama file CSV (Full)
    download_key         : unique key untuk st.download_button (Full)
    download_key_partial : unique key untuk st.download_button (Partial)
    year_key             : unique key untuk selectbox tahun di analysis
    month_key            : unique key untuk selectbox bulan di analysis
    show_analysis        : True = tampilkan Detailed Analysis section (admin only)
    """
    role   = st.session_state.get('role', 'student')
    t_data = used_p.get('tariff_data', {})

    st.divider()
    st.markdown("### 📋 Generated Simulation Info")
    _render_sim_info(used_p, vc, role)
    _render_tariff_details(t_data)

    if role == 'admin':
        _render_battery_logic(used_p, vc, t_data)

    st.markdown("### 💾 Export Data")

    if csv_bytes_partial is not None:
        # Assignment 2
        if role == 'admin':
            # Admin: 2 tombol — Full + Partial
            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                st.download_button(
                    label="Download Dataset (CSV)",
                    data=csv_bytes,
                    file_name=download_filename,
                    mime="text/csv",
                    key=download_key,
                )
            with dl_col2:
                _partial_filename = download_filename.replace(".csv", "_partial.csv")
                st.download_button(
                    label="Download Partial Dataset (CSV)",
                    data=csv_bytes_partial,
                    file_name=_partial_filename,
                    mime="text/csv",
                    key=download_key_partial,
                )
        else:
            # Student: 1 tombol, label biasa tapi isinya partial
            _partial_filename = download_filename.replace(".csv", "_partial.csv")
            st.download_button(
                label="Download Dataset (CSV)",
                data=csv_bytes_partial,
                file_name=_partial_filename,
                mime="text/csv",
                key=download_key,
            )
    else:
        # Assignment 1 (atau default): 1 tombol — full CSV
        st.download_button(
            label=download_label,
            data=csv_bytes,
            file_name=download_filename,
            mime="text/csv",
            key=download_key,
        )

    if show_analysis and role == 'admin' and df_result is not None:
        st.divider()
        st.subheader("📊 Detailed Analysis")
        _render_analysis(df_result, vc, year_key, month_key)

