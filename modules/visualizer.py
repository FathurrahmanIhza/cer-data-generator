import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import numpy as np
import calendar
import pandas as pd

def plot_annual_overview(df_vis_year, col_bat, selected_vis_year, vis_config: dict = None):
    # Default: tampilkan semua chart (perilaku Assignment 1)
    if vis_config is None:
        vis_config = {
            "show_battery_charts":   True,
            "show_vpp_charts":       True,
            "show_monthly_analysis": True,
            "show_row5":             True,
        }
    _show_bat = vis_config.get("show_battery_charts", True)
    _show_vpp = vis_config.get("show_vpp_charts", True)
    _show_row5 = vis_config.get("show_row5", True)
    DT_HOURS = 5.0 / 60.0

    col_load = 'load_profile' if 'load_profile' in df_vis_year.columns else 'beban_rumah_kw'

    if not isinstance(df_vis_year.index, pd.DatetimeIndex):
        df_calc = df_vis_year.set_index('timestamp')
    else:
        df_calc = df_vis_year

    df_calc = df_calc.copy()
    df_calc["month"] = df_calc.index.month
    df_calc["hour"]  = df_calc.index.hour


    if col_bat in df_calc.columns:
        df_calc['battery_discharge_kw'] = df_calc[col_bat].clip(lower=0)
    else:
        df_calc['battery_discharge_kw'] = 0.0

    # --- Kolom energi (kW) yang akan di-sum lalu ×DT_HOURS ---
    energy_cols_kw = {
        'solar_output_kw':                    'solar_output_kwh',
        'battery_discharge_kw':               'battery_discharge_kwh',
        'grid_import_kw':                     'grid_import_kwh',
        'grid_export_kw':                     'grid_export_kwh',
        col_load:                             'load_kwh',
        'vpp_grid_import_after_discharge_kw': 'extra_import_kwh',
    }
    if 'vpp_battery_discharge_kw' in df_calc.columns:
        energy_cols_kw['vpp_battery_discharge_kw'] = 'vpp_bat_dis_kwh_tmp'
    if 'vpp_grid_export_kw' in df_calc.columns:
        energy_cols_kw['vpp_grid_export_kw'] = 'vpp_grid_export_kwh'

    # Kolom moneter (AUD) — sum langsung tanpa ×DT_HOURS karena sudah dalam satuan AUD per baris
    monetary_cols = [c for c in [
        'bill_actual', 'bill_solar_only', 'bill_grid_only',
        'vpp_export_value_AUD', 'vpp_extra_import_cost_AUD', 'vpp_operational_net_value_AUD',
    ] if c in df_calc.columns]

    # Kolom untuk heatmap (sudah dalam satuan waktu/energi per 5-menit, di-sum per jam)
    heatmap_raw_cols = {}
    if 'vpp_status' in df_calc.columns:
        df_calc['vpp_discharge_hours_raw'] = df_calc['vpp_status'].astype(int) * DT_HOURS
        heatmap_raw_cols['vpp_discharge_hours_raw'] = 'vpp_discharge_hours'
    if 'vpp_grid_import_after_discharge_kw' in df_calc.columns:
        df_calc['extra_import_kwh_raw'] = df_calc['vpp_grid_import_after_discharge_kw'] * DT_HOURS
        heatmap_raw_cols['extra_import_kwh_raw'] = 'extra_import_kwh_hm'

    # Siapkan semua kolom kW mentah
    src_kw_cols = list(energy_cols_kw.keys())

    # ── STEP 1: 5-min → Hourly ──
    agg_dict_hourly = {c: 'sum' for c in src_kw_cols + list(heatmap_raw_cols.keys()) + monetary_cols
                       if c in df_calc.columns}
    if 'vpp_status' in df_calc.columns:
        agg_dict_hourly['vpp_status'] = 'max'

    hourly = df_calc.resample('h').agg(agg_dict_hourly)

    # Konversi kW → kWh di level hourly (×DT_HOURS)
    for src, dst in energy_cols_kw.items():
        if src in hourly.columns:
            hourly[dst] = hourly[src] * DT_HOURS
    for src, dst in heatmap_raw_cols.items():
        if src in hourly.columns:
            hourly[dst] = hourly[src]   

    if 'battery_discharge_kwh' in hourly.columns:
        hourly['battery_discharge_kwh'] = hourly['battery_discharge_kwh'].clip(lower=0)

    hourly = hourly.round(2)

    # ── STEP 2: Hourly → Daily ──
    energy_dst_cols = list(energy_cols_kw.values()) + monetary_cols
    heatmap_dst_cols = list(heatmap_raw_cols.values())
    agg_dict_daily = {c: 'sum' for c in energy_dst_cols + heatmap_dst_cols if c in hourly.columns}
    if 'vpp_status' in hourly.columns:
        agg_dict_daily['vpp_status'] = 'max'

    daily = hourly.resample('D').agg(agg_dict_daily)
    daily = daily.round(2)

    # ── STEP 3: Daily → Monthly ──
    monthly = daily.resample('ME').agg({c: 'sum' for c in agg_dict_daily.keys()})
    monthly = monthly.round(2)

    months_labels = [d.strftime("%b") for d in monthly.index]

    _hm_month = df_calc.index.month
    _hm_hour  = df_calc.index.hour
    if 'vpp_discharge_hours_raw' in df_calc.columns:
        heatmap_vpp = pd.pivot_table(
            df_calc.assign(_month=_hm_month, _hour=_hm_hour),
            index='_month', columns='_hour',
            values='vpp_discharge_hours_raw', aggfunc='sum'
        ).reindex(index=range(1, 13), columns=range(24)).fillna(0)
    else:
        heatmap_vpp = pd.DataFrame(0, index=range(1, 13), columns=range(24))
    if 'extra_import_kwh_raw' in df_calc.columns:
        heatmap_imp = pd.pivot_table(
            df_calc.assign(_month=_hm_month, _hour=_hm_hour),
            index='_month', columns='_hour',
            values='extra_import_kwh_raw', aggfunc='sum'
        ).reindex(index=range(1, 13), columns=range(24)).fillna(0)
    else:
        heatmap_imp = pd.DataFrame(0, index=range(1, 13), columns=range(24))

    factor = DT_HOURS
    
    # --- ROW 1 Prep ---
    if 'battery_discharge_kwh' not in monthly.columns:
        monthly['battery_discharge_kwh'] = 0.0
    if 'grid_import_kwh' not in monthly.columns:
        monthly['grid_import_kwh'] = 0.0

    elec_cols = ["solar_output_kwh", "battery_discharge_kwh", "grid_import_kwh"]
    monthly_pct = (monthly[elec_cols].div(monthly[elec_cols].sum(axis=1), axis=0) * 100)

    annual_load = monthly["load_kwh"].sum()
    annual_pv   = monthly["solar_output_kwh"].sum()

    # --- ROW 3 Prep (Price Profile — dari data 5-menit asli) ---
    col_price = 'price_profile' if 'price_profile' in df_calc.columns else 'price_import'
    price_series = df_calc[col_price] if col_price in df_calc.columns else pd.Series(dtype=np.float64)
    vpp_mask_price = df_calc['vpp_status'] > 0 if 'vpp_status' in df_calc.columns else pd.Series(False, index=df_calc.index)
    dispatch_price_threshold = df_calc.loc[vpp_mask_price, col_price].min() if vpp_mask_price.any() else np.inf
    mask_price_pos  = (price_series > 0) & (price_series < dispatch_price_threshold)
    mask_price_neg  = price_series < 0
    mask_price_disp = price_series >= dispatch_price_threshold

    # --- ROW 4 Prep (Cumulative VPP — dari daily yang sudah di-resample) ---
    threshold_contract = 1000
    if 'vpp_grid_export_kwh' in daily.columns:
        cumulative_vpp = daily['vpp_grid_export_kwh'].cumsum()
    elif 'vpp_grid_export_kw' in df_calc.columns:
        cumulative_vpp = df_calc['vpp_grid_export_kw'].resample('D').sum().cumsum() * DT_HOURS
    else:
        cumulative_vpp = pd.Series(0, index=daily.index)

    # --- ROW 5 Prep ---
    monthly["self_consumption_pct"] = ((monthly["solar_output_kwh"] - monthly["grid_export_kwh"]) / monthly["solar_output_kwh"].replace(0, np.nan)) * 100
    monthly["self_sufficiency_pct"] = (1 - (monthly["grid_import_kwh"] / monthly["load_kwh"].replace(0, np.nan))) * 100
    monthly.fillna(0, inplace=True)

    # --- ROW 6 Prep (Dispatch events — dari data 5-menit asli) ---
    event_df = pd.DataFrame()
    if 'vpp_status' in df_calc.columns:
        vpp_mask_event = df_calc['vpp_status'] > 0
        df_calc['event_id'] = (vpp_mask_event != vpp_mask_event.shift()).cumsum()
        vpp_events_data = df_calc[vpp_mask_event]
        if not vpp_events_data.empty:
            grouped_events = vpp_events_data.groupby('event_id')
            duration_h = grouped_events.size() * DT_HOURS
            bat_power_kw = df_calc[col_bat].abs().max() or 15.0
            requested_vpp_kwh = duration_h * bat_power_kw
            vpp_dis_col = 'vpp_battery_discharge_kw' if 'vpp_battery_discharge_kw' in df_calc.columns else col_bat
            actual_vpp_discharge_kwh = grouped_events[vpp_dis_col].sum() * DT_HOURS
            dispatch_limited = actual_vpp_discharge_kwh < (requested_vpp_kwh - 0.1)
            event_df = pd.DataFrame({
                'requested_vpp_kwh': requested_vpp_kwh,
                'actual_vpp_discharge_kwh': actual_vpp_discharge_kwh,
                'dispatch_limited': dispatch_limited
            })

    vpp_bat_col = 'vpp_bat_dis_kwh_tmp' if 'vpp_bat_dis_kwh_tmp' in monthly.columns else None
    monthly["normal_battery_discharge_kwh"] = (
        (monthly["battery_discharge_kwh"] - monthly[vpp_bat_col]).clip(lower=0)
        if vpp_bat_col else monthly["battery_discharge_kwh"]
    )

    # --- ROW 7 Prep (VPP Financials) ---
    monthly["vpp_payment"] = 20.0
    if "vpp_operational_net_value_AUD" in monthly.columns:
        monthly["net_cost"]     = monthly["vpp_extra_import_cost_AUD"] - monthly["vpp_export_value_AUD"]
        total_extra_import_cost = monthly["vpp_extra_import_cost_AUD"].sum()
        total_export_value      = monthly["vpp_export_value_AUD"].sum()
        contract_payment        = monthly["vpp_payment"].sum()
        total_net_cost          = total_extra_import_cost - total_export_value - contract_payment
        after_export_value      = total_extra_import_cost - total_export_value
        after_contract          = after_export_value - contract_payment

    # --- ROW 8 Prep (Bill Comparisons) — mengikuti formula notebook ---
    if "bill_actual" in monthly.columns:
        monthly["bill_pv_battery_with_vpp_payment"] = monthly["bill_actual"] - monthly["vpp_payment"]
        monthly["bill_pv_battery_no_vpp"] = (
            monthly["bill_actual"] + monthly["vpp_operational_net_value_AUD"]
        )
        bill_cols   = ["bill_pv_battery_with_vpp_payment", "bill_pv_battery_no_vpp", "bill_solar_only", "bill_grid_only"]
        labels_bill = ["PV + Battery + VPP", "PV + Battery (No VPP)", "Solar Only", "No Battery & Solar"]
        colors_bill = ["#1f77b4", "#9467bd", "#ff7f0e", "#2ca02c"]
        yearly_bill_values = [monthly[col].sum() for col in bill_cols]


    # Render Area Yearly
    st.markdown(f"### 📅 Annual Overview ({selected_vis_year})")

    # ============================================================
    # ROW 1: Monthly Contributions (Absolut | Percentage)
    # ============================================================
    
    c1, c2 = st.columns(2, gap="large")
    colors_src = ["#FFD166", "#06D6A0", "#EF476F"]
    labels_src = ["PV Generation", "Battery Discharge", "Grid Import"]

    with c1:
        fig1, ax1 = plt.subplots(figsize=(6.5, 4.2))
        ax1.bar(months_labels, monthly['solar_output_kwh'], color=colors_src[0], label=labels_src[0], width=0.8)
        if _show_bat:
            ax1.bar(months_labels, monthly['battery_discharge_kwh'], bottom=monthly['solar_output_kwh'], color=colors_src[1], label=labels_src[1], width=0.8)
            ax1.bar(months_labels, monthly['grid_import_kwh'], bottom=monthly['solar_output_kwh']+monthly['battery_discharge_kwh'], color=colors_src[2], label=labels_src[2], width=0.8)
        else:
            ax1.bar(months_labels, monthly['grid_import_kwh'], bottom=monthly['solar_output_kwh'], color=colors_src[2], label=labels_src[2], width=0.8)
        ax1.plot(months_labels, monthly['load_kwh'], color="black", marker="o", linewidth=2.5, label="Load")
        
        ax1.set_title("Monthly Energy Contributions vs Load (kWh)")
        ax1.set_ylabel("Energy (kWh)"); ax1.set_xlabel("Month")
        
        ax1.legend(title="Energy Source", fontsize='small', loc='upper center') 
        
        ax1.text(
            0.98, 0.95,
            f"Annual Load: {annual_load:,.0f} kWh\n"
            f"Annual PV: {annual_pv:,.0f} kWh",
            transform=ax1.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.8)
        )
        
        ax1.grid(axis='y', alpha=0.3); ax1.margins(x=0.02)
        plt.tight_layout(); st.pyplot(fig1); plt.close(fig1)

    # --- KOLOM KANAN: Grafik Persentase ---
    with c2:
        fig2, ax2 = plt.subplots(figsize=(6.5, 4.2))
        ax2.bar(months_labels, monthly_pct['solar_output_kwh'], color=colors_src[0], label=labels_src[0], width=0.8)
        if _show_bat:
            ax2.bar(months_labels, monthly_pct['battery_discharge_kwh'], bottom=monthly_pct['solar_output_kwh'], color=colors_src[1], label=labels_src[1], width=0.8)
            ax2.bar(months_labels, monthly_pct['grid_import_kwh'], bottom=monthly_pct['solar_output_kwh']+monthly_pct['battery_discharge_kwh'], color=colors_src[2], label=labels_src[2], width=0.8)
        else:
            ax2.bar(months_labels, monthly_pct['grid_import_kwh'], bottom=monthly_pct['solar_output_kwh'], color=colors_src[2], label=labels_src[2], width=0.8)
        ax2.set_title("Monthly Energy Contributions (%)")
        ax2.set_ylabel("Percentage (%)"); ax2.set_xlabel("Month")
        ax2.set_ylim(0, 100)
        ax2.legend(title="Energy Source", fontsize='small', loc='lower right')
        ax2.grid(axis='y', alpha=0.3); ax2.margins(x=0.02)
        plt.tight_layout(); st.pyplot(fig2); plt.close(fig2)

    st.divider()

    # ============================================================
    # ROW 2: Heatmaps (VPP Discharge | Extra Import)
    # ============================================================
    if 'vpp_status' in df_calc.columns:
        c3, c4 = st.columns(2, gap="large")
        with c3:
            fig_vpp, ax_vpp = plt.subplots(figsize=(6.5, 4.2))
            im_vpp = ax_vpp.imshow(heatmap_vpp, aspect="auto", cmap="Oranges")
            ax_vpp.set_title("VPP Discharge Hours")
            ax_vpp.set_xlabel("Hour of Day"); ax_vpp.set_ylabel("Month")
            ax_vpp.set_xticks(np.arange(24)); ax_vpp.set_xticklabels(np.arange(24), fontsize=8)
            ax_vpp.set_yticks(np.arange(12)); ax_vpp.set_yticklabels([calendar.month_abbr[i] for i in range(1, 13)], fontsize=9)
            cbar_vpp = ax_vpp.figure.colorbar(im_vpp, ax=ax_vpp); cbar_vpp.set_label("Total Hours")
            plt.tight_layout(); st.pyplot(fig_vpp); plt.close(fig_vpp)
            
        with c4:
            fig_imp, ax_imp = plt.subplots(figsize=(6.5, 4.2))
            im_imp = ax_imp.imshow(heatmap_imp, aspect="auto", cmap="Reds")
            ax_imp.set_title("Extra Import Energy (kWh)")
            ax_imp.set_xlabel("Hour of Day")
            ax_imp.set_xticks(np.arange(24)); ax_imp.set_xticklabels(np.arange(24), fontsize=8)
            ax_imp.set_yticks(np.arange(12)); ax_imp.set_yticklabels([calendar.month_abbr[i] for i in range(1, 13)], fontsize=9)
            cbar_imp = ax_imp.figure.colorbar(im_imp, ax=ax_imp); cbar_imp.set_label("Total Energy (kWh)")
            plt.tight_layout(); st.pyplot(fig_imp); plt.close(fig_imp)
            
        st.divider()

    # ============================================================
    # ROW 3: Electricity Spot Market Price Profile 
    # ============================================================
    if col_price in df_calc.columns:
        fig_price, ax_p = plt.subplots(figsize=(14, 4)) 
        if mask_price_pos.any(): ax_p.vlines(price_series.index[mask_price_pos], 0, price_series[mask_price_pos], color='#2FBF71', alpha=0.6, linewidth=1.5, label='Positive Price')
        if mask_price_neg.any(): ax_p.vlines(price_series.index[mask_price_neg], 0, price_series[mask_price_neg], color='#E76F51', alpha=0.6, linewidth=1.5, label='Negative Price')
        if mask_price_disp.any(): ax_p.vlines(price_series.index[mask_price_disp], 0, price_series[mask_price_disp], color='#7B2CBF', alpha=0.6, linewidth=1.5, label='Dispatch Price Event')
        if not np.isinf(dispatch_price_threshold): ax_p.axhline(dispatch_price_threshold, color='#7B2CBF', linestyle='--', linewidth=1.5, label='Dispatch Price Threshold')
        ax_p.xaxis.set_major_locator(mdates.MonthLocator()); ax_p.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax_p.margins(x=0); ax_p.set_title('Electricity Spot Market Price (5 Minutes)'); ax_p.set_ylabel('Price (AUD)')
        ax_p.grid(True, alpha=0.25); ax_p.legend(loc='upper right', fontsize='small')
        plt.tight_layout(); st.pyplot(fig_price); plt.close(fig_price)
        st.divider()

    # ============================================================
    # ROW 4: Cumulative VPP Discharge 
    # ============================================================
    if 'vpp_grid_export_kw' in df_calc.columns:
        fig_cum, ax_cum = plt.subplots(figsize=(14, 4))
        ax_cum.plot(cumulative_vpp.index, cumulative_vpp, linestyle='-', linewidth=2.5, color='blue', label='Cumulative VPP Discharge')
        ax_cum.fill_between(cumulative_vpp.index, cumulative_vpp, threshold_contract, where=(cumulative_vpp <= threshold_contract), color='green', alpha=0.3, label='Below limit')
        ax_cum.fill_between(cumulative_vpp.index, cumulative_vpp, threshold_contract, where=(cumulative_vpp > threshold_contract), color='red', alpha=0.3, label='Above limit')
        ax_cum.axhline(threshold_contract, color='black', linestyle='--', label=f'Contract limit {threshold_contract} kWh')
        ax_cum.set_title('Cumulative VPP Discharge'); ax_cum.set_ylabel('Energy discharged (kWh)')
        ax_cum.xaxis.set_major_locator(mdates.MonthLocator()); ax_cum.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax_cum.margins(x=0); ax_cum.grid(True, alpha=0.3); ax_cum.legend(loc='upper left', fontsize='small')
        plt.tight_layout(); st.pyplot(fig_cum); plt.close(fig_cum)
        st.divider()

    # ============================================================
    # ROW 5: Monthly Self Consumption, Sufficiency, PV Gen & VPP
    # ============================================================
    if _show_row5:
        fig_ss, ax1_ss = plt.subplots(figsize=(14, 4))
        x_m = np.arange(len(months_labels))
        ax1_ss.plot(x_m, monthly["self_consumption_pct"], marker="o", linewidth=2, label="PV Self-Consumption")
        ax1_ss.plot(x_m, monthly["self_sufficiency_pct"], marker="s", linewidth=2, label="Home Self-Sufficiency")
        ax1_ss.set_ylabel("Percentage (%)")
        ax1_ss.set_ylim(0, 110)
        ax1_ss.set_xticks(x_m); ax1_ss.set_xticklabels(months_labels); ax1_ss.set_xlabel("Month")
        ax1_ss.axvspan(4.5, 6.5, color="grey", alpha=0.18, label="High VPP Activity Period")
        
        ax2_ss = ax1_ss.twinx()
        bar_width = 0.35
        
        ax2_ss.bar(
            x_m - bar_width/2, 
            monthly["solar_output_kwh"], 
            alpha=0.4, 
            color="orange", 
            width=bar_width, 
            label="PV Generation"
        )
        
        _vpp_bar_data = monthly["vpp_bat_dis_kwh_tmp"] if "vpp_bat_dis_kwh_tmp" in monthly.columns else pd.Series(0, index=monthly.index)
        ax2_ss.bar(
            x_m + bar_width/2, 
            _vpp_bar_data, 
            alpha=0.6, 
            color="red", 
            width=bar_width, 
            label="VPP Discharge"
        )
        
        ax2_ss.set_ylabel("Energy (kWh)")
        
        lines1_ss, labels1_ss = ax1_ss.get_legend_handles_labels()
        lines2_ss, labels2_ss = ax2_ss.get_legend_handles_labels()
        ax1_ss.legend(lines1_ss + lines2_ss, labels1_ss + labels2_ss, loc="lower left", fontsize='small')
        
        ax1_ss.set_title("Monthly Self-Consumption, Self-Sufficiency, PV Generation, and VPP Dispatch Activity")
        ax1_ss.grid(alpha=0.3); ax1_ss.margins(x=0.02)
        
        plt.tight_layout()
        st.pyplot(fig_ss)
        plt.close(fig_ss)
        st.divider()

    # ============================================================
    # ROW 6: Request vs Actual Dispatch | Battery Breakdown 
    # ============================================================
    if 'vpp_status' in df_calc.columns and not event_df.empty:
        c6_1, c6_2 = st.columns(2, gap="large")
        
        with c6_1:
            fig_sc, ax_sc = plt.subplots(figsize=(6.5, 4.2))
            normal = event_df[~event_df["dispatch_limited"]]; limited = event_df[event_df["dispatch_limited"]]
            
            ax_sc.scatter(normal["requested_vpp_kwh"], normal["actual_vpp_discharge_kwh"], color="steelblue", edgecolor="black", alpha=0.75, s=70, linewidth=0.8, label="Full Dispatch Achieved", zorder=10)
            ax_sc.scatter(limited["requested_vpp_kwh"], limited["actual_vpp_discharge_kwh"], color="red", edgecolor="black", alpha=1.0, s=130, linewidth=1.5, marker="X", label="Dispatch Limited", zorder=5)
            
            max_val = max(event_df["requested_vpp_kwh"].max(), event_df["actual_vpp_discharge_kwh"].max())
            ax_sc.plot([0, max_val], [0, max_val], linestyle="--", color="black", alpha=0.6, linewidth=1.5, label="Requested = Actual", zorder=1)
            
            ax_sc.set_xlabel("Requested VPP Energy (kWh)")
            ax_sc.set_ylabel("Actual VPP Discharge (kWh)")
            ax_sc.set_title("Requested vs Actual VPP Dispatch Energy")
            ax_sc.legend(fontsize='small'); ax_sc.grid(alpha=0.3)
            plt.tight_layout(); st.pyplot(fig_sc); plt.close(fig_sc)

        with c6_2:
            fig_bar_dis, ax_bar_dis = plt.subplots(figsize=(6.5, 4.2))
            _vpp_col = "vpp_bat_dis_kwh_tmp" if "vpp_bat_dis_kwh_tmp" in monthly.columns else None
            if _vpp_col:
                monthly[["normal_battery_discharge_kwh", _vpp_col]].plot(
                    kind="bar", stacked=True, ax=ax_bar_dis, color=["skyblue", "orange"], width=0.8
                )
                ax_bar_dis.legend(["Normal Discharge", "VPP Discharge"], title="Source", fontsize='small')
            else:
                monthly[["normal_battery_discharge_kwh"]].plot(
                    kind="bar", ax=ax_bar_dis, color=["skyblue"], width=0.8
                )
                ax_bar_dis.legend(["Normal Discharge"], title="Source", fontsize='small')
            ax_bar_dis.set_title("Monthly Battery Discharge Breakdown")
            ax_bar_dis.set_ylabel("Energy (kWh)")
            ax_bar_dis.set_xlabel("Month")
            ax_bar_dis.set_xticklabels(months_labels, rotation=0)
            ax_bar_dis.grid(axis="y", alpha=0.3)
            plt.tight_layout(); st.pyplot(fig_bar_dis); plt.close(fig_bar_dis)
            
        st.divider()

    # ============================================================
    # ROW 7: Monthly VPP Export Value | Annual VPP Financial (2 Kolom)
    # ============================================================
    if 'vpp_export_value_AUD' in monthly.columns:
        c7_1, c7_2 = st.columns(2, gap="large")
        
        with c7_1:
            x_pos = np.arange(len(months_labels))
            colors_econ = np.where(monthly["net_cost"] <= 0, "#55A868", np.where(monthly["net_cost"] <= monthly["vpp_payment"], "#DD8452", "#C44E52"))
            
            fig_econ, ax_econ = plt.subplots(figsize=(6.5, 4.2))
            
            ax_econ.bar(x_pos, monthly["net_cost"], color=colors_econ, width=0.7)
            
            line1, = ax_econ.plot(x_pos, monthly["vpp_payment"], color="black", linestyle="--", linewidth=2)
            line2, = ax_econ.plot(x_pos, monthly["vpp_extra_import_cost_AUD"], marker="o", linestyle="-", linewidth=2)
            line3, = ax_econ.plot(x_pos, monthly["vpp_export_value_AUD"], color="orange", marker="s", linestyle="-", linewidth=2)
            
            # 2. Simpan handle untuk Dummy Bars (akan ditaruh di Kanan)
            bar1 = ax_econ.bar(np.nan, np.nan, color="#55A868")
            bar2 = ax_econ.bar(np.nan, np.nan, color="#DD8452")
            bar3 = ax_econ.bar(np.nan, np.nan, color="#C44E52")
            
            ax_econ.set_xticks(x_pos)
            ax_econ.set_xticklabels(months_labels)
            ax_econ.set_ylabel("Value ($)")
            ax_econ.set_title("Monthly VPP Export Value vs Extra Import Cost")
            
            # ============================================================
            # PROSES MEMECAH LEGEND
            # ============================================================
            
            # Legend 1: Sisi Kiri (Upper Left) - Untuk Line data
            legend_left = ax_econ.legend(
                handles=[line1, line2, line3],
                labels=["VPP Payment", "Extra Import Cost ($)", "VPP Export Value ($)"],
                fontsize='small', 
                loc='upper left'
            )
            # Kunci utama: tambahkan legend pertama ke grafik agar tidak terhapus
            ax_econ.add_artist(legend_left)
            
            # Legend 2: Sisi Kanan (Upper Right) - Untuk Keterangan Warna Bar
            ax_econ.legend(
                handles=[bar1, bar2, bar3],
                labels=["Net Cost Negative", "Net Cost Positive", "Net Cost > VPP Subscription"],
                fontsize='small', 
                loc='upper right'
            )
            
            # ============================================================
            
            ax_econ.grid(axis='y', linestyle='--', alpha=0.5)
            ax_econ.margins(x=0.02)
            
            plt.tight_layout()
            st.pyplot(fig_econ)
            plt.close(fig_econ)

        with c7_2:
            fig_fin, ax_fin = plt.subplots(figsize=(6.5, 4.2))
            ax_fin.bar(0, total_extra_import_cost, color="#D55E00", label="Extra Import Cost")
            ax_fin.bar(1, -total_export_value, bottom=total_extra_import_cost, color="#4C72B0", label="VPP Export Value")
            ax_fin.bar(2, -contract_payment, bottom=after_export_value, color="#F0E442", label="VPP Contract Payment")
            ax_fin.bar(3, total_net_cost, color="#55A868" if total_net_cost <= 0 else "#C44E52", label="Net Cost")
            
            ax_fin.set_xticks([0, 1, 2, 3])
            ax_fin.set_xticklabels(["Extra Import\nCost", "Export\nValue", "VPP\nPayment", "Net\nCost"])
            ax_fin.set_ylabel("Total Value ($)")
            ax_fin.set_title("Annual VPP Financial Summary")
            
            ax_fin.text(0, total_extra_import_cost, f"{total_extra_import_cost:.1f}", ha="center", va="bottom")
            ax_fin.text(1, after_export_value, f"-{total_export_value:.1f}", ha="center", va="top")
            ax_fin.text(2, after_contract, f"-{contract_payment:.1f}", ha="center", va="top")
            ax_fin.text(3, total_net_cost, f"{total_net_cost:.1f}", ha="center", va="bottom" if total_net_cost >= 0 else "top")
            
            ax_fin.axhline(0, color="black", linewidth=1); ax_fin.grid(axis="y", alpha=0.3)
            ax_fin.legend(fontsize='small', loc='upper right')
            plt.tight_layout(); st.pyplot(fig_fin); plt.close(fig_fin)
            
        st.divider()

    # ============================================================
    # ROW 8: Monthly Bill Comparison | Yearly Bill Comparison (2 Kolom)
    # ============================================================
    if 'bill_actual' in monthly.columns:
        c8_1, c8_2 = st.columns(2, gap="large")
        
        with c8_1:
            x_b = np.arange(len(months_labels))
            width = 0.2
            
            fig_m_bill, ax_m_bill = plt.subplots(figsize=(6.5, 4.2))
            for i, col in enumerate(bill_cols):
                ax_m_bill.bar(x_b + (i - 1.5) * width, monthly[col], width, label=labels_bill[i], color=colors_bill[i])
                
            ax_m_bill.set_xticks(x_b); ax_m_bill.set_xticklabels(months_labels)
            ax_m_bill.axhline(0, color="black", linewidth=1); ax_m_bill.set_ylabel("Bill ($)"); ax_m_bill.set_xlabel("Month")
            ax_m_bill.set_title("Monthly Electricity Bill Comparison")
            ax_m_bill.legend(title="Scenario", fontsize='small'); ax_m_bill.grid(axis="y", alpha=0.3)
            plt.tight_layout(); st.pyplot(fig_m_bill); plt.close(fig_m_bill)
            
        with c8_2:
            fig_y_bill, ax_y_bill = plt.subplots(figsize=(6.5, 4.2))
            bars = ax_y_bill.bar(labels_bill, yearly_bill_values, color=colors_bill, width=0.6)
            
            padding = max(max(np.abs(yearly_bill_values)) * 0.15, 10)
            ax_y_bill.set_ylim(min(yearly_bill_values) - padding, max(yearly_bill_values) + padding)
            ax_y_bill.axhline(0, color="black", linewidth=1)
            
            offset = max(np.abs(yearly_bill_values)) * 0.03
            for bar in bars:
                height = bar.get_height()
                y_text = height + offset if height >= 0 else height - offset
                va = "bottom" if height >= 0 else "top"
                ax_y_bill.text(bar.get_x() + bar.get_width() / 2, y_text, f"${height:,.2f}", ha="center", va=va, fontsize=10, fontweight="bold")

            ax_y_bill.set_title("Yearly Electricity Bill Comparison"); ax_y_bill.set_ylabel("Bill ($)")
            ax_y_bill.set_xticks(np.arange(len(labels_bill)))
            ax_y_bill.set_xticklabels(["PV+Bat\n+VPP", "PV+Bat\n(No VPP)", "Solar\nOnly", "Grid\nOnly"])
            ax_y_bill.grid(axis="y", alpha=0.3)
            plt.tight_layout(); st.pyplot(fig_y_bill); plt.close(fig_y_bill)
            
        st.divider()


def plot_monthly_analysis(df_vis_month, col_load, selected_month_name, selected_vis_year):
    st.markdown(f"### 📉 Monthly Analysis ({selected_month_name} {selected_vis_year})")
    if not isinstance(df_vis_month.index, pd.DatetimeIndex): df_vis_month = df_vis_month.set_index('timestamp')

    factor = 5.0/60.0
    
    # PRE-CALCULATION MONTHLY GRAPH
    df_heat_solar = df_vis_month[['irradiance']].resample('h').sum() * factor
    df_heat_solar['d'] = df_heat_solar.index.day; df_heat_solar['h'] = df_heat_solar.index.hour
    solar_matrix = df_heat_solar.pivot(index='h', columns='d', values='irradiance')
    curr_year = df_vis_month.index.year[0]; curr_month = df_vis_month.index.month[0]
    days_in_month = calendar.monthrange(curr_year, curr_month)[1]
    solar_matrix = solar_matrix.reindex(index=range(24), columns=range(1, days_in_month + 1)).fillna(0)
    
    df_m_calc = df_vis_month.copy()
    df_m_calc['solar_output_kwh'] = df_m_calc['solar_output_kw'] * factor
    df_m_calc['load_kwh'] = df_m_calc[col_load] * factor
    df_m_calc['grid_import_kwh'] = df_m_calc['grid_net_kw'].clip(lower=0) * factor
    hourly_sample = df_m_calc[['solar_output_kwh', 'load_kwh', 'grid_import_kwh']].resample('h').sum()
    hourly_sample['battery_soc_kwh'] = df_m_calc['battery_soc_kwh'].resample('h').mean()
    vpp_discharge = df_m_calc['vpp_status'].astype(bool) if 'vpp_status' in df_m_calc.columns else pd.Series(False, index=df_m_calc.index)
    vpp_charge = df_m_calc['vpp_charge'] if 'vpp_charge' in df_m_calc.columns else pd.Series(False, index=df_m_calc.index)

    # RENDER AREA MONTHLY
    # --- CHART 1: Irradiance Heatmap (Kompensasi Tinggi) ---
    fig_h_sol, ax_hs = plt.subplots(figsize=(14, 5))
    im_sol = ax_hs.imshow(solar_matrix.to_numpy(), cmap='YlOrRd', aspect='auto', interpolation='nearest', origin='lower')
    ax_hs.set_xlabel("Day"); ax_hs.set_ylabel("Hour"); ax_hs.set_title(f"Irradiance Heatmap - {selected_month_name}")
    ax_hs.set_xticks(np.arange(0, days_in_month)); ax_hs.set_xticklabels(np.arange(1, days_in_month + 1))
    cbar_sol = ax_hs.figure.colorbar(im_sol, ax=ax_hs, fraction=0.046, pad=0.04); cbar_sol.set_label("Irradiance ($Wh/m^2$)")
    plt.tight_layout(); st.pyplot(fig_h_sol); plt.close(fig_h_sol)

    st.divider()

    # --- CHART 2: Monthly Scrollable Battery Operation (Kompensasi Tinggi Ekstrem) ---
   
    fig_bat, ax1 = plt.subplots(figsize=(24, 8.5))
    ax1.plot(hourly_sample.index, hourly_sample["solar_output_kwh"], label="PV Generation", linewidth=1.2)
    ax1.plot(hourly_sample.index, hourly_sample["load_kwh"], label="Load", linewidth=1.2)
    ax1.plot(hourly_sample.index, hourly_sample["grid_import_kwh"], label="Grid Import", linewidth=1.2)
    ax1.set_ylabel("Energy (kWh)"); ax1.set_xlabel("Time")

    y_max = ax1.get_ylim()[1]
    if vpp_discharge.any(): ax1.fill_between(df_m_calc.index, 0, y_max, where=vpp_discharge, color="red", alpha=0.22, label="VPP Discharge Signal")
    if vpp_charge.any(): ax1.fill_between(df_m_calc.index, 0, y_max, where=vpp_charge, color="blue", alpha=0.18, label="VPP Charge Signal")

    ax2 = ax1.twinx()
    soc_max = df_m_calc["battery_soc_kwh"].max()
    ax2.plot(hourly_sample.index, hourly_sample["battery_soc_kwh"], label="Battery SOC", linewidth=1.8, linestyle="--", color="black")
    ax2.set_ylabel("Battery SOC (kWh)"); ax2.set_ylim(0, soc_max * 1.1 if soc_max > 0 else 1.0)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize='small', ncol=5)
    
    plt.title(f"Battery Operation with VPP Signals - {selected_month_name}")
    ax1.grid(alpha=0.3); ax1.margins(x=0); ax1.xaxis.set_major_locator(mdates.DayLocator(interval=2)); ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    plt.tight_layout(); st.pyplot(fig_bat); plt.close(fig_bat)