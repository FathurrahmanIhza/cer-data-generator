import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import numpy as np
import calendar
import pandas as pd

def plot_annual_overview(df_vis_year, col_bat, selected_vis_year):
    factor = 5.0 / 60.0 
    
    # PRE-CALCULATION YEARLY GRAPH
    df_calc = df_vis_year.copy()
    col_load = 'load_profile' if 'load_profile' in df_calc.columns else 'beban_rumah_kw'
    
    # 1. Konversi Daya Dasar (kW) ke Energi (kWh) & Tambah Kolom Helper Beruntun
    df_calc['solar_output_kwh'] = df_calc['solar_output_kw'] * factor
    df_calc['battery_discharge_kwh'] = df_calc[col_bat].clip(lower=0) * factor
    df_calc['grid_import_kwh'] = df_calc['grid_import_kw'] * factor
    df_calc['grid_export_kwh'] = df_calc['grid_export_kw'] * factor
    df_calc['load_kwh'] = df_calc[col_load] * factor
    df_calc['vpp_bat_dis_kwh_tmp'] = df_calc['vpp_battery_discharge_kw'] * factor
    df_calc['vpp_discharge_hours'] = df_calc["vpp_status"].astype(int) * factor
    df_calc['extra_import_kwh'] = df_calc['vpp_grid_import_after_discharge_kw'] * factor
    
    df_calc = df_calc.set_index('timestamp')
    df_calc["month"] = df_calc.index.month
    df_calc["hour"] = df_calc.index.hour

    # 2. Agregasi Bulanan Utama (Untuk Row 1, Row 5, Row 6, Row 7)
    monthly = df_calc.resample('ME').sum()
    months_labels = [d.strftime("%b") for d in monthly.index]
    
    # Persentase Kontribusi Energi (Row 1)
    elec_cols = ["solar_output_kwh", "battery_discharge_kwh", "grid_import_kwh"]
    monthly_pct = (monthly[elec_cols].div(monthly[elec_cols].sum(axis=1), axis=0) * 100)
    annual_load = monthly["load_kwh"].sum()
    annual_pv = monthly["solar_output_kwh"].sum()

    # 3. Matriks Heatmap (Row 2)
    heatmap_vpp = df_calc.pivot_table(index="month", columns="hour", values="vpp_discharge_hours", aggfunc="sum").reindex(index=range(1, 13), columns=range(24)).fillna(0)
    heatmap_imp = df_calc.pivot_table(index="month", columns="hour", values="extra_import_kwh", aggfunc="sum").reindex(index=range(1, 13), columns=range(24)).fillna(0)

    # 4. Filter Masking Harga Pasar Spot (Row 3)
    col_price = 'price_profile' if 'price_profile' in df_calc.columns else 'price_import'
    price_series = df_calc[col_price] if col_price in df_calc.columns else pd.Series(dtype=np.float64)
    vpp_mask_price = df_calc['vpp_status'] > 0
    dispatch_price_threshold = df_calc.loc[vpp_mask_price, col_price].min() if vpp_mask_price.any() else np.inf
    mask_price_pos = (price_series > 0) & (price_series < dispatch_price_threshold)
    mask_price_neg = price_series < 0
    mask_price_disp = price_series >= dispatch_price_threshold

    # 5. Akumulasi Garis Kumulatif VPP (Row 4)
    cumulative_vpp = df_calc['vpp_grid_export_kw'].resample('D').sum().cumsum() * factor
    threshold_contract = 1000

    # 6. Keuangan & Profitabilitas VPP (Row 5)
    monthly["vpp_payment"] = 20.0 
    monthly["net_cost"] = monthly["vpp_operational_net_value_AUD"] * -1  # Negatif diubah ke positif cost agar sesuai grafik

    # 7. Skenario Tahunan Akhir (Row 7)
    actual_with_vpp = monthly['bill_actual'].sum()
    actual_without_vpp_approx = actual_with_vpp + monthly['vpp_operational_net_value_AUD'].sum() + (20.0 * len(months_labels))
    yearly_bill_series = pd.Series({
        "Actual\n(PV + Battery + VPP)": actual_with_vpp,
        "Approx. Actual\n(PV + Battery, no VPP)": actual_without_vpp_approx,
        "Solar Only": monthly['bill_solar_only'].sum(),
        "Grid Only": monthly['bill_grid_only'].sum()
    })

    # 8. Analisis Batasan Event VPP (Row 8)
    vpp_mask_event = df_calc['vpp_status'] > 0
    df_calc['event_id'] = (vpp_mask_event != vpp_mask_event.shift()).cumsum()
    vpp_events_data = df_calc[vpp_mask_event]
    
    event_df = pd.DataFrame()
    if not vpp_events_data.empty:
        grouped_events = vpp_events_data.groupby('event_id')
        duration_h = grouped_events.size() * factor
        bat_power_kw = df_calc[col_bat].abs().max() or 15.0
        requested_vpp_kwh = duration_h * bat_power_kw
        actual_vpp_discharge_kwh = grouped_events['vpp_bat_dis_kwh_tmp'].sum()
        dispatch_limited = actual_vpp_discharge_kwh < (requested_vpp_kwh - 0.1)
        event_df = pd.DataFrame({'requested_vpp_kwh': requested_vpp_kwh, 'actual_vpp_discharge_kwh': actual_vpp_discharge_kwh, 'dispatch_limited': dispatch_limited})

    # 9. Efisiensi & Mandiri Energi Rumah (Row 9)
    monthly_metrics = df_calc[['solar_output_kwh', 'grid_export_kwh', 'grid_import_kwh', 'load_kwh', 'vpp_bat_dis_kwh_tmp']].resample('ME').sum()
    solar_sum_safe = monthly_metrics['solar_output_kwh'].replace(0, np.nan)
    load_sum_safe = monthly_metrics['load_kwh'].replace(0, np.nan)
    monthly_metrics["self_consumption_pct"] = ((monthly_metrics["solar_output_kwh"] - monthly_metrics["grid_export_kwh"]) / solar_sum_safe) * 100
    monthly_metrics["self_sufficiency_pct"] = (1 - (monthly_metrics["grid_import_kwh"] / load_sum_safe)) * 100
    monthly_metrics.fillna(0, inplace=True)


    # RENDER AREA YEARLY
    st.markdown(f"### 📅 Annual Overview ({selected_vis_year})")

    # --- ROW 1: Energy Source & Percentage ---
    c1, c2 = st.columns(2)
    colors_src = ["#FFD166", "#06D6A0", "#EF476F"]
    labels_src = ["PV Generation", "Battery Discharge", "Grid Import"]

    with c1:
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.bar(months_labels, monthly['solar_output_kwh'], color=colors_src[0], label=labels_src[0], width=0.8)
        ax1.bar(months_labels, monthly['battery_discharge_kwh'], bottom=monthly['solar_output_kwh'], color=colors_src[1], label=labels_src[1], width=0.8)
        ax1.bar(months_labels, monthly['grid_import_kwh'], bottom=monthly['solar_output_kwh']+monthly['battery_discharge_kwh'], color=colors_src[2], label=labels_src[2], width=0.8)
        ax1.plot(months_labels, monthly['load_kwh'], color="black", marker="o", linewidth=2.5, label="Load")
        ax1.set_title("Monthly Energy Contributions vs Load (kWh)")
        ax1.set_ylabel("Energy (kWh)"); ax1.set_xlabel("Month")
        ax1.legend(title="Energy Source", fontsize='small', loc='upper center')
        ax1.text(0.98, 0.95, f"Annual Load: {annual_load:,.0f} kWh\nAnnual PV: {annual_pv:,.0f} kWh", transform=ax1.transAxes, ha="right", va="top", fontsize=9, bbox=dict(facecolor="white", alpha=0.8))
        ax1.grid(axis='y', alpha=0.3)
        plt.tight_layout(); st.pyplot(fig1); plt.close(fig1)

    with c2:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.bar(months_labels, monthly_pct['solar_output_kwh'], color=colors_src[0], label=labels_src[0], width=0.8)
        ax2.bar(months_labels, monthly_pct['battery_discharge_kwh'], bottom=monthly_pct['solar_output_kwh'], color=colors_src[1], label=labels_src[1], width=0.8)
        ax2.bar(months_labels, monthly_pct['grid_import_kwh'], bottom=monthly_pct['solar_output_kwh']+monthly_pct['battery_discharge_kwh'], color=colors_src[2], label=labels_src[2], width=0.8)
        ax2.set_title("Monthly Energy Contributions (%)")
        ax2.set_ylabel("Percentage (%)"); ax2.set_xlabel("Month")
        ax2.set_ylim(0, 100)
        ax2.legend(title="Energy Source", fontsize='small', loc='lower right')
        ax2.grid(axis='y', alpha=0.3)
        plt.tight_layout(); st.pyplot(fig2); plt.close(fig2)

    # --- ROW 2: Heatmaps (VPP Discharge & Extra Import) ---
    c3, c4 = st.columns(2)
    with c3:
        if 'vpp_status' in df_calc.columns:
            fig_vpp, ax_vpp = plt.subplots(figsize=(8, 5))
            im_vpp = ax_vpp.imshow(heatmap_vpp, aspect="auto", cmap="Oranges")
            ax_vpp.set_title("VPP Discharge Hours")
            ax_vpp.set_xlabel("Hour of Day"); ax_vpp.set_ylabel("Month")
            ax_vpp.set_xticks(np.arange(24)); ax_vpp.set_xticklabels(np.arange(24), fontsize=8)
            ax_vpp.set_yticks(np.arange(12)); ax_vpp.set_yticklabels([calendar.month_abbr[i] for i in range(1, 13)], fontsize=9)
            cbar_vpp = ax_vpp.figure.colorbar(im_vpp, ax=ax_vpp); cbar_vpp.set_label("Total Hours")
            plt.tight_layout(); st.pyplot(fig_vpp); plt.close(fig_vpp)
        else:
            st.info("No VPP Status Data Available")

    with c4:
        if 'vpp_grid_import_after_discharge_kw' in df_calc.columns:
            fig_imp, ax_imp = plt.subplots(figsize=(8, 5))
            im_imp = ax_imp.imshow(heatmap_imp, aspect="auto", cmap="Reds")
            ax_imp.set_title("Extra Import Energy (kWh)")
            ax_imp.set_xlabel("Hour of Day")
            ax_imp.set_xticks(np.arange(24)); ax_imp.set_xticklabels(np.arange(24), fontsize=8)
            ax_imp.set_yticks(np.arange(12)); ax_imp.set_yticklabels([calendar.month_abbr[i] for i in range(1, 13)], fontsize=9)
            cbar_imp = ax_imp.figure.colorbar(im_imp, ax=ax_imp); cbar_imp.set_label("Total Energy (kWh)")
            plt.tight_layout(); st.pyplot(fig_imp); plt.close(fig_imp)

    # --- ROW 3: Price Profile ---
    if col_price in df_calc.columns:
        fig_price, ax_p = plt.subplots(figsize=(12, 3.5)) 
        if mask_price_pos.any(): ax_p.vlines(price_series.index[mask_price_pos], 0, price_series[mask_price_pos], color='#2FBF71', alpha=0.6, linewidth=1.5, label='Positive Price')
        if mask_price_neg.any(): ax_p.vlines(price_series.index[mask_price_neg], 0, price_series[mask_price_neg], color='#E76F51', alpha=0.6, linewidth=1.5, label='Negative Price')
        if mask_price_disp.any(): ax_p.vlines(price_series.index[mask_price_disp], 0, price_series[mask_price_disp], color='#7B2CBF', alpha=0.6, linewidth=1.5, label='Dispatch Price Event')
        if not np.isinf(dispatch_price_threshold): ax_p.axhline(dispatch_price_threshold, color='#7B2CBF', linestyle='--', linewidth=1.5, label='Dispatch Price Threshold')
        ax_p.xaxis.set_major_locator(mdates.MonthLocator()); ax_p.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax_p.margins(x=0); ax_p.set_title('Electricity Spot Market Price (5 Minutes)'); ax_p.set_ylabel('Price (AUD)')
        ax_p.grid(True, alpha=0.25); ax_p.legend(loc='upper right', fontsize='small')
        plt.tight_layout(); st.pyplot(fig_price); plt.close(fig_price)

    # --- ROW 4: Cumulative VPP Discharge ---
    if 'vpp_grid_export_kw' in df_calc.columns:
        fig_cum, ax_cum = plt.subplots(figsize=(12, 4))
        ax_cum.plot(cumulative_vpp.index, cumulative_vpp, linestyle='-', linewidth=2.5, color='blue', label='Cumulative VPP Discharge')
        ax_cum.fill_between(cumulative_vpp.index, cumulative_vpp, threshold_contract, where=(cumulative_vpp <= threshold_contract), color='green', alpha=0.3, label='Below limit')
        ax_cum.fill_between(cumulative_vpp.index, cumulative_vpp, threshold_contract, where=(cumulative_vpp > threshold_contract), color='red', alpha=0.3, label='Above limit')
        ax_cum.axhline(threshold_contract, color='black', linestyle='--', label=f'Contract limit {threshold_contract} kWh')
        ax_cum.set_title('Cumulative VPP Discharge'); ax_cum.set_ylabel('Energy discharged (kWh)')
        ax_cum.xaxis.set_major_locator(mdates.MonthLocator()); ax_cum.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax_cum.margins(x=0); ax_cum.grid(True, alpha=0.3); ax_cum.legend(loc='upper right', fontsize='small')
        plt.tight_layout(); st.pyplot(fig_cum); plt.close(fig_cum)

    # --- ROW 5: Monthly VPP Economics ---
    if 'vpp_export_value_AUD' in monthly.columns:
        x_pos = np.arange(len(months_labels))
        colors_econ = np.where(monthly["net_cost"] <= 0, "#55A868", np.where(monthly["net_cost"] <= monthly["vpp_payment"], "#DD8452", "#C44E52"))
        fig_econ, ax_econ = plt.subplots(figsize=(14, 6))
        ax_econ.bar(x_pos, monthly["net_cost"], color=colors_econ, label="Net Operational Cost", width=0.7)
        ax_econ.plot(x_pos, monthly["vpp_payment"], color="black", linestyle="--", linewidth=2, label="VPP Payment")
        ax_econ.plot(x_pos, monthly["vpp_extra_import_cost_AUD"], marker="o", linestyle="-", linewidth=2, label="Extra Import Cost ($)")
        ax_econ.plot(x_pos, monthly["vpp_export_value_AUD"], color="orange", marker="s", linestyle="-", linewidth=2, label="VPP Export Value ($)")
        ax_econ.set_xticks(x_pos); ax_econ.set_xticklabels(months_labels); ax_econ.set_ylabel("Value ($)")
        ax_econ.set_title("Monthly VPP Export Value vs Extra Import Cost")
        ax_econ.legend(fontsize='small', loc='upper right'); ax_econ.grid(axis='y', linestyle='--', alpha=0.5); ax_econ.margins(x=0.02)
        plt.tight_layout(); st.pyplot(fig_econ); plt.close(fig_econ)

    # --- ROW 6: Monthly Electricity Bill Comparison ---
    if 'bill_actual' in monthly.columns:

        bill_cols = ["bill_actual", "bill_solar_only", "bill_grid_only"]
        labels_bill = ["Actual (PV + Battery)", "Solar Only", "No Battery & Solar"]
        x_b = np.arange(len(months_labels)); width = 0.25
        fig_bill, ax_bill = plt.subplots(figsize=(14, 6))
        colors_bill = ['#2ca02c', '#ff7f0e', '#1f77b4']
        for i, col in enumerate(bill_cols):
            ax_bill.bar(x_b + i * width, monthly[col], width=width, label=labels_bill[i], color=colors_bill[i])
        ax_bill.set_xticks(x_b + width); ax_bill.set_xticklabels(months_labels)
        ax_bill.set_xlabel("Month"); ax_bill.set_ylabel("Bill ($)")
        ax_bill.set_title("Monthly Electricity Bill Comparison"); ax_bill.legend(title="Scenario")
        ax_bill.grid(axis="y", alpha=0.3); ax_bill.margins(x=0.02)
        plt.tight_layout(); st.pyplot(fig_bill); plt.close(fig_bill)

    # --- ROW 7: Yearly Electricity Bill Comparison ---
    if 'bill_actual' in monthly.columns:
        fig_y_bill, ax_y_bill = plt.subplots(figsize=(10, 6))
        bars = ax_y_bill.bar(yearly_bill_series.index, yearly_bill_series.values, color=["#1f77b4", "#9467bd", "#ff7f0e", "#2ca02c"], width=0.6)
        padding = max(abs(yearly_bill_series.max()), abs(yearly_bill_series.min())) * 0.15
        ax_y_bill.set_ylim(yearly_bill_series.min() - padding, yearly_bill_series.max() + padding)
        ax_y_bill.axhline(0, color="black", linewidth=1)
        offset = max(abs(yearly_bill_series.max()), abs(yearly_bill_series.min())) * 0.03
        for bar in bars:
            height = bar.get_height()
            y_text = height + offset if height >= 0 else height - offset
            va = "bottom" if height >= 0 else "top"
            ax_y_bill.text(bar.get_x() + bar.get_width() / 2, y_text, f"${height:,.2f}", ha="center", va=va, fontsize=10, fontweight="bold")
        ax_y_bill.set_title("Yearly Electricity Bill Comparison")
        ax_y_bill.set_ylabel("Bill ($)"); ax_y_bill.set_xlabel("Scenario"); ax_y_bill.grid(axis="y", alpha=0.3)
        plt.tight_layout(); st.pyplot(fig_y_bill); plt.close(fig_y_bill)

    # --- ROW 8: VPP Event-Level Dispatch Limitation Analysis ---
    if 'vpp_status' in df_calc.columns and not event_df.empty:
        
        fig_scatter, ax_sc = plt.subplots(figsize=(9, 6))
        normal = event_df[~event_df["dispatch_limited"]]; limited = event_df[event_df["dispatch_limited"]]
        ax_sc.scatter(normal["requested_vpp_kwh"], normal["actual_vpp_discharge_kwh"], color="steelblue", edgecolor="black", alpha=0.75, s=70, label="Full Dispatch Achieved")
        ax_sc.scatter(limited["requested_vpp_kwh"], limited["actual_vpp_discharge_kwh"], color="red", edgecolor="black", alpha=0.9, s=90, label="Dispatch Limited")
        max_val = max(event_df["requested_vpp_kwh"].max(), event_df["actual_vpp_discharge_kwh"].max())
        ax_sc.plot([0, max_val], [0, max_val], linestyle="--", color="black", alpha=0.6, label="Requested = Actual")
        ax_sc.set_xlabel("Requested VPP Energy (kWh)"); ax_sc.set_ylabel("Actual VPP Discharge Energy (kWh)")
        ax_sc.set_title("Requested vs Actual VPP Dispatch Energy"); ax_sc.legend(); ax_sc.grid(alpha=0.3)
        plt.tight_layout(); st.pyplot(fig_scatter); plt.close(fig_scatter)

    # --- ROW 9: Monthly Self-Consumption & Self-Sufficiency ---
    fig_ss, ax1_ss = plt.subplots(figsize=(14, 6))
    x_m = np.arange(len(months_labels))
    ax1_ss.plot(x_m, monthly_metrics["self_consumption_pct"], marker="o", linewidth=2, label="PV Self-Consumption")
    ax1_ss.plot(x_m, monthly_metrics["self_sufficiency_pct"], marker="s", linewidth=2, label="Home Self-Sufficiency")
    ax1_ss.set_ylabel("Percentage (%)"); ax1_ss.set_ylim(0, 100)
    ax1_ss.set_xticks(x_m); ax1_ss.set_xticklabels(months_labels); ax1_ss.set_xlabel("Month")
    ax1_ss.axvspan(4.5, 6.5, color="grey", alpha=0.18, label="High VPP Activity Period")
    
    ax2_ss = ax1_ss.twinx()
    
    ax2_ss.bar(x_m, monthly_metrics["vpp_bat_dis_kwh_tmp"], alpha=0.25, color="red", width=0.65, label="VPP Discharge")
    ax2_ss.set_ylabel("VPP Discharge (kWh)")
    
    lines1_ss, labels1_ss = ax1_ss.get_legend_handles_labels()
    lines2_ss, labels2_ss = ax2_ss.get_legend_handles_labels()
    ax1_ss.legend(lines1_ss + lines2_ss, labels1_ss + labels2_ss, loc="lower left", fontsize='small')
    ax1_ss.set_title("Monthly Self-Consumption, Self-Sufficiency, and VPP Dispatch Activity")
    ax1_ss.grid(alpha=0.3); ax1_ss.margins(x=0.02)
    plt.tight_layout(); st.pyplot(fig_ss); plt.close(fig_ss)


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
    # --- CHART 1: Irradiance Heatmap ---
    fig_h_sol, ax_hs = plt.subplots(figsize=(14, 4))
    im_sol = ax_hs.imshow(solar_matrix.to_numpy(), cmap='YlOrRd', aspect='auto', interpolation='nearest', origin='lower')
    ax_hs.set_xlabel("Day"); ax_hs.set_ylabel("Hour"); ax_hs.set_title(f"Irradiance Heatmap - {selected_month_name}")
    ax_hs.set_xticks(np.arange(0, days_in_month)); ax_hs.set_xticklabels(np.arange(1, days_in_month + 1))
    cbar_sol = ax_hs.figure.colorbar(im_sol, ax=ax_hs, fraction=0.046, pad=0.04); cbar_sol.set_label("Irradiance ($Wh/m^2$)")
    plt.tight_layout(); st.pyplot(fig_h_sol); plt.close(fig_h_sol)

    # --- CHART 2: Monthly Scrollable Battery Operation ---
    fig_bat, ax1 = plt.subplots(figsize=(24, 5))
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
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=5, fontsize='small')
    plt.title(f"Battery Operation with VPP Signals - {selected_month_name}")
    ax1.grid(alpha=0.3); ax1.margins(x=0); ax1.xaxis.set_major_locator(mdates.DayLocator(interval=2)); ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    plt.tight_layout(); st.pyplot(fig_bat); plt.close(fig_bat)