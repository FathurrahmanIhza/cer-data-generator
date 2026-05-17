import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import numpy as np
import calendar
import pandas as pd

def plot_annual_overview(df_vis_year, col_bat, selected_vis_year):
    factor = 5.0 / 60.0 
    
    df_calc = df_vis_year.copy()
    
    col_load = 'load_profile' if 'load_profile' in df_calc.columns else 'beban_rumah_kw'
    
    df_calc['solar_output_kwh'] = df_calc['solar_output_kw'] * factor
    df_calc['battery_discharge_kwh'] = df_calc[col_bat].clip(lower=0) * factor
    df_calc['grid_import_kwh'] = df_calc['grid_net_kw'].clip(lower=0) * factor
    df_calc['load_kwh'] = df_calc[col_load] * factor

    df_calc = df_calc.set_index('timestamp')

    st.markdown(f"### 📅 Annual Overview ({selected_vis_year})")

    elec_cols = [
        "solar_output_kwh",
        "battery_discharge_kwh",
        "grid_import_kwh"
    ]
    
    monthly = df_calc[elec_cols + ['load_kwh']].resample('ME').sum()
    months = [d.strftime("%b") for d in monthly.index]

    monthly_pct = (
        monthly[elec_cols]
        .div(monthly[elec_cols].sum(axis=1), axis=0)
        * 100
    )
    
    annual_load = monthly["load_kwh"].sum()
    annual_pv = monthly["solar_output_kwh"].sum()

    # --- VISUALISASI BARIS 1 (Energy Source & Percentage) ---
    c1, c2 = st.columns(2)
    
    colors = ["#FFD166", "#06D6A0", "#EF476F"]
    labels = ["PV Generation", "Battery Discharge", "Grid Import"]

    with c1:
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        
        ax1.bar(months, monthly['solar_output_kwh'], color=colors[0], label=labels[0], width=0.8)
        ax1.bar(months, monthly['battery_discharge_kwh'], bottom=monthly['solar_output_kwh'], color=colors[1], label=labels[1], width=0.8)
        ax1.bar(months, monthly['grid_import_kwh'], bottom=monthly['solar_output_kwh']+monthly['battery_discharge_kwh'], color=colors[2], label=labels[2], width=0.8)
        
        ax1.plot(months, monthly['load_kwh'], color="black", marker="o", linewidth=2.5, label="Load")
        
        ax1.set_title("Monthly Energy Contributions vs Load (kWh)")
        ax1.set_ylabel("Energy (kWh)")
        ax1.set_xlabel("Month")
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

        ax1.grid(axis='y', alpha=0.3)
        plt.tight_layout() 
        st.pyplot(fig1)
        plt.close(fig1)

    with c2:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        
        ax2.bar(months, monthly_pct['solar_output_kwh'], color=colors[0], label=labels[0], width=0.8)
        ax2.bar(months, monthly_pct['battery_discharge_kwh'], bottom=monthly_pct['solar_output_kwh'], color=colors[1], label=labels[1], width=0.8)
        ax2.bar(months, monthly_pct['grid_import_kwh'], bottom=monthly_pct['solar_output_kwh']+monthly_pct['battery_discharge_kwh'], color=colors[2], label=labels[2], width=0.8)
        
        ax2.set_title("Monthly Energy Contributions (%)")
        ax2.set_ylabel("Percentage (%)")
        ax2.set_xlabel("Month")
        ax2.set_ylim(0, 100)
        ax2.legend(title="Energy Source", fontsize='small', loc='lower right')
        
        ax2.grid(axis='y', alpha=0.3)
        plt.tight_layout() 
        st.pyplot(fig2)
        plt.close(fig2)

    # --- VISUALISASI BARIS 2 (HEATMAPS) ---
    c3, c4 = st.columns(2)

    df_calc["month"] = df_calc.index.month
    df_calc["hour"] = df_calc.index.hour

    with c3:
        if 'vpp_status' in df_calc.columns:
            df_calc["vpp_discharge_hours"] = df_calc["vpp_status"].astype(int) * factor
            
            heatmap_vpp = df_calc.pivot_table(
                index="month",
                columns="hour",
                values="vpp_discharge_hours",
                aggfunc="sum"
            ).reindex(index=range(1, 13), columns=range(24)).fillna(0)

            fig_vpp, ax_vpp = plt.subplots(figsize=(8, 5))
            
            im_vpp = ax_vpp.imshow(heatmap_vpp, aspect="auto", cmap="Oranges")
            
            ax_vpp.set_title("VPP Discharge Hours")
            ax_vpp.set_xlabel("Hour of Day")
            ax_vpp.set_ylabel("Month")
            
            ax_vpp.set_xticks(np.arange(24))
            ax_vpp.set_xticklabels(np.arange(24), fontsize=8)
            ax_vpp.set_yticks(np.arange(12))
            ax_vpp.set_yticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], fontsize=9)
            
            cbar_vpp = ax_vpp.figure.colorbar(im_vpp, ax=ax_vpp)
            cbar_vpp.set_label("Total Hours")
            
            plt.tight_layout()
            st.pyplot(fig_vpp)
            plt.close(fig_vpp)
        else:
            st.info("No VPP Status Data Available")

    with c4:
        heatmap_imp = df_calc.pivot_table(
            index="month",
            columns="hour",
            values="grid_import_kwh", 
            aggfunc="sum"
        ).reindex(index=range(1, 13), columns=range(24)).fillna(0)

        fig_imp, ax_imp = plt.subplots(figsize=(8, 5))
        
        im_imp = ax_imp.imshow(heatmap_imp, aspect="auto", cmap="Reds")
        
        ax_imp.set_title("Extra Import Energy (kWh)")
        ax_imp.set_xlabel("Hour of Day")
        
        ax_imp.set_xticks(np.arange(24))
        ax_imp.set_xticklabels(np.arange(24), fontsize=8)
        
        ax_imp.set_yticks(np.arange(12))
        ax_imp.set_yticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], fontsize=9)
        
        cbar_imp = ax_imp.figure.colorbar(im_imp, ax=ax_imp)
        cbar_imp.set_label("Total Energy (kWh)")
        
        plt.tight_layout()
        st.pyplot(fig_imp)
        plt.close(fig_imp)

    # --- VISUALISASI BARIS 3 (Price Profile) ---
    col_price = 'price_profile' if 'price_profile' in df_calc.columns else 'price_import'
    
    if col_price in df_calc.columns:
        price = df_calc[col_price]
        
        vpp_mask = df_calc['vpp_status'] > 0 if 'vpp_status' in df_calc.columns else pd.Series(False, index=df_calc.index)
        
        if vpp_mask.any():
            dispatch_price = df_calc.loc[vpp_mask, col_price].min()
        else:
            dispatch_price = np.inf
            
        fig_price, ax_p = plt.subplots(figsize=(12, 3.5)) 
        
        mask_pos = (price > 0) & (price < dispatch_price)
        mask_neg = price < 0
        mask_disp = price >= dispatch_price
        
        if mask_pos.any():
            ax_p.vlines(price.index[mask_pos], 0, price[mask_pos], color='#2FBF71', alpha=0.6, linewidth=1.5, label='Positive Price')
            
        if mask_neg.any():
            ax_p.vlines(price.index[mask_neg], 0, price[mask_neg], color='#E76F51', alpha=0.6, linewidth=1.5, label='Negative Price')
            
        if mask_disp.any():
            ax_p.vlines(price.index[mask_disp], 0, price[mask_disp], color='#7B2CBF', alpha=0.6, linewidth=1.5, label='Dispatch Price Event')
            
        if not np.isinf(dispatch_price):
            ax_p.axhline(dispatch_price, color='#7B2CBF', linestyle='--', linewidth=1.5, label='Dispatch Price Threshold')
            
        ax_p.xaxis.set_major_locator(mdates.MonthLocator())
        ax_p.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax_p.margins(x=0)
        
        ax_p.set_title('Electricity Spot Market Price (5 Minutes)')
        ax_p.set_ylabel('Price (AUD)')
        ax_p.grid(True, alpha=0.25)
        ax_p.legend(loc='upper right', fontsize='small')
        
        plt.tight_layout()
        st.pyplot(fig_price)
        plt.close(fig_price)
    else:
        st.warning("Price profile data not found.")


def plot_monthly_analysis(df_vis_month, col_load, selected_month_name, selected_vis_year):
    st.markdown(f"### 📉 Monthly Analysis ({selected_month_name} {selected_vis_year})")
    
    if not isinstance(df_vis_month.index, pd.DatetimeIndex):
        df_vis_month = df_vis_month.set_index('timestamp')

    # --- VISUALISASI BARIS 1 (Irradiance Heatmap) ---
    factor = 5.0/60.0
    df_heat_solar = df_vis_month[['irradiance']].resample('h').sum() * factor
    df_heat_solar['d'] = df_heat_solar.index.day
    df_heat_solar['h'] = df_heat_solar.index.hour
    
    solar_matrix = df_heat_solar.pivot(index='h', columns='d', values='irradiance')
    
    curr_year = df_vis_month.index.year[0]
    curr_month = df_vis_month.index.month[0]
    days_in_month = calendar.monthrange(curr_year, curr_month)[1]
    
    solar_matrix = solar_matrix.reindex(index=range(24), columns=range(1, days_in_month + 1)).fillna(0)
    data_matrix_solar = solar_matrix.to_numpy()
    
    fig_h_sol, ax_hs = plt.subplots(figsize=(14, 4))
    im_sol = ax_hs.imshow(data_matrix_solar, cmap='YlOrRd', aspect='auto', interpolation='nearest', origin='lower')
    
    ax_hs.set_xlabel("Day")
    ax_hs.set_ylabel("Hour")
    ax_hs.set_title(f"Irradiance Heatmap - {selected_month_name}")
    
    ax_hs.set_xticks(np.arange(0, days_in_month))
    ax_hs.set_xticklabels(np.arange(1, days_in_month + 1))
    
    cbar_sol = ax_hs.figure.colorbar(im_sol, ax=ax_hs, fraction=0.046, pad=0.04)
    cbar_sol.set_label("Irradiance ($Wh/m^2$)")
    
    plt.tight_layout()
    st.pyplot(fig_h_sol)
    plt.close(fig_h_sol)

    # --- VISUALISASI BARIS 2 (Monthly Scrollable Battery Operation Plot) ---
    st.divider()
    
    df_m_calc = df_vis_month.copy()
    df_m_calc['solar_output_kwh'] = df_m_calc['solar_output_kw'] * factor
    df_m_calc['load_kwh'] = df_m_calc[col_load] * factor
    df_m_calc['grid_import_kwh'] = df_m_calc['grid_net_kw'].clip(lower=0) * factor
    
    hourly_sample = df_m_calc[['solar_output_kwh', 'load_kwh', 'grid_import_kwh']].resample('h').sum()
    hourly_sample['battery_soc_kwh'] = df_m_calc['battery_soc_kwh'].resample('h').mean()
    
    vpp_discharge = df_m_calc['vpp_status'].astype(bool) if 'vpp_status' in df_m_calc.columns else pd.Series(False, index=df_m_calc.index)
    col_price = 'price_profile' if 'price_profile' in df_m_calc.columns else 'price_import'
    vpp_charge = (df_m_calc[col_price] < 0) if col_price in df_m_calc.columns else pd.Series(False, index=df_m_calc.index)

    fig_bat, ax1 = plt.subplots(figsize=(24, 5))

    ax1.plot(hourly_sample.index, hourly_sample["solar_output_kwh"], label="PV Generation", linewidth=1.2)
    ax1.plot(hourly_sample.index, hourly_sample["load_kwh"], label="Load", linewidth=1.2)
    ax1.plot(hourly_sample.index, hourly_sample["grid_import_kwh"], label="Grid Import", linewidth=1.2)

    ax1.set_ylabel("Energy (kWh)")
    ax1.set_xlabel("Time")

    y_max = ax1.get_ylim()[1]

    if vpp_discharge.any():
        ax1.fill_between(
            df_m_calc.index, 0, y_max,
            where=vpp_discharge,
            color="red", alpha=0.22, label="VPP Discharge Signal"
        )

    if vpp_charge.any():
        ax1.fill_between(
            df_m_calc.index, 0, y_max,
            where=vpp_charge,
            color="blue", alpha=0.18, label="VPP Charge Signal"
        )

    ax2 = ax1.twinx()
    soc_max = df_m_calc["battery_soc_kwh"].max()
    soc_ylim = soc_max * 1.1 if soc_max > 0 else 1.0

    ax2.plot(
        hourly_sample.index, hourly_sample["battery_soc_kwh"],
        label="Battery SOC", linewidth=1.8, linestyle="--", color="black"
    )

    ax2.set_ylabel("Battery SOC (kWh)")
    ax2.set_ylim(0, soc_ylim)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize='small')

    plt.title(f"Battery Operation with VPP Signals - {selected_month_name}")

    ax1.grid(alpha=0.3)
    ax1.margins(x=0)
    
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))

    plt.tight_layout()
    st.pyplot(fig_bat)
    plt.close(fig_bat)