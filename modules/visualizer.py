import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import numpy as np
import calendar
import pandas as pd

def plot_annual_overview(df_vis_year, col_bat, selected_vis_year):
    """
    Menampilkan visualisasi tahunan.
    Optimasi diterapkan pada Chart Energi (Barchart) & Grid Net.
    Price Profile tetap mempertahankan resolusi 5 menit (High Fidelity).
    """
    
    factor = 5.0 / 60.0 
    
    df_calc = df_vis_year.copy()
    df_calc['pv_kwh'] = df_calc['solar_output_kw'] * factor
    df_calc['bat_dis_kwh'] = df_calc[col_bat].clip(lower=0) * factor
    
    # Grid Import
    df_calc['grid_imp_kwh'] = df_calc['grid_net_kw'].clip(lower=0) * factor

    df_calc = df_calc.set_index('timestamp')

    st.markdown(f"### 📅 Annual Overview ({selected_vis_year})")

    # AGGREGASI DATA BULANAN  ---
    df_monthly_agg = df_calc[['pv_kwh', 'bat_dis_kwh', 'grid_imp_kwh']].resample('ME').sum()
    df_monthly_agg['month_name'] = df_monthly_agg.index.strftime('%b') 
    
    df_monthly_agg['total_supply'] = df_monthly_agg['pv_kwh'] + df_monthly_agg['bat_dis_kwh'] + df_monthly_agg['grid_imp_kwh']
    df_monthly_agg['total_supply'] = df_monthly_agg['total_supply'].replace(0, 1)
    
    df_monthly_agg['pct_pv'] = (df_monthly_agg['pv_kwh'] / df_monthly_agg['total_supply']) * 100
    df_monthly_agg['pct_bat'] = (df_monthly_agg['bat_dis_kwh'] / df_monthly_agg['total_supply']) * 100
    df_monthly_agg['pct_grid'] = (df_monthly_agg['grid_imp_kwh'] / df_monthly_agg['total_supply']) * 100
    
    colors = ['#1f77b4', '#8c564b', '#add8e6'] 
    months = df_monthly_agg['month_name']

    # --- VISUALISASI BARIS 1 (Energy Source & Percentage) ---
    c1, c2 = st.columns(2)
    
    with c1:
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.bar(months, df_monthly_agg['pv_kwh'], color=colors[0], label='PV')
        ax1.bar(months, df_monthly_agg['bat_dis_kwh'], bottom=df_monthly_agg['pv_kwh'], color=colors[1], label='Bat Discharge')
        ax1.bar(months, df_monthly_agg['grid_imp_kwh'], bottom=df_monthly_agg['pv_kwh']+df_monthly_agg['bat_dis_kwh'], color=colors[2], label='Grid Import')
        
        ax1.set_title("Monthly Energy Source")
        ax1.set_ylabel("Energy (kWh)")
        ax1.legend(fontsize='small', loc='lower right')
        ax1.grid(axis='y', alpha=0.3)
        plt.tight_layout() 
        st.pyplot(fig1)

    with c2:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.bar(months, df_monthly_agg['pct_pv'], color=colors[0], label='PV')
        ax2.bar(months, df_monthly_agg['pct_bat'], bottom=df_monthly_agg['pct_pv'], color=colors[1], label='Bat Discharge')
        ax2.bar(months, df_monthly_agg['pct_grid'], bottom=df_monthly_agg['pct_pv']+df_monthly_agg['pct_bat'], color=colors[2], label='Grid Import')
        
        ax2.set_title("Energy Contribution")
        ax2.set_ylabel("Percentage (%)")
        ax2.set_ylim(0, 100)
        ax2.legend(fontsize='small', loc='lower right')
        ax2.grid(axis='y', alpha=0.3)
        plt.tight_layout() 
        st.pyplot(fig2)

    # --- VISUALISASI BARIS 2 (HEATMAPS) ---
    c3, c4 = st.columns(2)

    def create_heatmap_matrix(series_data):
        daily = series_data.resample('D').sum().reset_index()
        daily['m'] = daily['timestamp'].dt.month
        daily['d'] = daily['timestamp'].dt.day
        pivot = daily.pivot(index='m', columns='d', values=series_data.name)
        pivot = pivot.reindex(index=range(1, 13), columns=range(1, 32)).fillna(0)
        return pivot.to_numpy()

    with c3:
        if 'vpp_status' in df_calc.columns:
            vpp_minutes_series = df_calc['vpp_status'].astype(int) * 5
            vpp_minutes_series.name = 'vpp_minutes'
            
            data_matrix = create_heatmap_matrix(vpp_minutes_series)
            
            fig_heat, ax_h = plt.subplots(figsize=(6, 4))
            cmap_custom = mcolors.LinearSegmentedColormap.from_list("WhiteOrange", ["white", "orange", "#8B4500"])
            
            max_val = data_matrix.max()
            vmax_val = max(max_val, 60) 
            
            im = ax_h.imshow(data_matrix, cmap=cmap_custom, aspect='auto', interpolation='nearest', vmin=0, vmax=vmax_val)
            
            ax_h.set_xticks(np.arange(0, 31, 2))
            ax_h.set_xticklabels(np.arange(1, 32, 2), fontsize=7)
            ax_h.set_yticks(np.arange(12))
            ax_h.set_yticklabels([calendar.month_abbr[i] for i in range(1, 13)], fontsize=8)
            ax_h.set_title("VPP Discharge")
            
            cbar = ax_h.figure.colorbar(im, ax=ax_h, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)
            cbar.set_label("Minutes Active")
            
            plt.tight_layout()
            st.pyplot(fig_heat)
        else:
            st.info("No VPP Status Data Available")

    with c4:
        col_price = 'price_import' if 'price_import' in df_calc.columns else 'price_profile'
        
        if col_price in df_calc.columns:
            neg_minutes_series = (df_calc[col_price] < 0).astype(int) * 5
            neg_minutes_series.name = 'neg_minutes'
            
            data_matrix_neg = create_heatmap_matrix(neg_minutes_series)
            
            fig_neg, ax_n = plt.subplots(figsize=(6, 4))
            cmap_neg = mcolors.LinearSegmentedColormap.from_list("WhiteGreen", ["white", "#2ecc71", "#006400"])
            
            max_val_n = data_matrix_neg.max()
            vmax_val_n = max(max_val_n, 60)
            
            im_n = ax_n.imshow(data_matrix_neg, cmap=cmap_neg, aspect='auto', interpolation='nearest', vmin=0, vmax=vmax_val_n)
            
            ax_n.set_xticks(np.arange(0, 31, 2))
            ax_n.set_xticklabels(np.arange(1, 32, 2), fontsize=7)
            ax_n.set_yticks(np.arange(12))
            ax_n.set_yticklabels([calendar.month_abbr[i] for i in range(1, 13)], fontsize=8)
            ax_n.set_title("VPP Charge")
            
            cbar_n = ax_n.figure.colorbar(im_n, ax=ax_n, fraction=0.046, pad=0.04)
            cbar_n.ax.tick_params(labelsize=8)
            cbar_n.set_label("Minutes Active")
            
            plt.tight_layout()
            st.pyplot(fig_neg)
        else:
            st.info("Price Data Not Available")


    # --- VISUALISASI BARIS 3 (VPP vs Normal Stackplot) ---
    arr_dis_kwh = df_calc['bat_dis_kwh'].to_numpy()
    
    if 'vpp_status' in df_calc.columns:
        arr_vpp_stat = df_calc['vpp_status'].to_numpy().astype(bool)
        series_vpp = np.where(arr_vpp_stat, arr_dis_kwh, 0)
        series_norm = np.where(~arr_vpp_stat, arr_dis_kwh, 0)
    else:
        series_vpp = np.zeros_like(arr_dis_kwh)
        series_norm = arr_dis_kwh

    df_stack = pd.DataFrame({
        'vpp': series_vpp,
        'norm': series_norm
    }, index=df_calc.index)
    
    daily_bat = df_stack.resample('D').sum()
    
    fig_break, ax_b = plt.subplots(figsize=(12, 3.5))
    colors_bat = ['#d35400', '#55a868'] 
    labels_bat = ['VPP Discharge', 'Normal Discharge']
    
    ax_b.stackplot(daily_bat.index, 
                   daily_bat['vpp'], 
                   daily_bat['norm'],
                   colors=colors_bat, labels=labels_bat, alpha=0.8)
    
    ax_b.set_title("VPP Discharge VS Normal Discharge")
    ax_b.set_ylabel("Energy (kWh)")
    
    ax_b.xaxis.set_major_locator(mdates.MonthLocator())
    ax_b.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax_b.set_xlim(daily_bat.index[0], daily_bat.index[-1])
    
    ax_b.legend(loc='upper right', fontsize='small')
    ax_b.grid(True, alpha=0.3)
    
    plt.tight_layout() 
    st.pyplot(fig_break)


    # --- VISUALISASI BARIS 4 (Price Profile) ---
    if 'price_profile' in df_calc.columns:
        
        df_price_raw = df_calc['price_profile']
        
        fig_price, ax_p = plt.subplots(figsize=(12, 3)) 
        
        ax_p.plot(df_price_raw.index, df_price_raw, color='black', linewidth=0.5, alpha=0.3)
        
        ax_p.fill_between(df_price_raw.index, df_price_raw, 0, 
                          where=(df_price_raw >= 0), 
                          interpolate=True, color='#2ecc71', alpha=0.6, label='Positive Price')
        
        ax_p.fill_between(df_price_raw.index, df_price_raw, 0, 
                          where=(df_price_raw < 0), 
                          interpolate=True, color='#e74c3c', alpha=0.6, label='Negative Price')
        
        ax_p.set_ylabel("Price (AUD)")
        ax_p.set_title("Electricity Spot Market Price (5 Minutes)") 

        ax_p.xaxis.set_major_locator(mdates.MonthLocator())
        ax_p.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax_p.set_xlim(df_price_raw.index[0], df_price_raw.index[-1])
        
        ax_p.grid(True, alpha=0.3)
        ax_p.legend(loc='upper right', fontsize='small')
        
        st.pyplot(fig_price)
    else:
        st.warning("Price profile data not found.")
    
    # --- VISUALISASI BARIS 5 (Grid Net) ---
    if 'grid_net_kw' in df_calc.columns:
        s_imp_daily = df_calc['grid_imp_kwh'].resample('D').sum()
        s_exp_daily = df_calc['grid_net_kw'].clip(upper=0).resample('D').sum() * factor
        
        fig_grid, ax_g = plt.subplots(figsize=(12, 3)) 
        
        ax_g.plot(s_imp_daily.index, s_imp_daily, color='#e74c3c', linewidth=1, label='Total Import')
        ax_g.fill_between(s_imp_daily.index, s_imp_daily, 0, color='#e74c3c', alpha=0.5)
        
        ax_g.plot(s_exp_daily.index, s_exp_daily, color='#2ecc71', linewidth=1, label='Total Export')
        ax_g.fill_between(s_exp_daily.index, s_exp_daily, 0, color='#2ecc71', alpha=0.5)
        
        ax_g.axhline(0, color='black', linewidth=0.8, linestyle='--')
        
        ax_g.set_ylabel("Energy (kWh)")
        ax_g.set_title("Grid Exchange: Import vs Export")

        ax_g.xaxis.set_major_locator(mdates.MonthLocator())
        ax_g.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax_g.set_xlim(s_imp_daily.index[0], s_imp_daily.index[-1])
        
        ax_g.grid(True, alpha=0.3)
        ax_g.legend(loc='upper right', fontsize='small')
        
        st.pyplot(fig_grid)
    else:
        st.warning("Grid interaction data not found.")


def plot_monthly_analysis(df_vis_month, col_load, selected_month_name, selected_vis_year):
   
    st.markdown(f"### 📉 Monthly Analysis ({selected_month_name} {selected_vis_year})")
    
    if not isinstance(df_vis_month.index, pd.DatetimeIndex):
        df_vis_month = df_vis_month.set_index('timestamp')

    c1, c2 = st.columns(2)
    
    with c1:
        df_profile = df_vis_month.groupby(df_vis_month.index.hour)[['solar_output_kw', col_load]].mean()
        
        max_val = max(df_profile['solar_output_kw'].max(), df_profile[col_load].max())
        y_limit = max_val * 1.1 if max_val > 0 else 1.0

        fig_prof, ax_p1 = plt.subplots(figsize=(6, 4))
        
        color_ghi = 'orange'
        ax_p1.set_xlabel('Hour (0-23)')
        ax_p1.set_ylabel('Solar Output (kW)', color=color_ghi)
        line1 = ax_p1.plot(df_profile.index, df_profile['solar_output_kw'], color=color_ghi, linewidth=2, label='Solar Output')
        ax_p1.tick_params(axis='y', labelcolor=color_ghi)
        ax_p1.grid(True, alpha=0.3)
        ax_p1.set_ylim(0, y_limit) 
        
        ax_p2 = ax_p1.twinx()  
        color_load = '#d62728' 
        ax_p2.set_ylabel('Load (kW)', color=color_load)
        line2 = ax_p2.plot(df_profile.index, df_profile[col_load], color=color_load, linewidth=2, linestyle='--', label='Load Profile')
        ax_p2.tick_params(axis='y', labelcolor=color_load)
        ax_p2.set_ylim(0, y_limit) 
        
        ax_p1.set_title("Avg Daily Profile: Solar vs Load")
    
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax_p1.legend(lines, labels, loc='upper right', fontsize='small')
        
        plt.tight_layout()
        st.pyplot(fig_prof)

    with c2:
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
        
        fig_h_sol, ax_hs = plt.subplots(figsize=(6, 4))
        im_sol = ax_hs.imshow(data_matrix_solar, cmap='YlOrRd', aspect='auto', interpolation='nearest', origin='lower')
        
        ax_hs.set_xlabel("Day")
        ax_hs.set_ylabel("Hour")
        ax_hs.set_title("Irradiance Heatmap")
        
        cbar_sol = ax_hs.figure.colorbar(im_sol, ax=ax_hs, fraction=0.046, pad=0.04)
        cbar_sol.set_label("Irradiance ($Wh/m^2$)")
        plt.tight_layout()
        st.pyplot(fig_h_sol)