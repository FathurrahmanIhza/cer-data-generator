import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import calendar
import pandas as pd

def plot_annual_overview(df_vis_year, col_bat, selected_vis_year):

    
    factor = 5/60 
    st.markdown(f"### 📅 Annual Overview ({selected_vis_year})")

    df_monthly_vis = df_vis_year.set_index('timestamp').copy()
    df_monthly_vis['pv_kwh'] = df_monthly_vis['solar_output_kw'] * factor
    df_monthly_vis['bat_dis_kwh'] = df_monthly_vis[col_bat].apply(lambda x: x if x > 0 else 0) * factor
    df_monthly_vis['grid_imp_kwh'] = df_monthly_vis['grid_net_kw'].apply(lambda x: x if x > 0 else 0) * factor

    df_monthly_agg = df_monthly_vis[['pv_kwh', 'bat_dis_kwh', 'grid_imp_kwh']].resample('ME').sum()
    df_monthly_agg['month_name'] = df_monthly_agg.index.strftime('%b') 
    
    df_monthly_agg['total_supply'] = df_monthly_agg['pv_kwh'] + df_monthly_agg['bat_dis_kwh'] + df_monthly_agg['grid_imp_kwh']
    df_monthly_agg['total_supply'] = df_monthly_agg['total_supply'].replace(0, 1)
    
    df_monthly_agg['pct_pv'] = (df_monthly_agg['pv_kwh'] / df_monthly_agg['total_supply']) * 100
    df_monthly_agg['pct_bat'] = (df_monthly_agg['bat_dis_kwh'] / df_monthly_agg['total_supply']) * 100
    df_monthly_agg['pct_grid'] = (df_monthly_agg['grid_imp_kwh'] / df_monthly_agg['total_supply']) * 100
    
    colors = ['#1f77b4', '#8c564b', '#add8e6'] 
    months = df_monthly_agg['month_name']

    # --- BARIS 1 (Energy Source kWh & Percentage) ---
    c1, c2 = st.columns(2)
    
    with c1:
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.bar(months, df_monthly_agg['pv_kwh'], color=colors[0], label='PV')
        ax1.bar(months, df_monthly_agg['bat_dis_kwh'], bottom=df_monthly_agg['pv_kwh'], color=colors[1], label='Bat Discharge')
        ax1.bar(months, df_monthly_agg['grid_imp_kwh'], bottom=df_monthly_agg['pv_kwh']+df_monthly_agg['bat_dis_kwh'], color=colors[2], label='Grid Import')
        
        ax1.set_title("Monthly Energy Source")
        ax1.set_ylabel("Energy (kWh)")
        ax1.legend(fontsize='small')
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
        ax2.grid(axis='y', alpha=0.3)
        plt.tight_layout() 
        st.pyplot(fig2)

    # --- BARIS 2 (VPP Heatmap & Battery Breakdown) ---
    c3, c4 = st.columns(2)

    with c3:
        if 'vpp_status' in df_vis_year.columns:
            df_vis_year['vpp_hours'] = df_vis_year['vpp_status'].apply(lambda x: 5/60 if x else 0)
            daily_vpp = df_vis_year.set_index('timestamp')['vpp_hours'].resample('D').sum().reset_index()
            
            daily_vpp['m'] = daily_vpp['timestamp'].dt.month
            daily_vpp['d'] = daily_vpp['timestamp'].dt.day
            
            heatmap_data = daily_vpp.pivot(index='m', columns='d', values='vpp_hours')
            heatmap_data = heatmap_data.reindex(index=range(1, 13), columns=range(1, 32))
            data_matrix = heatmap_data.to_numpy()
            
            fig_heat, ax_h = plt.subplots(figsize=(6, 4))
            im = ax_h.imshow(data_matrix, cmap='Oranges', aspect='auto', interpolation='nearest', vmin=0)
            
            ax_h.set_xticks(np.arange(0, 31, 2))
            ax_h.set_xticklabels(np.arange(1, 32, 2), fontsize=7)
            ax_h.set_yticks(np.arange(12))
            ax_h.set_yticklabels([calendar.month_abbr[i] for i in range(1, 13)], fontsize=8)
            
            ax_h.set_title("VPP Heatmap")
            
            
            cbar = ax_h.figure.colorbar(im, ax=ax_h, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)
            cbar.set_label("Hour")
            
            plt.tight_layout()
            st.pyplot(fig_heat)
        else:
            st.info("No VPP Status Data Available")

    with c4:
        df_bat = df_vis_year.copy()
        
        df_bat['vpp_dis_kw'] = df_bat.apply(
            lambda x: x[col_bat] if (x.get('vpp_status') == True and x[col_bat] > 0) else 0, axis=1
        )
        df_bat['norm_dis_kw'] = df_bat.apply(
            lambda x: x[col_bat] if (x.get('vpp_status') == False and x[col_bat] > 0) else 0, axis=1
        )
        
        daily_bat = df_bat.set_index('timestamp')[['vpp_dis_kw', 'norm_dis_kw']].resample('D').sum() * factor
        
        fig_break, ax_b = plt.subplots(figsize=(6, 4))
        colors_bat = ['#d35400', '#55a868'] 
        labels_bat = ['VPP', 'Normal']
        
        ax_b.stackplot(daily_bat.index, 
                       daily_bat['vpp_dis_kw'], 
                       daily_bat['norm_dis_kw'],
                       colors=colors_bat, labels=labels_bat, alpha=0.8)
        
        ax_b.set_title("Normal VS VPP Discharge")
        ax_b.set_ylabel("Energy (kWh)")
        
        ax_b.xaxis.set_major_locator(mdates.MonthLocator())
        ax_b.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        plt.setp(ax_b.get_xticklabels(), rotation=0, ha="center", fontsize=9)
        
        ax_b.set_xlim(daily_bat.index[0], daily_bat.index[-1])
        ax_b.legend(loc='upper left', fontsize='x-small')
        ax_b.grid(True, alpha=0.3)
        
        plt.tight_layout() 
        st.pyplot(fig_break)


def plot_monthly_analysis(df_vis_month, col_load, selected_month_name, selected_vis_year):

    st.markdown(f"### 📉 Monthly Analysis ({selected_month_name} {selected_vis_year})")
    
    # --- BARIS 3 (Avg GHI vs Load Profile & Hourly Solar Heatmap) ---
    c1, c2 = st.columns(2)
    
    with c1:
        df_profile = df_vis_month.groupby(df_vis_month['timestamp'].dt.hour)[['solar_output_kw', col_load]].mean()
        
        fig_prof, ax_p1 = plt.subplots(figsize=(6, 4))
        
        color_ghi = 'orange'
        ax_p1.set_xlabel('Hour (0-23)')
        ax_p1.set_ylabel('Solar Output (kW)', color=color_ghi)
        line1 = ax_p1.plot(df_profile.index, df_profile['solar_output_kw'], color=color_ghi, linewidth=2, label='GHI')
        ax_p1.tick_params(axis='y', labelcolor=color_ghi)
        ax_p1.grid(True, alpha=0.3)
        
        ax_p2 = ax_p1.twinx()  
        color_load = '#d62728' 
        ax_p2.set_ylabel('Load (kW)', color=color_load)
        line2 = ax_p2.plot(df_profile.index, df_profile[col_load], color=color_load, linewidth=2, linestyle='--', label='Load')
        ax_p2.tick_params(axis='y', labelcolor=color_load)
        
        ax_p1.set_title("Solar Output VS Load Profile")
        plt.tight_layout()
        st.pyplot(fig_prof)

    with c2:
        factor = 5/60
        df_heat_solar = df_vis_month.set_index('timestamp')[['irradiance']].resample('h').sum() * factor
        df_heat_solar = df_heat_solar.reset_index()
        df_heat_solar['d'] = df_heat_solar['timestamp'].dt.day
        df_heat_solar['h'] = df_heat_solar['timestamp'].dt.hour
        
        solar_matrix = df_heat_solar.pivot(index='h', columns='d', values='irradiance')
        
        curr_year = df_vis_month['timestamp'].dt.year.iloc[0]
        curr_month = df_vis_month['timestamp'].dt.month.iloc[0]
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