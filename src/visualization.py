import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap, MarkerCluster, TimestampedGeoJson
from pathlib import Path
from datetime import datetime
import calendar
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12


def plot_cluster_scatter(df, cluster_col='kmeans_cluster', title='Cluster Map', save_path=None):
    fig, ax = plt.subplots(figsize=(16, 10))
    
    unique_clusters = sorted(df[cluster_col].unique())
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))
    
    for idx, cluster in enumerate(unique_clusters):
        cluster_data = df[df[cluster_col] == cluster]
        label = f'Cluster {cluster} (n={len(cluster_data)})' if cluster != -1 else f'Noise (n={len(cluster_data)})'
        ax.scatter(cluster_data['longitude'], cluster_data['latitude'], 
                  c=[colors[idx]], label=label, alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('Longitude', fontsize=13)
    ax.set_ylabel('Latitude', fontsize=13)
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, 
             frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return fig


def plot_spatial_distribution(df, title='Spatial Distribution', save_path=None):
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Create density-based coloring
    scatter = ax.scatter(df['longitude'], df['latitude'], 
                        alpha=0.5, s=30, c='crimson', 
                        edgecolors='darkred', linewidth=0.4,
                        cmap='hot_r')
    
    ax.set_xlabel('Longitude', fontsize=13)
    ax.set_ylabel('Latitude', fontsize=13)
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_facecolor('#f0f0f0')
    
    # Add count annotation
    ax.text(0.02, 0.98, f'Total: {len(df):,} incidents', 
            transform=ax.transAxes, fontsize=12, 
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return fig


def plot_distance_histogram(df, col='dist_to_bathroom_m', title='Distance Distribution', save_path=None):
    fig, ax = plt.subplots(figsize=(14, 7))
    
    data = df[col].dropna()
    n, bins, patches = ax.hist(data, bins=50, color='steelblue', 
                               edgecolor='black', alpha=0.7, linewidth=0.5)
    
    # Color bars by distance
    cm = plt.cm.RdYlGn_r
    for i, patch in enumerate(patches):
        patch.set_facecolor(cm(i / len(patches)))
    
    # Add statistics
    mean_dist = data.mean()
    median_dist = data.median()
    
    ax.axvline(mean_dist, color='blue', linestyle='-', linewidth=2.5, 
               label=f'Mean: {mean_dist:.0f}m', alpha=0.8)
    ax.axvline(median_dist, color='green', linestyle='-', linewidth=2.5, 
               label=f'Median: {median_dist:.0f}m', alpha=0.8)
    ax.axvline(500, color='red', linestyle='--', linewidth=2, 
               label='500m threshold', alpha=0.7)
    ax.axvline(800, color='orange', linestyle='--', linewidth=2, 
               label='800m threshold', alpha=0.7)
    
    ax.set_xlabel(f'{col.replace("_", " ").title()}', fontsize=13)
    ax.set_ylabel('Frequency', fontsize=13)
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    ax.legend(fontsize=11, frameon=True, fancybox=True, shadow=True)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return fig


def plot_cluster_bar_chart(df, cluster_col='kmeans_cluster', title='Cluster Sizes', save_path=None):
    fig, ax = plt.subplots(figsize=(12, 7))
    
    cluster_counts = df[cluster_col].value_counts().sort_index()
    colors = sns.color_palette('viridis', len(cluster_counts))
    
    bars = ax.bar(cluster_counts.index, cluster_counts.values, 
                   color=colors, edgecolor='black', alpha=0.8, linewidth=1.5)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Cluster ID', fontsize=13)
    ax.set_ylabel('Count', fontsize=13)
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return fig


def create_folium_heatmap(df, title='Heatmap', save_path=None):
    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles='OpenStreetMap')
    
    heat_data = [[row['latitude'], row['longitude']] for _, row in df.iterrows()]
    HeatMap(heat_data, radius=15, blur=25, max_zoom=13).add_to(m)
    
    folium.LayerControl().add_to(m)
    
    if save_path:
        m.save(save_path)
    
    return m


def create_folium_cluster_map(df, cluster_col='kmeans_cluster', title='Cluster Map', save_path=None):
    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles='OpenStreetMap')
    
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 
              'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'white', 'pink', 'lightblue', 
              'lightgreen', 'gray', 'black', 'lightgray']
    
    unique_clusters = sorted(df[cluster_col].unique())
    
    for cluster in unique_clusters:
        cluster_data = df[df[cluster_col] == cluster]
        color = colors[cluster % len(colors)] if cluster != -1 else 'black'
        
        for _, row in cluster_data.iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=5,
                popup=f"Cluster {cluster}",
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.6
            ).add_to(m)
    
    if save_path:
        m.save(save_path)
    
    return m


def create_multi_layer_map(needle_df, encampment_df, bathroom_df, save_path=None):
    center_lat = needle_df['latitude'].mean()
    center_lon = needle_df['longitude'].mean()
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles='OpenStreetMap')
    
    needle_layer = folium.FeatureGroup(name='Needle Cases', show=True)
    encampment_layer = folium.FeatureGroup(name='Encampments', show=True)
    bathroom_layer = folium.FeatureGroup(name='Bathrooms', show=True)
    
    for _, row in needle_df.sample(min(1000, len(needle_df))).iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=3,
            color='red',
            fill=True,
            fillOpacity=0.4,
            popup='Needle Case'
        ).add_to(needle_layer)
    
    for _, row in encampment_df.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=5,
            color='orange',
            fill=True,
            fillOpacity=0.6,
            popup='Encampment'
        ).add_to(encampment_layer)
    
    for _, row in bathroom_df.iterrows():
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            icon=folium.Icon(color='blue', icon='tint', prefix='fa'),
            popup='Bathroom'
        ).add_to(bathroom_layer)
    
    needle_layer.add_to(m)
    encampment_layer.add_to(m)
    bathroom_layer.add_to(m)
    
    folium.LayerControl().add_to(m)
    
    if save_path:
        m.save(save_path)
    
    return m


def create_spatiotemporal_map(df, date_col='opened', title='Spatio-Temporal Map', save_path=None, time_period='month', max_points=5000):
    df_temp = df.copy()
    df_temp[date_col] = pd.to_datetime(df_temp[date_col], errors='coerce')
    df_temp = df_temp.dropna(subset=[date_col, 'latitude', 'longitude'])
    
    if len(df_temp) == 0:
        print("No valid temporal data for spatio-temporal map")
        return None
    
    # Track original size for reporting
    original_size = len(df_temp)
    
    # Sample entire dataset if needed (before time aggregation)
    # Note: With cumulative display, each incident is duplicated across ~45 avg periods
    # So max_points * 45 = total features in the HTML file
    if len(df_temp) > max_points:
        df_temp = df_temp.sample(n=max_points, random_state=42)
        print(f"    Sampling {max_points:,} incidents from {original_size:,} total ({100*max_points/original_size:.1f}%) for performance")
    
    # Create time periods for each incident
    if time_period == 'month':
        df_temp['time_period'] = df_temp[date_col].dt.to_period('M')
        df_temp['time_str'] = df_temp['time_period'].astype(str)
        df_temp['time_start'] = df_temp[date_col].dt.to_period('M').dt.start_time
    elif time_period == 'quarter':
        df_temp['time_period'] = df_temp[date_col].dt.to_period('Q')
        df_temp['time_str'] = df_temp['time_period'].astype(str)
        df_temp['time_start'] = df_temp[date_col].dt.to_period('Q').dt.start_time
    elif time_period == 'year':
        df_temp['time_period'] = df_temp[date_col].dt.to_period('Y')
        df_temp['time_str'] = df_temp['time_period'].astype(str)
        df_temp['time_start'] = df_temp[date_col].dt.to_period('Y').dt.start_time
    else:
        df_temp['time_period'] = df_temp[date_col].dt.to_period('M')
        df_temp['time_str'] = df_temp['time_period'].astype(str)
        df_temp['time_start'] = df_temp[date_col].dt.to_period('M').dt.start_time
    
    # Sort by time
    df_temp = df_temp.sort_values(date_col)
    
    # Get center coordinates
    center_lat = df_temp['latitude'].mean()
    center_lon = df_temp['longitude'].mean()
    
    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles='CartoDB positron',
        prefer_canvas=True
    )
    
    # Get sorted time periods and create color mapping
    time_periods = sorted(df_temp['time_str'].unique())
    time_to_color = {}
    period_to_timestamp = {}  # Pre-compute timestamps for performance
    
    for idx, period in enumerate(time_periods):
        color_idx = idx / max(len(time_periods) - 1, 1)
        color = plt.cm.RdYlBu_r(color_idx)
        hex_color = '#{:02x}{:02x}{:02x}'.format(
            int(color[0] * 255), 
            int(color[1] * 255), 
            int(color[2] * 255)
        )
        time_to_color[period] = hex_color
        
        # Pre-compute period start time to avoid repeated Period object creation
        period_obj = pd.Period(period, freq=time_period[0].upper())
        period_to_timestamp[period] = period_obj.start_time.isoformat()
    
    # Add neighborhood column if available for popups
    has_neighborhood = 'neighborhood' in df_temp.columns
    
    print(f"    Generating cumulative features for {len(df_temp):,} incidents across {len(time_periods)} periods...")
    
    # Create GeoJSON features for TimestampedGeoJson with CUMULATIVE display
    # Each incident is duplicated across all future time periods to create accumulation effect
    features = []
    
    for _, row in df_temp.iterrows():
        # Get color based on time period when incident occurred
        hex_color = time_to_color[row['time_str']]
        
        # Build popup text
        popup_text = f"{row[date_col].strftime('%Y-%m-%d %H:%M')}"
        if has_neighborhood and pd.notna(row.get('neighborhood')):
            popup_text += f" | {row['neighborhood']}"
        if 'status' in row and pd.notna(row.get('status')):
            popup_text += f" | {row['status']}"
        
        # Create copies of this incident for all time periods >= its occurrence time
        # This makes points accumulate as the slider progresses
        incident_period_idx = time_periods.index(row['time_str'])
        
        for future_period_idx in range(incident_period_idx, len(time_periods)):
            future_period = time_periods[future_period_idx]
            # Use pre-computed timestamp for efficiency
            future_time_iso = period_to_timestamp[future_period]
            
            feature = {
                'type': 'Feature',
                'geometry': {
                    'type': 'Point',
                    'coordinates': [row['longitude'], row['latitude']],
                },
                'properties': {
                    'time': future_time_iso,
                    'popup': popup_text,
                    'icon': 'circle',
                    'iconstyle': {
                        'fillColor': hex_color,  # Keep original color
                        'fillOpacity': 0.6,
                        'stroke': 'true',
                        'radius': 5,
                        'weight': 1,
                        'color': '#333'
                    }
                }
            }
            features.append(feature)
    
    # Create TimestampedGeoJson
    timestamped_geojson = TimestampedGeoJson(
        {
            'type': 'FeatureCollection',
            'features': features
        },
        period=f'P1{time_period[0].upper()}',  # P1M for month, P1Q for quarter, P1Y for year
        auto_play=False,
        loop=False,
        max_speed=2,
        loop_button=True,
        date_options='YYYY-MM',
        time_slider_drag_update=True,
        duration='P1D'  # How long each period displays
    )
    
    timestamped_geojson.add_to(m)
    
    # Add title
    title_html = f'''
    <div style="position: fixed; 
                top: 10px; 
                left: 50px; 
                width: 450px; 
                height: 70px; 
                background-color: white; 
                border:2px solid grey; 
                z-index:9999; 
                font-size:16px;
                padding: 10px;
                border-radius: 5px;
                box-shadow: 2px 2px 6px rgba(0,0,0,0.3);
                ">
    <h4 style="margin:0;">{title}</h4>
    <p style="margin:5px 0 0 0; font-size:12px;">
        Use the timeline slider to view cumulative incidents over time<br>
        <span style="font-size:11px; color:#666;">▶️ Play | ⏸️ Pause | 🔄 Loop • Points accumulate as time progresses</span>
    </p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Add legend
    legend_html = f'''
    <div style="position: fixed; 
                bottom: 120px; 
                left: 50px; 
                width: 220px; 
                background-color: white; 
                border:2px solid grey; 
                z-index:9999; 
                font-size:12px;
                padding: 10px;
                border-radius: 5px;
                box-shadow: 2px 2px 6px rgba(0,0,0,0.3);
                ">
    <p style="margin:0; font-weight:bold;">Time Period Colors</p>
    <div style="margin-top:5px;">
        <span style="background: linear-gradient(to right, #313695, #ffffbf, #a50026); 
                     display: block; 
                     height: 15px; 
                     width: 100%;
                     border: 1px solid #999;"></span>
    </div>
    <div style="display: flex; justify-content: space-between; margin-top: 3px; font-size:10px;">
        <span>{time_periods[0]}</span>
        <span>{time_periods[-1]}</span>
    </div>
    <p style="margin:10px 0 0 0; font-size:11px;">
        <b>Total Periods:</b> {len(time_periods)}<br>
        <b>Total Incidents:</b> {len(df_temp):,}<br>
        <b>Aggregation:</b> {time_period.capitalize()}
    </p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    if save_path:
        m.save(save_path)
        print(f"    Cumulative map saved: {len(time_periods)} periods × {len(df_temp):,} incidents = ~{len(features):,} total features")
    
    return m


# ==================== TEMPORAL VISUALIZATIONS ==

def plot_time_series(df, date_col='opened', title='Time Series', save_path=None, freq='D'):
    """
    Plot time series of incident counts over time.
    
    Parameters:
    - df: DataFrame with datetime column
    - date_col: Name of datetime column
    - title: Plot title
    - save_path: Path to save figure
    - freq: Frequency for resampling ('D'=daily, 'W'=weekly, 'M'/'ME'=monthly)
    """
    df_temp = df.copy()
    df_temp[date_col] = pd.to_datetime(df_temp[date_col], errors='coerce')
    df_temp = df_temp.dropna(subset=[date_col])
    
    if len(df_temp) == 0:
        print(f"No valid dates in {date_col}")
        return None
    
    # Use 'ME' for monthly frequency in newer pandas
    if freq == 'M':
        try:
            ts = df_temp.set_index(date_col).resample('ME').size()
        except ValueError:
            ts = df_temp.set_index(date_col).resample('M').size()
    else:
        ts = df_temp.set_index(date_col).resample(freq).size()
    
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(ts.index, ts.values, linewidth=2, color='steelblue', alpha=0.8)
    ax.fill_between(ts.index, ts.values, alpha=0.3, color='steelblue')
    
    # Add moving average
    window = 7 if freq == 'D' else 4
    ma = ts.rolling(window=window, center=True).mean()
    ax.plot(ma.index, ma.values, linewidth=3, color='coral', 
            label=f'{window}-period Moving Avg', linestyle='--')
    
    ax.set_xlabel('Date', fontsize=13)
    ax.set_ylabel('Count', fontsize=13)
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return fig


def plot_monthly_trends(df, date_col='opened', title='Monthly Trends', save_path=None):
    """Plot monthly aggregated data with trend line."""
    df_temp = df.copy()
    df_temp[date_col] = pd.to_datetime(df_temp[date_col], errors='coerce')
    df_temp = df_temp.dropna(subset=[date_col])
    
    if len(df_temp) == 0:
        return None
    
    # Monthly counts - use 'ME' for month end (newer pandas versions)
    try:
        monthly = df_temp.set_index(date_col).resample('ME').size()
    except ValueError:
        monthly = df_temp.set_index(date_col).resample('M').size()
    
    fig, ax = plt.subplots(figsize=(16, 7))
    
    # Bar plot
    bars = ax.bar(monthly.index, monthly.values, width=20, 
                   color='teal', alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Color bars by value
    colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, len(monthly)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # Add trend line
    x_numeric = np.arange(len(monthly))
    z = np.polyfit(x_numeric, monthly.values, 2)
    p = np.poly1d(z)
    ax.plot(monthly.index, p(x_numeric), linewidth=3, 
            color='darkblue', linestyle='--', label='Trend', alpha=0.8)
    
    ax.set_xlabel('Month', fontsize=13)
    ax.set_ylabel('Count', fontsize=13)
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return fig


def plot_day_of_week_pattern(df, date_col='opened', title='Day of Week Pattern', save_path=None):
    """Analyze and visualize patterns by day of week."""
    df_temp = df.copy()
    df_temp[date_col] = pd.to_datetime(df_temp[date_col], errors='coerce')
    df_temp = df_temp.dropna(subset=[date_col])
    
    if len(df_temp) == 0:
        return None
    
    df_temp['day_of_week'] = df_temp[date_col].dt.dayofweek
    df_temp['day_name'] = df_temp[date_col].dt.day_name()
    
    # Count by day
    day_counts = df_temp.groupby(['day_of_week', 'day_name']).size().reset_index(name='count')
    day_counts = day_counts.sort_values('day_of_week')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = sns.color_palette("coolwarm", len(day_counts))
    bars = ax.bar(day_counts['day_name'], day_counts['count'], 
                   color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Day of Week', fontsize=13)
    ax.set_ylabel('Count', fontsize=13)
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return fig


def plot_hour_of_day_pattern(df, date_col='opened', title='Hour of Day Pattern', save_path=None):
    """Analyze and visualize patterns by hour of day."""
    df_temp = df.copy()
    df_temp[date_col] = pd.to_datetime(df_temp[date_col], errors='coerce')
    df_temp = df_temp.dropna(subset=[date_col])
    
    if len(df_temp) == 0:
        return None
    
    df_temp['hour'] = df_temp[date_col].dt.hour
    hour_counts = df_temp['hour'].value_counts().sort_index()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Create color gradient
    colors = plt.cm.plasma(np.linspace(0, 1, 24))
    
    bars = ax.bar(hour_counts.index, hour_counts.values, 
                   color=colors, edgecolor='black', linewidth=0.5, alpha=0.8)
    
    # Highlight peak hours
    max_hour = hour_counts.idxmax()
    bars[max_hour].set_color('red')
    bars[max_hour].set_alpha(1.0)
    bars[max_hour].set_edgecolor('darkred')
    bars[max_hour].set_linewidth(2)
    
    ax.set_xlabel('Hour of Day', fontsize=13)
    ax.set_ylabel('Count', fontsize=13)
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(range(24))
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add time labels
    ax.axvspan(0, 6, alpha=0.1, color='blue', label='Night')
    ax.axvspan(6, 12, alpha=0.1, color='yellow', label='Morning')
    ax.axvspan(12, 18, alpha=0.1, color='orange', label='Afternoon')
    ax.axvspan(18, 24, alpha=0.1, color='purple', label='Evening')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return fig


def plot_seasonal_pattern(df, date_col='opened', title='Seasonal Pattern', save_path=None):
    """Analyze and visualize seasonal patterns (by month)."""
    df_temp = df.copy()
    df_temp[date_col] = pd.to_datetime(df_temp[date_col], errors='coerce')
    df_temp = df_temp.dropna(subset=[date_col])
    
    if len(df_temp) == 0:
        return None
    
    df_temp['month'] = df_temp[date_col].dt.month
    df_temp['month_name'] = df_temp[date_col].dt.month_name()
    
    # Count by month
    month_counts = df_temp.groupby(['month', 'month_name']).size().reset_index(name='count')
    month_counts = month_counts.sort_values('month')
    
    # Create a complete 12-month dataset with zeros for missing months
    all_months = pd.DataFrame({
        'month': range(1, 13),
        'month_name': [calendar.month_name[i] for i in range(1, 13)]
    })
    month_counts_complete = all_months.merge(month_counts, on=['month', 'month_name'], how='left')
    month_counts_complete['count'] = month_counts_complete['count'].fillna(0)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bar chart
    colors = sns.color_palette("Spectral", 12)
    axes[0].bar(month_counts_complete['month_name'], month_counts_complete['count'], 
                color=colors, edgecolor='black', linewidth=1, alpha=0.8)
    axes[0].set_xlabel('Month', fontsize=13)
    axes[0].set_ylabel('Count', fontsize=13)
    axes[0].set_title(f'{title} - Bar Chart', fontsize=14, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Polar plot
    theta = np.linspace(0, 2*np.pi, 12, endpoint=False)
    width = 2*np.pi / 12
    
    ax_polar = plt.subplot(122, projection='polar')
    bars = ax_polar.bar(theta, month_counts_complete['count'], width=width, 
                         color=colors, edgecolor='black', linewidth=1, alpha=0.8)
    ax_polar.set_theta_zero_location('N')
    ax_polar.set_theta_direction(-1)
    ax_polar.set_xticks(theta)
    ax_polar.set_xticklabels([m[:3] for m in month_counts_complete['month_name']])
    ax_polar.set_title(f'{title} - Polar View', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return fig


def plot_calendar_heatmap(df, date_col='opened', title='Calendar Heatmap', save_path=None):
    """Create a calendar heatmap showing daily counts."""
    df_temp = df.copy()
    df_temp[date_col] = pd.to_datetime(df_temp[date_col], errors='coerce')
    df_temp = df_temp.dropna(subset=[date_col])
    
    if len(df_temp) == 0:
        return None
    
    # Daily counts
    daily = df_temp.set_index(date_col).resample('D').size()
    
    # Create pivot table for heatmap
    df_heatmap = pd.DataFrame({
        'date': daily.index,
        'count': daily.values
    })
    df_heatmap['year'] = df_heatmap['date'].dt.year
    df_heatmap['month'] = df_heatmap['date'].dt.month
    df_heatmap['day'] = df_heatmap['date'].dt.day
    
    # Get most recent year for visualization
    recent_year = df_heatmap['year'].max()
    df_year = df_heatmap[df_heatmap['year'] == recent_year]
    
    pivot = df_year.pivot_table(values='count', index='day', columns='month', aggfunc='sum', fill_value=0)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    sns.heatmap(pivot, cmap='YlOrRd', annot=False, fmt='d', 
                linewidths=0.5, cbar_kws={'label': 'Count'},
                ax=ax, robust=True)
    
    ax.set_xlabel('Month', fontsize=13)
    ax.set_ylabel('Day of Month', fontsize=13)
    ax.set_title(f'{title} - {recent_year}', fontsize=15, fontweight='bold', pad=20)
    
    # Set month labels only for columns that exist
    month_labels = [calendar.month_abbr[int(i)] for i in pivot.columns]
    ax.set_xticklabels(month_labels, rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return fig


def plot_temporal_neighborhood_comparison(df, date_col='opened', neighborhood_col='neighborhood', title='Neighborhood Comparison Over Time', save_path=None):
    df_temp = df.copy()
    df_temp[date_col] = pd.to_datetime(df_temp[date_col], errors='coerce')
    df_temp = df_temp.dropna(subset=[date_col, neighborhood_col])
    
    if len(df_temp) == 0:
        return None
    
    # Get top neighborhoods
    top_neighborhoods = df_temp[neighborhood_col].value_counts().head(10).index
    df_filtered = df_temp[df_temp[neighborhood_col].isin(top_neighborhoods)]
    
    # Monthly counts by neighborhood
    df_filtered['year_month'] = df_filtered[date_col].dt.to_period('M')
    monthly_neighborhood = df_filtered.groupby(['year_month', neighborhood_col]).size().reset_index(name='count')
    monthly_neighborhood['year_month'] = monthly_neighborhood['year_month'].astype(str)
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Plot each neighborhood
    for neighborhood in top_neighborhoods:
        data = monthly_neighborhood[monthly_neighborhood[neighborhood_col] == neighborhood]
        ax.plot(data['year_month'], data['count'], 
                marker='o', linewidth=2, alpha=0.7, label=neighborhood)
    
    ax.set_xlabel('Month', fontsize=13)
    ax.set_ylabel('Count', fontsize=13)
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return fig


def plot_yearly_comparison(df, date_col='opened', title='Year-over-Year Comparison', save_path=None):
    df_temp = df.copy()
    df_temp[date_col] = pd.to_datetime(df_temp[date_col], errors='coerce')
    df_temp = df_temp.dropna(subset=[date_col])
    
    if len(df_temp) == 0:
        return None
    
    df_temp['year'] = df_temp[date_col].dt.year
    df_temp['month'] = df_temp[date_col].dt.month
    
    # Count by year and month
    yearly_monthly = df_temp.groupby(['year', 'month']).size().reset_index(name='count')
    
    fig, ax = plt.subplots(figsize=(16, 7))
    
    years = sorted(yearly_monthly['year'].unique())
    colors = sns.color_palette("tab10", len(years))
    
    for i, year in enumerate(years):
        data = yearly_monthly[yearly_monthly['year'] == year]
        ax.plot(data['month'], data['count'], 
                marker='o', linewidth=2.5, alpha=0.8, 
                label=year, color=colors[i], markersize=8)
    
    ax.set_xlabel('Month', fontsize=13)
    ax.set_ylabel('Count', fontsize=13)
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels([calendar.month_abbr[i] for i in range(1, 13)])
    ax.legend(title='Year', fontsize=11, title_fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return fig


def create_interactive_temporal_plot(df, date_col='opened', title='Interactive Time Series', save_path=None):
    """Create an interactive plotly time series visualization."""
    df_temp = df.copy()
    df_temp[date_col] = pd.to_datetime(df_temp[date_col], errors='coerce')
    df_temp = df_temp.dropna(subset=[date_col])
    
    if len(df_temp) == 0:
        return None
    
    # Daily counts
    daily = df_temp.set_index(date_col).resample('D').size().reset_index(name='count')
    
    # Create figure with subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Daily Incident Count', 'Cumulative Count'),
        row_heights=[0.6, 0.4],
        vertical_spacing=0.12
    )
    
    # Daily counts
    fig.add_trace(
        go.Scatter(x=daily[date_col], y=daily['count'],
                   mode='lines', name='Daily Count',
                   line=dict(color='steelblue', width=2),
                   fill='tozeroy', fillcolor='rgba(70, 130, 180, 0.3)'),
        row=1, col=1
    )
    
    # Cumulative
    daily['cumulative'] = daily['count'].cumsum()
    fig.add_trace(
        go.Scatter(x=daily[date_col], y=daily['cumulative'],
                   mode='lines', name='Cumulative',
                   line=dict(color='coral', width=3)),
        row=2, col=1
    )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative Count", row=2, col=1)
    
    fig.update_layout(
        title_text=title,
        title_font_size=18,
        showlegend=True,
        height=800,
        hovermode='x unified'
    )
    
    if save_path:
        fig.write_html(save_path)
    
    return fig


def plot_status_comparison(df, date_col='opened', status_col='status', 
                          title='Status Comparison Over Time', save_path=None):
    """Compare open vs closed status over time."""
    df_temp = df.copy()
    df_temp[date_col] = pd.to_datetime(df_temp[date_col], errors='coerce')
    df_temp = df_temp.dropna(subset=[date_col])
    
    if status_col not in df_temp.columns or len(df_temp) == 0:
        return None
    
    # Monthly counts by status
    df_temp['year_month'] = df_temp[date_col].dt.to_period('M')
    status_monthly = df_temp.groupby(['year_month', status_col]).size().unstack(fill_value=0)
    status_monthly.index = status_monthly.index.astype(str)
    
    fig, ax = plt.subplots(figsize=(16, 7))
    
    # Stacked area chart
    status_monthly.plot(kind='area', ax=ax, alpha=0.7, linewidth=2, 
                       color=['#2ecc71', '#e74c3c'], stacked=True)
    
    ax.set_xlabel('Month', fontsize=13)
    ax.set_ylabel('Count', fontsize=13)
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    ax.legend(title='Status', fontsize=11, title_fontsize=12, 
             loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return fig


def plot_spatial_temporal_stats(df, date_col='opened', title='Spatial-Temporal Statistics', save_path=None):
    """Create a dashboard of spatial and temporal statistics."""
    df_temp = df.copy()
    df_temp[date_col] = pd.to_datetime(df_temp[date_col], errors='coerce')
    df_temp = df_temp.dropna(subset=[date_col])
    
    if len(df_temp) == 0:
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    
    # 1. Distance to bathroom distribution box plot
    if 'dist_to_bathroom_m' in df_temp.columns:
        df_temp['dist_to_bathroom_m'].dropna().plot(kind='box', ax=axes[0, 0], 
                                                     patch_artist=True, vert=False,
                                                     boxprops=dict(facecolor='lightblue', alpha=0.7))
        axes[0, 0].set_xlabel('Distance (m)', fontsize=12)
        axes[0, 0].set_title('Distance to Bathroom Distribution', fontsize=13, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Incidents per neighborhood (top 10)
    if 'neighborhood' in df_temp.columns:
        neighborhood_counts = df_temp['neighborhood'].value_counts().head(10)
        colors = sns.color_palette('viridis', len(neighborhood_counts))
        neighborhood_counts.plot(kind='barh', ax=axes[0, 1], color=colors, edgecolor='black')
        axes[0, 1].set_xlabel('Count', fontsize=12)
        axes[0, 1].set_ylabel('', fontsize=12)
        axes[0, 1].set_title('Top 10 Neighborhoods', fontsize=13, fontweight='bold')
        axes[0, 1].grid(axis='x', alpha=0.3)
    
    # 3. Monthly trend
    try:
        monthly = df_temp.set_index(date_col).resample('ME').size()
    except ValueError:
        monthly = df_temp.set_index(date_col).resample('M').size()
    axes[1, 0].plot(monthly.index, monthly.values, marker='o', 
                   linewidth=2.5, color='coral', markersize=6)
    axes[1, 0].fill_between(monthly.index, monthly.values, alpha=0.3, color='coral')
    axes[1, 0].set_xlabel('Month', fontsize=12)
    axes[1, 0].set_ylabel('Count', fontsize=12)
    axes[1, 0].set_title('Monthly Incident Count', fontsize=13, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. Summary statistics text
    axes[1, 1].axis('off')
    stats_text = f"""
    SUMMARY STATISTICS
    {'='*40}
    
    Total Incidents: {len(df_temp):,}
    Date Range: {df_temp[date_col].min().strftime('%Y-%m-%d')} to 
                {df_temp[date_col].max().strftime('%Y-%m-%d')}
    Duration: {(df_temp[date_col].max() - df_temp[date_col].min()).days} days
    
    Average per Day: {len(df_temp) / max((df_temp[date_col].max() - df_temp[date_col].min()).days, 1):.1f}
    Average per Month: {len(df_temp) / max(len(monthly), 1):.1f}
    """
    
    if 'dist_to_bathroom_m' in df_temp.columns:
        stats_text += f"""
    Distance to Bathroom:
      Mean: {df_temp['dist_to_bathroom_m'].mean():.1f}m
      Median: {df_temp['dist_to_bathroom_m'].median():.1f}m
      Max: {df_temp['dist_to_bathroom_m'].max():.1f}m
        """
    
    if 'bathrooms_within_500m' in df_temp.columns:
        stats_text += f"""
    Bathrooms within 500m:
      Mean: {df_temp['bathrooms_within_500m'].mean():.2f}
      Max: {df_temp['bathrooms_within_500m'].max()}
        """
    
    if 'underserved' in df_temp.columns:
        underserved_pct = (df_temp['underserved'].sum() / len(df_temp)) * 100
        stats_text += f"""
    Underserved Areas: {underserved_pct:.1f}% of incidents
        """
    
    axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes,
                   fontsize=11, verticalalignment='top', family='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return fig


def plot_correlation_heatmap(df, title='Feature Correlation Heatmap', save_path=None):
    """Create a correlation heatmap of numerical features."""
    # Select numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Exclude cluster labels and identifiers
    exclude_cols = ['kmeans_cluster', 'dbscan_cluster', 'supervisor_district']
    numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
    
    if len(numerical_cols) < 2:
        return None
    
    correlation = df[numerical_cols].corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, 
                cbar_kws={'label': 'Correlation'}, ax=ax,
                vmin=-1, vmax=1)
    
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return fig


def generate_all_visualizations(needle_df, encampment_df, bathroom_df):
    output_dir = Path("outputs/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("GENERATING COMPREHENSIVE VISUALIZATIONS")
    print("="*60)
    
    # ==================== SPATIAL VISUALIZATIONS ====================
    print("\n[1/4] Generating needle case visualizations...")
    plot_spatial_distribution(needle_df, 'Needle Cases Distribution', output_dir / 'needles_distribution.png')
    plot_cluster_scatter(needle_df, 'kmeans_cluster', 'Needle Cases - K-Means Clusters', output_dir / 'needles_kmeans.png')
    plot_cluster_scatter(needle_df, 'dbscan_cluster', 'Needle Cases - DBSCAN Clusters', output_dir / 'needles_dbscan.png')
    plot_distance_histogram(needle_df, 'dist_to_bathroom_m', 'Needle Distance to Bathrooms', output_dir / 'needles_distance_hist.png')
    plot_cluster_bar_chart(needle_df, 'kmeans_cluster', 'Needle K-Means Cluster Sizes', output_dir / 'needles_cluster_sizes.png')
    
    print("[2/4] Generating encampment visualizations...")
    plot_spatial_distribution(encampment_df, 'Homeless Encampments Distribution', output_dir / 'encampments_distribution.png')
    plot_cluster_scatter(encampment_df, 'kmeans_cluster', 'Encampments - K-Means Clusters', output_dir / 'encampments_kmeans.png')
    plot_distance_histogram(encampment_df, 'dist_to_bathroom_m', 'Encampment Distance to Bathrooms', output_dir / 'encampments_distance_hist.png')
    
    print("[3/4] Generating bathroom visualizations...")
    plot_spatial_distribution(bathroom_df, 'Public Bathrooms Distribution', output_dir / 'bathrooms_distribution.png')
    plot_cluster_scatter(bathroom_df, 'kmeans_cluster', 'Bathrooms - K-Means Clusters', output_dir / 'bathrooms_kmeans.png')
    
    # ==================== TEMPORAL VISUALIZATIONS ====================
    print("\n[4/4] Generating temporal visualizations...")
    
    # Needle cases temporal analysis
    if 'opened' in needle_df.columns:
        print("  - Needle cases temporal analysis...")
        plot_time_series(needle_df, 'opened', 'Needle Cases - Daily Time Series', 
                        output_dir / 'needles_timeseries_daily.png', freq='D')
        plot_time_series(needle_df, 'opened', 'Needle Cases - Weekly Time Series', 
                        output_dir / 'needles_timeseries_weekly.png', freq='W')
        plot_monthly_trends(needle_df, 'opened', 'Needle Cases - Monthly Trends', 
                           output_dir / 'needles_monthly_trends.png')
        plot_day_of_week_pattern(needle_df, 'opened', 'Needle Cases - Day of Week Pattern', 
                                output_dir / 'needles_day_of_week.png')
        plot_hour_of_day_pattern(needle_df, 'opened', 'Needle Cases - Hour of Day Pattern', 
                                output_dir / 'needles_hour_of_day.png')
        plot_seasonal_pattern(needle_df, 'opened', 'Needle Cases - Seasonal Pattern', 
                             output_dir / 'needles_seasonal.png')
        plot_calendar_heatmap(needle_df, 'opened', 'Needle Cases - Calendar Heatmap', 
                             output_dir / 'needles_calendar_heatmap.png')
        plot_yearly_comparison(needle_df, 'opened', 'Needle Cases - Year-over-Year', 
                              output_dir / 'needles_yearly_comparison.png')
        
        if 'neighborhood' in needle_df.columns:
            plot_temporal_neighborhood_comparison(needle_df, 'opened', 'neighborhood',
                                                 'Needle Cases - Neighborhood Trends', 
                                                 output_dir / 'needles_neighborhood_temporal.png')
        
        # Interactive temporal plot
        create_interactive_temporal_plot(needle_df, 'opened', 
                                        'Needle Cases - Interactive Time Series',
                                        output_dir / 'needles_interactive_temporal.html')
        
        # Status comparison (Open vs Closed)
        if 'status' in needle_df.columns:
            plot_status_comparison(needle_df, 'opened', 'status',
                                  'Needle Cases - Status Over Time',
                                  output_dir / 'needles_status_comparison.png')
        
        # Spatial-temporal statistics dashboard
        plot_spatial_temporal_stats(needle_df, 'opened', 
                                   'Needle Cases - Statistics Dashboard',
                                   output_dir / 'needles_stats_dashboard.png')
    
    # Encampment temporal analysis
    if 'observed_month' in encampment_df.columns:
        print("  - Encampment temporal analysis...")
        plot_time_series(encampment_df, 'observed_month', 
                        'Encampments - Time Series', 
                        output_dir / 'encampments_timeseries.png', freq='M')
        plot_monthly_trends(encampment_df, 'observed_month', 
                           'Encampments - Monthly Trends', 
                           output_dir / 'encampments_monthly_trends.png')
        plot_seasonal_pattern(encampment_df, 'observed_month', 
                             'Encampments - Seasonal Pattern', 
                             output_dir / 'encampments_seasonal.png')
        plot_yearly_comparison(encampment_df, 'observed_month', 
                              'Encampments - Year-over-Year', 
                              output_dir / 'encampments_yearly_comparison.png')
        
        if 'sf_find_neighborhood' in encampment_df.columns:
            plot_temporal_neighborhood_comparison(encampment_df, 'observed_month', 
                                                 'sf_find_neighborhood',
                                                 'Encampments - Neighborhood Trends', 
                                                 output_dir / 'encampments_neighborhood_temporal.png')
        
        # Spatial-temporal statistics dashboard
        plot_spatial_temporal_stats(encampment_df, 'observed_month', 
                                   'Encampments - Statistics Dashboard',
                                   output_dir / 'encampments_stats_dashboard.png')
    
    # ==================== CORRELATION ANALYSIS ====================
    print("\nGenerating correlation heatmaps...")
    plot_correlation_heatmap(needle_df, 'Needle Cases - Feature Correlations',
                            output_dir / 'needles_correlation.png')
    plot_correlation_heatmap(encampment_df, 'Encampments - Feature Correlations',
                            output_dir / 'encampments_correlation.png')
    plot_correlation_heatmap(bathroom_df, 'Bathrooms - Feature Correlations',
                            output_dir / 'bathrooms_correlation.png')
    
    # ==================== INTERACTIVE MAPS ==
    print("\nGenerating interactive maps...")
    create_folium_heatmap(needle_df, 'Needle Heatmap', output_dir / 'needles_heatmap.html')
    create_folium_cluster_map(needle_df, 'kmeans_cluster', 'Needle Clusters', 
                              output_dir / 'needles_cluster_map.html')
    create_multi_layer_map(needle_df, encampment_df, bathroom_df, 
                          output_dir / 'combined_map.html')
    
    # Spatio-temporal maps
    if 'opened' in needle_df.columns:
        print("  - Creating needle cases spatio-temporal map (monthly)...")
        create_spatiotemporal_map(needle_df, 'opened', 
                                 'Needle Cases - Evolution Over Time',
                                 output_dir / 'needles_spatiotemporal.html',
                                 time_period='month', max_points=5000)
    
    if 'observed_month' in encampment_df.columns:
        # Check if there's sufficient temporal variation
        encampment_dates = pd.to_datetime(encampment_df['observed_month'], errors='coerce')
        unique_periods = encampment_dates.dropna().dt.to_period('Q').nunique()
        
        if unique_periods >= 8:  # Need at least 8 quarters (~2 years) for meaningful timeline
            print("  - Creating encampments spatio-temporal map (quarterly)...")
            create_spatiotemporal_map(encampment_df, 'observed_month',
                                     'Encampments - Evolution Over Time',
                                     output_dir / 'encampments_spatiotemporal.html',
                                     time_period='quarter', max_points=800)
        else:
            print(f"  - Skipping encampments spatio-temporal map (only {unique_periods} time periods, need 8+ for meaningful timeline)")
    
    print("\n" + "="*60)
    print(f"✓ All visualizations saved to: {output_dir.absolute()}")
    print("="*60)
    print(f"\nGenerated:")
    print(f"  - {len(list(output_dir.glob('*.png')))} static visualizations (PNG)")
    print(f"  - {len(list(output_dir.glob('*.html')))} interactive visualizations (HTML)")
    print("="*60 + "\n")
