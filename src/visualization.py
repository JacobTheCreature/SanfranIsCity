import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap, MarkerCluster
from pathlib import Path

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def plot_cluster_scatter(df, cluster_col='kmeans_cluster', title='Cluster Map', save_path=None):
    fig, ax = plt.subplots(figsize=(14, 10))
    
    unique_clusters = sorted(df[cluster_col].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))
    
    for idx, cluster in enumerate(unique_clusters):
        cluster_data = df[df[cluster_col] == cluster]
        label = f'Cluster {cluster}' if cluster != -1 else 'Noise'
        ax.scatter(cluster_data['longitude'], cluster_data['latitude'], 
                  c=[colors[idx]], label=label, alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return fig


def plot_spatial_distribution(df, title='Spatial Distribution', save_path=None):
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.scatter(df['longitude'], df['latitude'], alpha=0.4, s=20, c='crimson', edgecolors='black', linewidth=0.3)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return fig


def plot_distance_histogram(df, col='dist_to_bathroom_m', title='Distance Distribution', save_path=None):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(df[col].dropna(), bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel(f'{col} (meters)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axvline(500, color='red', linestyle='--', linewidth=2, label='500m threshold')
    ax.axvline(800, color='orange', linestyle='--', linewidth=2, label='800m threshold')
    ax.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return fig


def plot_cluster_bar_chart(df, cluster_col='kmeans_cluster', title='Cluster Sizes', save_path=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    cluster_counts = df[cluster_col].value_counts().sort_index()
    cluster_counts.plot(kind='bar', ax=ax, color='teal', edgecolor='black', alpha=0.8)
    ax.set_xlabel('Cluster ID', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
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


def plot_association_rules(rules_df, title='Top Association Rules', save_path=None):
    if len(rules_df) == 0:
        return None
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    top_rules = rules_df.head(15)
    scatter = ax.scatter(top_rules['support'], top_rules['confidence'], 
                        s=top_rules['lift']*100, c=top_rules['lift'], 
                        cmap='viridis', alpha=0.6, edgecolors='black', linewidth=1)
    
    ax.set_xlabel('Support', fontsize=12)
    ax.set_ylabel('Confidence', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Lift', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return fig


def generate_all_visualizations(needle_df, encampment_df, bathroom_df, needle_rules, encampment_rules, bathroom_rules):
    output_dir = Path("outputs/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating needle case visualizations...")
    plot_spatial_distribution(needle_df, 'Needle Cases Distribution', output_dir / 'needles_distribution.png')
    plot_cluster_scatter(needle_df, 'kmeans_cluster', 'Needle Cases - K-Means Clusters', output_dir / 'needles_kmeans.png')
    plot_cluster_scatter(needle_df, 'dbscan_cluster', 'Needle Cases - DBSCAN Clusters', output_dir / 'needles_dbscan.png')
    plot_distance_histogram(needle_df, 'dist_to_bathroom_m', 'Needle Distance to Bathrooms', output_dir / 'needles_distance_hist.png')
    plot_cluster_bar_chart(needle_df, 'kmeans_cluster', 'Needle K-Means Cluster Sizes', output_dir / 'needles_cluster_sizes.png')
    
    print("Generating encampment visualizations...")
    plot_spatial_distribution(encampment_df, 'Homeless Encampments Distribution', output_dir / 'encampments_distribution.png')
    plot_cluster_scatter(encampment_df, 'kmeans_cluster', 'Encampments - K-Means Clusters', output_dir / 'encampments_kmeans.png')
    plot_distance_histogram(encampment_df, 'dist_to_bathroom_m', 'Encampment Distance to Bathrooms', output_dir / 'encampments_distance_hist.png')
    
    print("Generating bathroom visualizations...")
    plot_spatial_distribution(bathroom_df, 'Public Bathrooms Distribution', output_dir / 'bathrooms_distribution.png')
    plot_cluster_scatter(bathroom_df, 'kmeans_cluster', 'Bathrooms - K-Means Clusters', output_dir / 'bathrooms_kmeans.png')
    
    print("Generating association rule visualizations...")
    if len(needle_rules) > 0:
        plot_association_rules(needle_rules, 'Needle Case Association Rules', output_dir / 'needles_rules.png')
    if len(encampment_rules) > 0:
        plot_association_rules(encampment_rules, 'Encampment Association Rules', output_dir / 'encampments_rules.png')
    if len(bathroom_rules) > 0:
        plot_association_rules(bathroom_rules, 'Bathroom Association Rules', output_dir / 'bathrooms_rules.png')
    
    print("Generating interactive maps...")
    create_folium_heatmap(needle_df, 'Needle Heatmap', output_dir / 'needles_heatmap.html')
    create_folium_cluster_map(needle_df, 'kmeans_cluster', 'Needle Clusters', output_dir / 'needles_cluster_map.html')
    create_multi_layer_map(needle_df, encampment_df, bathroom_df, output_dir / 'combined_map.html')
    
    print(f"\nAll visualizations saved to {output_dir}")
