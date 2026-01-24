from pathlib import Path
from src.data_loader import load_all_datasets
from src.helper_functions import print_dataset_info
from src.preprocessing import preprocess_bathrooms, preprocess_homeless_counts, preprocess_needle_cases
from src.spacial_calculations import integrate_spatial_data, prepare_geodataframes
import pandas as pd
from src.clustering import prepare_clustering_features, kmeans_clustering, dbscan_clustering, add_cluster_labels, get_cluster_stats

def main():
    print("Load the datasets")
    needle_cases, homeless_encampments, bathrooms = load_all_datasets()

    print_dataset_info(needle_cases, 'needle_cases')
    print_dataset_info(homeless_encampments, 'homeless_encampments')
    print_dataset_info(bathrooms, 'bathrooms')


    # Preprocessing
    needle_cases_clean = preprocess_needle_cases(needle_cases)
    homeless_encampments_clean = preprocess_homeless_counts(homeless_encampments)
    bathrooms_clean = preprocess_bathrooms(bathrooms)

    print_dataset_info(needle_cases_clean, 'needle_cases clean')
    print_dataset_info(homeless_encampments_clean, 'homeless_encampments clean')
    print_dataset_info(bathrooms_clean, 'bathrooms clean')

    # Check if spatial CSV files exist
    processed_dir = Path("CSVdata/processed")
    spacial_needle_csv = processed_dir / "needle_cases_spatial.csv"
    spacial_encampment_csv = processed_dir / "homeless_encampments_spatial.csv"
    spacial_bathroom_csv = processed_dir / "bathrooms_spatial.csv"
    
    if spacial_needle_csv.exists() and spacial_encampment_csv.exists() and spacial_bathroom_csv.exists():
        print("\nSpatial CSV files found. You're all set boss")
    else:
        print("\nSpatial CSV files not found. So lets make them")
        # Spacial integration
        needles_gdf, encampments_gdf, bathrooms_gdf = prepare_geodataframes(needle_cases_clean, homeless_encampments_clean, bathrooms_clean)

        spacial_needle_dataset, spacial_bathroom_dataset, spacial_encampment_dataset = integrate_spatial_data(needles_gdf, encampments_gdf, bathrooms_gdf)

        print_dataset_info(spacial_needle_dataset, 'spacial needles')
        print_dataset_info(spacial_bathroom_dataset, 'spacial bathroom')
        print_dataset_info(spacial_encampment_dataset, 'spacial encampments')
        
        # Save to CSV
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        spacial_needle_dataset.drop(columns=['geometry']).to_csv(spacial_needle_csv, index=False)
        spacial_encampment_dataset.drop(columns=['geometry']).to_csv(spacial_encampment_csv, index=False)
        spacial_bathroom_dataset.drop(columns=['geometry']).to_csv(spacial_bathroom_csv, index=False)

    # Clustering
    
    # Load spatial data
    needle_spatial = pd.read_csv(spacial_needle_csv)
    encampment_spatial = pd.read_csv(spacial_encampment_csv)
    bathroom_spatial = pd.read_csv(spacial_bathroom_csv)
    
    # Clustering needles by geographic + spatial features
    print("Clustering needle cases")
    needle_features = prepare_clustering_features(
        needle_spatial, 
        ['latitude', 'longitude', 'dist_to_bathroom_m', 'dist_to_encampment_m']
    )
    needle_kmeans_labels, _, _ = kmeans_clustering(needle_features, n_clusters=8)
    needle_dbscan_labels, _, _ = dbscan_clustering(needle_features, eps=0.3, min_samples=20)
    
    needle_spatial = add_cluster_labels(needle_spatial, needle_kmeans_labels, 'kmeans_cluster')
    needle_spatial = add_cluster_labels(needle_spatial, needle_dbscan_labels, 'dbscan_cluster')
    
    # Clustering encampments
    print("Clustering homeless encampments")
    encampment_features = prepare_clustering_features(
        encampment_spatial,
        ['latitude', 'longitude', 'dist_to_bathroom_m', 'needles_within_500m']
    )
    encampment_kmeans_labels, _, _ = kmeans_clustering(encampment_features, n_clusters=6)
    encampment_dbscan_labels, _, _ = dbscan_clustering(encampment_features, eps=0.3, min_samples=10)
    
    encampment_spatial = add_cluster_labels(encampment_spatial, encampment_kmeans_labels, 'kmeans_cluster')
    encampment_spatial = add_cluster_labels(encampment_spatial, encampment_dbscan_labels, 'dbscan_cluster')
    
    # Clustering bathrooms
    print("Clustering bathrooms")
    bathroom_features = prepare_clustering_features(
        bathroom_spatial,
        ['latitude', 'longitude', 'needles_within_500m', 'encampments_within_500m']
    )
    bathroom_kmeans_labels, _, _ = kmeans_clustering(bathroom_features, n_clusters=5)
    bathroom_dbscan_labels, _, _ = dbscan_clustering(bathroom_features, eps=0.4, min_samples=5)
    
    bathroom_spatial = add_cluster_labels(bathroom_spatial, bathroom_kmeans_labels, 'kmeans_cluster')
    bathroom_spatial = add_cluster_labels(bathroom_spatial, bathroom_dbscan_labels, 'dbscan_cluster')
    
    # Cluster statistics
    print("\nNeedle K-Means Clusters:")
    print(get_cluster_stats(needle_spatial, 'kmeans_cluster'))
    print("\nNeedle DBSCAN Clusters:")
    print(get_cluster_stats(needle_spatial, 'dbscan_cluster'))
    
    # Save clustered data
    clustered_dir = Path("CSVdata/clustered")
    clustered_dir.mkdir(parents=True, exist_ok=True)
    
    needle_spatial.to_csv(clustered_dir / "needle_cases_clustered.csv", index=False)
    encampment_spatial.to_csv(clustered_dir / "homeless_encampments_clustered.csv", index=False)
    bathroom_spatial.to_csv(clustered_dir / "bathrooms_clustered.csv", index=False)

if __name__ == "__main__":
    main()