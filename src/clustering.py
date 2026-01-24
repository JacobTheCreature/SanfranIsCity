import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler


def prepare_clustering_features(df, feature_cols):
    return df[feature_cols].values


def kmeans_clustering(data, n_clusters=5, random_state=42):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(data_scaled)
    return labels, kmeans, scaler


def dbscan_clustering(data, eps=0.5, min_samples=10):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(data_scaled)
    return labels, dbscan, scaler


def add_cluster_labels(df, labels, label_col='cluster'):
    df = df.copy()
    df[label_col] = labels
    return df


def get_cluster_stats(df, cluster_col='cluster'):
    stats = df.groupby(cluster_col).agg({
        'latitude': ['mean', 'count'],
        'longitude': 'mean'
    })
    stats.columns = ['lat_center', 'count', 'lon_center']
    stats = stats.reset_index()
    return stats
