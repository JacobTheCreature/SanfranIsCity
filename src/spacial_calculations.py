import geopandas as gpd
from shapely.geometry import Point
from scipy.spatial import cKDTree
import numpy as np


def create_geodataframe(df, lat_col='latitude', lon_col='longitude'):
    geometry = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
    return gdf


def prepare_geodataframes(needle_cases_clean, homeless_counts_clean, bathrooms_clean):
    needles_gdf = create_geodataframe(needle_cases_clean)
    encampments_gdf = create_geodataframe(homeless_counts_clean)
    bathrooms_gdf = create_geodataframe(bathrooms_clean)
    return needles_gdf, encampments_gdf, bathrooms_gdf


def find_nearest_facility(target_gdf, facility_gdf):
    tree = cKDTree(np.array(list(facility_gdf.geometry.apply(lambda x: (x.x, x.y)))))
    distances, indices = tree.query(np.array(list(target_gdf.geometry.apply(lambda x: (x.x, x.y)))))
    distances_meters = distances * 111000 * np.cos(np.radians(37.77))
    return distances_meters, indices


def count_facilities_within_radius(target_gdf, facility_gdf, radius=500):
    counts = []
    for target_point in target_gdf.geometry:
        distances = facility_gdf.geometry.apply(lambda facility: target_point.distance(facility) * 111000 * np.cos(np.radians(37.77)))
        count = (distances <= radius).sum()
        counts.append(count)
    return counts


def integrate_spatial_data(needles_gdf, encampments_gdf, bathrooms_gdf):
    # Distances to nearest facilities
    needles_gdf['dist_to_bathroom_m'], _ = find_nearest_facility(needles_gdf, bathrooms_gdf)
    needles_gdf['dist_to_encampment_m'], _ = find_nearest_facility(needles_gdf, encampments_gdf)
    encampments_gdf['dist_to_bathroom_m'], _ = find_nearest_facility(encampments_gdf, bathrooms_gdf)
    
    # Facility counts within 500m radius
    needles_gdf['bathrooms_within_500m'] = count_facilities_within_radius(needles_gdf, bathrooms_gdf, 500)
    needles_gdf['encampments_within_500m'] = count_facilities_within_radius(needles_gdf, encampments_gdf, 500)
    encampments_gdf['bathrooms_within_500m'] = count_facilities_within_radius(encampments_gdf, bathrooms_gdf, 500)
    encampments_gdf['needles_within_500m'] = count_facilities_within_radius(encampments_gdf, needles_gdf, 500)
    bathrooms_gdf['needles_within_500m'] = count_facilities_within_radius(bathrooms_gdf, needles_gdf, 500)
    bathrooms_gdf['encampments_within_500m'] = count_facilities_within_radius(bathrooms_gdf, encampments_gdf, 500)
    
    # Identify underserved points (>800m from bathrooms)
    needles_gdf['underserved'] = needles_gdf['dist_to_bathroom_m'] > 800
    encampments_gdf['underserved'] = encampments_gdf['dist_to_bathroom_m'] > 800
    
    return needles_gdf, encampments_gdf, bathrooms_gdf