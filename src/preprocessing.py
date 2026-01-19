import pandas as pd
import numpy as np
from typing import Tuple



# Remove invalid coordinates (0,0 or missing values)
def clean_coordinates(df: pd.DataFrame, lat_col: str = 'Latitude', lon_col: str = 'Longitude') -> pd.DataFrame:
    # Convert to numeric, coercing errors to NaN
    df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
    df[lon_col] = pd.to_numeric(df[lon_col], errors='coerce')
   
    # Remove rows with missing coordinates
    df_clean = df.dropna(subset=[lat_col, lon_col])
   
    # Remove invalid (0,0) coordinates
    df_clean = df_clean[
        (df_clean[lat_col] != 0) &
        (df_clean[lon_col] != 0)
    ]
   
    # San Francisco bounds check (approximately)
    # Latitude: 37.7 to 37.8, Longitude: -122.5 to -122.35
    df_clean = df_clean[
        (df_clean[lat_col] >= 37.7) & (df_clean[lat_col] <= 37.85) &
        (df_clean[lon_col] >= -122.52) & (df_clean[lon_col] <= -122.35)
    ]
   
    return df_clean



# Make all the column names lowercase and with underscores
def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    return df



# Cleaning the actual datasets functions



def preprocess_needle_cases(df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_column_names(df)
    df = clean_coordinates(df, lat_col='latitude', lon_col='longitude')
   
    # Convert date columns to datetime
    date_cols = ['opened', 'closed', 'updated']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
   
    return df



def preprocess_homeless_counts(df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_column_names(df)
    df = clean_coordinates(df, lat_col='latitude', lon_col='longitude')
   
    # Convert date column to datetime
    if 'observed_month' in df.columns:
        df['observed_month'] = pd.to_datetime(df['observed_month'], errors='coerce')
   
    return df



def preprocess_bathrooms(df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_column_names(df)
    df = clean_coordinates(df, lat_col='latitude', lon_col='longitude')
   
    return df
