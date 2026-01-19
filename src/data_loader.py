import pandas as pd
from pathlib import Path
from typing import Tuple


def load_needle_cases(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    return df


def load_homeless_counts(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    return df


def load_public_bathrooms(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    return df


def load_all_datasets(data_dir: str = "CSVdata") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base_path = Path(data_dir)
    
    needle_cases = load_needle_cases(base_path / "311_Cases__Needle-related_cases_after_January_1,_2017_20260117.csv")
    
    homeless_counts = load_homeless_counts(base_path / "Quarterly_count_of_tents,_structures,_and_lived-in_vehicles_20260117.csv")
    
    bathrooms = load_public_bathrooms(base_path / "San_Francisco_Public_Bathrooms_and_Water_Fountains_20260117.csv")
    
    return needle_cases, homeless_counts, bathrooms
