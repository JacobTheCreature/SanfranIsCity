import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union

def print_dataset_info(df: pd.DataFrame, name: str = "Dataset") -> None:
    print(f"\n{'='*50}")
    print(f"{name} Information")
    print(f"{'='*50}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nMissing values:")
    print(df.isnull().sum())
    print(f"{'='*50}\n")

