from src.data_loader import load_all_datasets
from src.helper_functions import print_dataset_info

def main():
    print("Load the datasets")
    needle_cases, homeless_encampments, bathrooms = load_all_datasets()

    print_dataset_info(needle_cases, 'needle_cases')
    print_dataset_info(homeless_encampments, 'homeless_encampments')
    print_dataset_info(bathrooms, 'bathrooms')

if __name__ == "__main__":
    main()