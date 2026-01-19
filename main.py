from src.data_loader import load_all_datasets
from src.helper_functions import print_dataset_info
from src.preprocessing import preprocess_bathrooms, preprocess_homeless_counts, preprocess_needle_cases

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


if __name__ == "__main__":
    main()