# src/data_loader.py

import pandas as pd
from typing import Dict

# Definujeme, že funkce vrací slovník
DataFrameDict = Dict[str, pd.DataFrame]


def load_data(filepath: str) -> pd.DataFrame:
    """Krok 1: Načte data z CSVčka."""
    print(f"Loading file: {filepath}...")
    try:
        df = pd.read_csv(filepath, encoding='latin-1')
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found :(")
        return pd.DataFrame()

    print("Loaded successfully.")
    return df


def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    """Krok 2: Vyhodím zbytečný sloupce a řádky, kde nic není."""
    print("Filtering data...")


    columns_to_keep = ['candidate', 'text', 'tweet_created', 'user_timezone']


    available_cols = [c for c in columns_to_keep if c in df.columns]

    if 'candidate' not in available_cols or 'text' not in available_cols:
        print(f"Error: Missing required columns 'candidate' or 'text'!")
        return pd.DataFrame()

    df_filtered = df[available_cols].copy()

    # Vyhodím řádky, kde chybí text nebo kandidát (NaN)
    df_filtered.dropna(subset=['candidate', 'text'], inplace=True)

    print("Data filtered.")
    return df_filtered


def split_by_all_candidates(df: pd.DataFrame) -> DataFrameDict:
    """
    Krok 3: Rozdělí data na slovník, kde klíč je jméno kandidáta.
    """
    print("Splitting data by ALL candidates...")

    # Slovník pro ukládání [jméno_kandidáta] -> [jeho_dataframe]
    candidate_dataframes = {}

    all_candidates = df['candidate'].unique()

    print(f"Found {len(all_candidates)} unique candidate categories.")

    for candidate_name in all_candidates:
        # Vytvořím dataframe jen pro tohoto kandidáta
        df_candidate = df[df['candidate'] == candidate_name].copy()

        # Uložím ho do slovníku
        candidate_dataframes[candidate_name] = df_candidate

        print(f"  > {candidate_name}: {len(df_candidate)} tweets")

    return candidate_dataframes


def load_and_process_data(filepath: str = 'data/Sentiment.csv') -> DataFrameDict:
    """
    Hlavní funkce, co zavolá ty ostatní popořadě.
    """
    # Krok 1
    df = load_data(filepath)
    if df.empty:
        return {}

    # Krok 2
    df_filtered = filter_data(df)
    if df_filtered.empty:
        return {}

    # Krok 3
    return split_by_all_candidates(df_filtered)


# ---- Kód pro testování ----
if __name__ == "__main__":
    print("--- Testing data_loader.py (dict version) ---")

    TEST_PATH = '../data/Sentiment.csv'

    # Teď je to jedna proměnná (slovník)
    all_data = load_and_process_data(TEST_PATH)

    print("\n--- Test output ---")
    if all_data:
        print(f"Number of candidates to analyze: {len(all_data)}")

        # Zkusím vytisknout data pro Trumpa ze slovníku
        if 'Donald Trump' in all_data:
            print("\nFirst 5 rows for Trump (for checking):")
            print(all_data['Donald Trump'].head())
        else:
            print("Donald Trump is not in the dictionary (which is weird).")
    else:
        print("Something went wrong, data is empty.")

    print("--- Test complete ---")