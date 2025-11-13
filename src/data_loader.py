# src/data_loader.py

import pandas as pd
from typing import Dict  # Měníme Tuple na Dict (slovník)

# Definujeme, že funkce vrací slovník
# Klíč bude string (jméno kandidáta) a hodnota bude DataFrame
DataFrameDict = Dict[str, pd.DataFrame]


def load_data(filepath: str) -> pd.DataFrame:
    """Krok 1: Načte data z CSVčka."""
    print(f"Načítám soubor: {filepath}...")
    try:
        df = pd.read_csv(filepath, encoding='latin-1')
    except FileNotFoundError:
        print(f"Error: Soubor '{filepath}' jsem nenašel :(")
        return pd.DataFrame()

    print("Načteno, v pohodě.")
    return df


def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    """Krok 2: Vyhodím zbytečný sloupce a řádky, kde nic není."""
    print("Filtruju data...")
    columns_to_keep = ['candidate', 'text']

    if not all(col in df.columns for col in columns_to_keep):
        print(f"Chyba: Chybí mi sloupce 'candidate' nebo 'text'!")
        return pd.DataFrame()

    df_filtered = df[columns_to_keep].copy()

    # Vyhodím řádky, kde chybí text nebo kandidát (NaN)
    df_filtered.dropna(subset=columns_to_keep, inplace=True)

    print("Data vyfiltrována.")
    return df_filtered


def split_by_all_candidates(df: pd.DataFrame) -> DataFrameDict:
    """
    Krok 3: Rozdělí data na slovník, kde klíč je jméno kandidáta.
    """
    print("Rozděluji data podle VŠECH kandidátů...")

    # Slovník pro ukládání [jméno_kandidáta] -> [jeho_dataframe]
    candidate_dataframes = {}

    # Získám seznam všech unikátních kandidátů (už jsme vyfiltrovali 'nan')
    all_candidates = df['candidate'].unique()

    print(f"Nalezeno {len(all_candidates)} unikátních kategorií kandidátů.")

    for candidate_name in all_candidates:
        # Vytvořím dataframe jen pro tohoto kandidáta
        df_candidate = df[df['candidate'] == candidate_name].copy()

        # Uložím ho do slovníku
        candidate_dataframes[candidate_name] = df_candidate

        print(f"  > {candidate_name}: {len(df_candidate)} tweetů")

    return candidate_dataframes


def load_and_process_data(filepath: str = 'data/Sentiment.csv') -> DataFrameDict:
    """
    Hlavní funkce, co zavolá ty ostatní popořadě.
    Tohle pak importuju do main.py.
    VRACÍ SLOVNÍK!
    """
    # Krok 1
    df = load_data(filepath)
    if df.empty:
        return {}  # Vrátí prázdný slovník

    # Krok 2
    df_filtered = filter_data(df)
    if df_filtered.empty:
        return {}

    # Krok 3
    return split_by_all_candidates(df_filtered)


# ---- Kód pro testování ----
if __name__ == "__main__":
    print("--- Testuju data_loader.py (verze se slovníkem) ---")

    TEST_PATH = '../data/Sentiment.csv'

    # Teď je to jedna proměnná (slovník)
    all_data = load_and_process_data(TEST_PATH)

    print("\n--- Testovací výpis ---")
    if all_data:
        print(f"Počet kandidátů k analýze: {len(all_data)}")

        # Zkusím vytisknout data pro Trumpa ze slovníku
        if 'Donald Trump' in all_data:
            print("\nPrvních 5 řádků Trump (pro kontrolu):")
            print(all_data['Donald Trump'].head())
        else:
            print("Donald Trump ve slovníku není (což je divné).")
    else:
        print("Něco se pokazilo, data jsou prázdná.")

    print("--- Test hotov ---")