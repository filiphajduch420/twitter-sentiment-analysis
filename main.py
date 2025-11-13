from src.data_loader import load_and_process_data


def run_project():
    print("Spouštím analýzu...")

    # Krok 1-3: Načtení dat. 'all_data' je teď SLOVNÍK.
    # Např: {'Donald Trump': DataFrame, 'Ted Cruz': DataFrame, ...}
    all_data = load_and_process_data('data/Sentiment.csv')

    if not all_data:
        print("Chyba při načítání dat, končím.")
        return

    # Krok 4-5: Analýza pro KAŽDÉHO kandidáta
    print("\n--- Zahajuji analýzu pro jednotlivé kandidáty ---")

    # Projedu slovník kandidát po kandidátovi
    for candidate_name, candidate_df in all_data.items():
        print(f"\n===== Analýza pro: {candidate_name} =====")

    print("\nAnalýza dokončena pro všechny kandidáty.")


if __name__ == "__main__":
    run_project()