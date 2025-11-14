# main.py

import pandas as pd  # <-- PŘIDEJ IMPORT PANDAS
from src.data_loader import load_and_process_data
from src.analysis import analyze_candidate_topics
# KROK 1: Importujeme naši novou funkci z reportingu
from src.reporting import save_sentiment_bar_chart


def run_project():
    print("Starting analysis...")

    all_data = load_and_process_data('data/Sentiment.csv')
    if not all_data:
        print("Error loading data, exiting.")
        return

    print("\n--- Starting analysis for individual candidates ---")

    # KROK 2: Vytvoříme prázdný slovník na sběr dat pro graf
    all_sentiment_counts = {}

    for candidate_name, candidate_df in all_data.items():
        print(f"\n===== Analyzing: {candidate_name} =====")

        # KROK 3: Uložíme si výsledky (počty) z analýzy
        pos, neg, neu = analyze_candidate_topics(candidate_df, candidate_name)

        # Uložíme počty do našeho sběrného slovníku
        all_sentiment_counts[candidate_name] = {
            'positive': pos,
            'negative': neg,
            'neutral': neu
        }

    # KROK 4: Po skončení smyčky máme všechna data. Vytvoříme report.
    print("\n--- Creating analysis reports ---")

    # Převedeme slovník na DataFrame (to je formát, co chce naše funkce)
    sentiment_df = pd.DataFrame.from_dict(all_sentiment_counts, orient='index')

    # Zavoláme funkci z reporting.py a předáme jí data
    save_sentiment_bar_chart(sentiment_df)

    print("\nAnalysis complete for all candidates.")


if __name__ == "__main__":
    run_project()