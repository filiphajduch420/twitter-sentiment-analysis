# main.py

import pandas as pd
from src.data_loader import load_and_process_data
from src.analysis import analyze_candidate_topics
from src.reporting import (
    save_sentiment_bar_chart,
    plot_sentiment_over_time,
    plot_top_words,
    plot_sentiment_distribution,
    plot_sentiment_by_timezone,
    plot_top_positive_candidates_by_timezone,
    plot_sentiment_wordclouds,
    plot_interaction_heatmap
)


def run_project():
    print("Starting analysis...")

    all_data = load_and_process_data('data/Sentiment.csv')
    if not all_data:
        print("Error loading data, exiting.")
        return

    print("\n--- Starting analysis for individual candidates ---")

    all_sentiment_counts = {}

    for candidate_name, candidate_df in all_data.items():
        print(f"\n===== Analyzing: {candidate_name} =====")

        # 1. Textová analýza
        pos, neg, neu = analyze_candidate_topics(candidate_df, candidate_name)

        all_sentiment_counts[candidate_name] = {
            'positive': pos,
            'negative': neg,
            'neutral': neu
        }

        # 2. Generování grafů pro kandidáta
        if len(candidate_df) > 10:
            print(f"   -> Generating graphs for: {candidate_name}...")

            # Základní grafy
            safe_name = candidate_name.replace(' ', '_')

            plot_sentiment_over_time(candidate_df, candidate_name, filepath=f"results/images/{safe_name}_time.png")
            plot_top_words(candidate_df, candidate_name, filepath=f"results/images/{safe_name}_words.png")
            plot_sentiment_distribution(candidate_df, candidate_name, filepath=f"results/images/{safe_name}_dist.png")
            plot_sentiment_by_timezone(candidate_df, candidate_name,
                                       filepath=f"results/images/{safe_name}_timezone.png")

            # (1) NOVÉ: Word Clouds (Pozitivní a Negativní)
            plot_sentiment_wordclouds(candidate_df, candidate_name,
                                      filepath_prefix=f"results/images/{safe_name}_wordcloud")

    print("\n--- Creating summary reports ---")

    # Souhrnný sloupcový graf
    sentiment_df = pd.DataFrame.from_dict(all_sentiment_counts, orient='index')
    save_sentiment_bar_chart(sentiment_df)

    # Srovnání podle časových zón
    plot_top_positive_candidates_by_timezone(all_data)

    # Heatmapa interakcí
    plot_interaction_heatmap(all_data)

    print("\nAnalysis complete. Check 'results/images/' for all graphs.")


if __name__ == "__main__":
    run_project()