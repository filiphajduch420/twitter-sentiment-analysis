# src/reporting.py

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import FreqDist
from src.preprocessing import preprocess_text

# Import pro WordCloud (ošetřeno, kdyby chyběl)
try:
    from wordcloud import WordCloud

    HAS_WORDCLOUD = True
except ImportError:
    HAS_WORDCLOUD = False
    print("Warning: 'wordcloud' library not found. WordCloud graphs will be skipped.")
    print("To install: pip install wordcloud")

# Inicializace VADERu
try:
    sia = SentimentIntensityAnalyzer()
except LookupError:
    sia = None
    print("Warning: VADER lexicon not found in reporting.")


def _ensure_dir(filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)


def save_sentiment_bar_chart(sentiment_data: pd.DataFrame,
                             filepath: str = "results/images/sentiment_overview.png"):
    _ensure_dir(filepath)
    sentiment_data['total'] = sentiment_data.sum(axis=1)
    sentiment_data.sort_values(by='total', ascending=False, inplace=True)
    sentiment_data.drop(columns='total', inplace=True)

    ax = sentiment_data.plot(
        kind='bar', stacked=True, figsize=(12, 7),
        color=['#2ca02c', '#d62728', '#8c8c8c'],
        title="Overall Sentiment Ratio"
    )
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    print(f"Summary chart saved: {filepath}")


def plot_sentiment_over_time(df: pd.DataFrame, candidate_name: str, filepath: str):
    _ensure_dir(filepath)
    df_time = df.copy()
    df_time['tweet_created'] = pd.to_datetime(df_time['tweet_created'], errors='coerce')
    df_time.dropna(subset=['tweet_created'], inplace=True)

    if sia:
        df_time['compound'] = df_time['text'].apply(lambda x: sia.polarity_scores(str(x))['compound'])
    else:
        return

    df_time.set_index('tweet_created', inplace=True)
    df_time.sort_index(inplace=True)

    # Resample po 10 minutách pro detailnější křivku
    sentiment_trend = df_time['compound'].resample('10min').mean()

    plt.figure(figsize=(10, 5))
    sentiment_trend.plot(kind='line', marker='o', color='purple', linewidth=2)
    plt.title(f"Sentiment Over Time: {candidate_name}")
    plt.ylabel("Avg Sentiment")
    plt.grid(True, alpha=0.3)
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()


def plot_top_words(df: pd.DataFrame, candidate_name: str, filepath: str, num=15):
    _ensure_dir(filepath)
    all_tokens = []
    for text in df['text']:
        all_tokens.extend(preprocess_text(text))

    if not all_tokens:
        return

    fdist = FreqDist(all_tokens)
    top_words = fdist.most_common(num)

    words = [w[0] for w in top_words]
    counts = [w[1] for w in top_words]
    words.reverse()
    counts.reverse()

    plt.figure(figsize=(10, 8))
    plt.barh(words, counts, color='skyblue', edgecolor='black')
    plt.title(f"Top Words: {candidate_name}")
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()


def plot_sentiment_distribution(df: pd.DataFrame, candidate_name: str, filepath: str):
    _ensure_dir(filepath)
    scores = []
    if sia:
        scores = [sia.polarity_scores(str(t))['compound'] for t in df['text']]

    if not scores:
        return

    plt.figure(figsize=(8, 5))
    plt.hist(scores, bins=20, color='orange', edgecolor='black', alpha=0.7)
    plt.title(f"Polarization: {candidate_name}")
    plt.axvline(0, color='black', linestyle='--', linewidth=1)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()


def plot_sentiment_by_timezone(df: pd.DataFrame, candidate_name: str, filepath: str):
    _ensure_dir(filepath)
    df_zone = df.dropna(subset=['user_timezone']).copy()
    if df_zone.empty:
        return

    top_zones = df_zone['user_timezone'].value_counts().head(5).index
    df_zone = df_zone[df_zone['user_timezone'].isin(top_zones)]

    if sia:
        df_zone['compound'] = df_zone['text'].apply(lambda x: sia.polarity_scores(str(x))['compound'])
    else:
        return

    timezone_sentiment = df_zone.groupby('user_timezone')['compound'].mean().sort_values()

    plt.figure(figsize=(10, 6))
    colors = ['red' if x < 0 else 'green' for x in timezone_sentiment.values]
    timezone_sentiment.plot(kind='barh', color=colors)
    plt.title(f"Sentiment by Timezone: {candidate_name}")
    plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()


def plot_top_positive_candidates_by_timezone(all_data: dict, filepath: str = "results/images/timezone_comparison.png"):
    _ensure_dir(filepath)
    print("Generating timezone comparison chart...")

    df_list = []
    for candidate, df in all_data.items():
        temp_df = df.copy()
        if 'candidate' not in temp_df.columns:
            temp_df['candidate'] = candidate
        df_list.append(temp_df)

    if not df_list: return
    full_df = pd.concat(df_list, ignore_index=True)
    full_df.dropna(subset=['user_timezone'], inplace=True)
    top_zones = full_df['user_timezone'].value_counts().head(5).index.tolist()

    if not sia: return

    full_df['sentiment_label'] = full_df['text'].apply(
        lambda x: 'Positive' if sia.polarity_scores(str(x))['compound'] > 0.05
        else ('Negative' if sia.polarity_scores(str(x))['compound'] < -0.05 else 'Neutral')
    )

    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(24, 6), sharey=False)
    colors = ['#2ca02c', '#8c8c8c', '#d62728']

    for i, zone in enumerate(top_zones):
        ax = axes[i]
        zone_df = full_df[full_df['user_timezone'] == zone]
        counts = pd.crosstab(zone_df['candidate'], zone_df['sentiment_label'])

        for col in ['Positive', 'Neutral', 'Negative']:
            if col not in counts.columns: counts[col] = 0

        counts['Total'] = counts.sum(axis=1)
        counts = counts[counts['Total'] > 10]

        if counts.empty:
            ax.text(0.5, 0.5, "No Data", ha='center')
            continue

        counts['Pos_Pct'] = counts['Positive'] / counts['Total']
        top_cands = counts.sort_values(by='Pos_Pct', ascending=True).tail(5)
        plot_data = top_cands[['Positive', 'Neutral', 'Negative']].div(top_cands['Total'], axis=0)

        plot_data.plot(kind='barh', stacked=True, ax=ax, color=colors, legend=False)
        ax.set_title(f"Zone: {zone}")
        ax.set_xlim(0, 1)
        ax.set_ylabel("")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=3, fontsize=12)
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()
    print(f"Timezone comparison saved: {filepath}")


# --- NOVÉ FUNKCE (1 & 3) ---

def plot_sentiment_wordclouds(df: pd.DataFrame, candidate_name: str, filepath_prefix: str):
    """
    (1) Vygeneruje dva Word Cloud obrázky: Pozitivní a Negativní.
    """
    if not HAS_WORDCLOUD or not sia:
        return

    _ensure_dir(filepath_prefix)

    # Rozdělení textů
    pos_text = []
    neg_text = []

    for text in df['text']:
        score = sia.polarity_scores(str(text))['compound']
        # Předpokládáme, že preprocess_text vrátí list tokenů -> spojíme zpět do stringu
        tokens = preprocess_text(text)
        clean_text = " ".join(tokens)

        if score > 0.05:
            pos_text.append(clean_text)
        elif score < -0.05:
            neg_text.append(clean_text)

    # Funkce pro vykreslení jednoho mraku
    def _save_cloud(text_list, sentiment_name, colormap):
        if not text_list: return
        full_text = " ".join(text_list)
        wc = WordCloud(width=800, height=400, background_color='white', colormap=colormap).generate(full_text)

        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        plt.title(f"{sentiment_name} Word Cloud: {candidate_name}")
        plt.tight_layout()
        path = f"{filepath_prefix}_{sentiment_name.lower()}.png"
        plt.savefig(path)
        plt.close()
        print(f"WordCloud saved: {path}")

    _save_cloud(pos_text, "Positive", "Greens")
    _save_cloud(neg_text, "Negative", "Reds")


def plot_interaction_heatmap(all_data: dict, filepath: str = "results/images/interaction_heatmap.png"):
    """
    (3) Heatmapa: Kdo mluví o kom?
    """
    _ensure_dir(filepath)
    print("Generating interaction heatmap...")

    candidates = list(all_data.keys())
    # Vyhodíme "No candidate mentioned" ze sloupců (koho zmiňují), ale necháme v řádcích (kdo mluví)
    targets = [c for c in candidates if c != "No candidate mentioned"]

    # Matice: řádky = Kdo mluví (Source), sloupce = O kom (Target)
    matrix = pd.DataFrame(0, index=candidates, columns=targets)

    for speaker, df in all_data.items():
        for text in df['text']:
            text_lower = str(text).lower()
            for target in targets:
                # Jednoduchá detekce: Hledáme příjmení
                # Trump -> trump, Jeb Bush -> bush, Ben Carson -> carson
                # Pozor na jména jako "Paul" (Rand Paul), která jsou běžná slova, ale zkusíme to.
                last_name = target.split()[-1].lower()
                if last_name in text_lower:
                    matrix.loc[speaker, target] += 1

    # Normalizace (volitelné) - abychom viděli intenzitu bez ohledu na počet tweetů
    # Ale absolutní čísla jsou pro heatmapu taky fajn. Necháme absolutní.

    # Vykreslení pomocí Matplotlib
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(matrix.values, cmap="YlGnBu")

    # Popisky os
    ax.set_xticks(np.arange(len(targets)))
    ax.set_yticks(np.arange(len(candidates)))
    ax.set_xticklabels(targets, rotation=45, ha="right")
    ax.set_yticklabels(candidates)

    # Zobrazení hodnot v políčkách
    for i in range(len(candidates)):
        for j in range(len(targets)):
            text = ax.text(j, i, matrix.iloc[i, j],
                           ha="center", va="center", color="black", fontsize=8)

    ax.set_title("Candidate Mention Heatmap (Who mentions whom?)")
    fig.tight_layout()
    plt.colorbar(im, ax=ax)
    plt.savefig(filepath)
    plt.close()
    print(f"Heatmap saved: {filepath}")