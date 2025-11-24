# src/analysis.py

import pandas as pd
from src.preprocessing import preprocess_text  # Importujeme tvůj hotový čistič

# Importy pro VADER a NLTK analýzu
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.probability import FreqDist
from nltk import Text

# Inicializace VADERu (stačí jednou, když se soubor načte)
# Předpokládá, že 'vader_lexicon' je stažený (což řeší preprocessing.py)
try:
    sia = SentimentIntensityAnalyzer()
except LookupError:
    print("CHYBA: VADER lexicon nebyl nalezen.")
    print("Prosím, ujisti se, že 'nltk.download(\"vader_lexicon\")' je v src/preprocessing.py.")
    # Pokud VADER není, skript spadne, což je v pořádku,
    # protože bez něj nemůže analýza běžet.


def analyze_candidate_topics(candidate_df: pd.DataFrame, candidate_name: str):
    """
    Provede kompletní Krok 4 a 5 pro jeden dataframe kandidáta.
    Tuto funkci bude volat main.py.

    Vrací:
        Tuple (int, int, int): (počet pozitivních, počet negativních, počet neutrálních)
    """
    print(f"\n=== OBECNÁ ANALÝZA (všechny tweety) pro: {candidate_name} ===")
    all_tokens = []
    for text in candidate_df['text']:
        all_tokens.extend(preprocess_text(text))

    _run_full_nltk_analysis(all_tokens)
    print("=======================================================\n")

    positive_raw_tweets = []
    negative_raw_tweets = []
    neutral_raw_tweets = []  # Potřebujeme pro Graf 1

    print(f"Running VADER sentiment analysis on {len(candidate_df)} tweets for {candidate_name}...")

    # Krok 5: Rozdělení tweetů podle sentimentu
    for text in candidate_df['text']:
        # VADER spouštíme na surovém textu
        score = sia.polarity_scores(text)['compound']

        if score > 0.05:
            positive_raw_tweets.append(text)
        elif score < -0.05:
            negative_raw_tweets.append(text)
        else:
            neutral_raw_tweets.append(text)

    print(
        f"Found {len(positive_raw_tweets)} positive, {len(negative_raw_tweets)} negative, and {len(neutral_raw_tweets)} neutral tweets.")

    # --- Krok 4: Analýza témat ---

    # Část A: Analýza POZITIVNÍCH témat (čemu se věnovat)
    print("\n--- Analýza POZITIVNÍCH témat (čemu se věnovat) ---")

    # 1. Vyčistíme jen pozitivní tweety
    positive_tokens = []
    for text in positive_raw_tweets:
        positive_tokens.extend(preprocess_text(text))  # Použijeme importovanou funkci

    # 2. Spustíme kompletní NLTK analýzu (všechny 3 body)
    _run_full_nltk_analysis(positive_tokens)

    # Část B: Analýza NEGATIVNÍCH témat (čemu se vyhnout)
    print("\n--- Analýza NEGATIVNÍCH témat (čemu se vyhnout) ---")

    # 1. Vyčistíme jen negativní tweety
    negative_tokens = []
    for text in negative_raw_tweets:
        negative_tokens.extend(preprocess_text(text))

    # 2. Spustíme kompletní NLTK analýzu (všechny 3 body)
    _run_full_nltk_analysis(negative_tokens)

    # Konec analýzy pro tohoto kandidáta

    # Vrátíme počty, které si main.py převezme pro reporting
    return len(positive_raw_tweets), len(negative_raw_tweets), len(neutral_raw_tweets)


def _run_full_nltk_analysis(tokens: list, num_topics=10):
    """
    Privátní/pomocná funkce, která provede všechny 3 NLTK analýzy.
    (Frekvence, Kolokace, Shody)
    """
    if not tokens:
        print("No relevant tokens found to analyze (empty list).")
        return

    # KROK 1: FREKVENČNÍ CHARAKTERISTIKY
    print(f"\nTop {num_topics} témat (Frekvence):")
    try:
        fdist = FreqDist(tokens)
        top_topics = fdist.most_common(num_topics)
        print(top_topics)
    except Exception as e:
        print(f"Error calculating FreqDist: {e}")
        return  # Pokud selže FreqDist, nemá smysl pokračovat

    # Vytvoříme NLTK Text objekt, který potřebujeme pro Krok 2 a 3
    text_obj = Text(tokens)

    # KROK 2: KOLOKACE (FRÁZE)
    print("\nČasté fráze (Kolokace):")
    try:
        text_obj.collocations(num=5)  # .collocations() si to vypíše samo
    except Exception as e:
        print(f"Error calculating Collocations: {e}")

    # KROK 3: SHODY (CONCORDANCE)
    print("\nKontext pro top 3 témata (Shody):")
    if not top_topics:
        print("Žádná top témata k zobrazení kontextu.")
        return

    try:
        # Projedeme první 3 slova z 'top_topics' a ukážeme jejich kontext
        # top_topics je seznam [('great', 50), ('better', 30), ...]
        for topic_tuple in top_topics[:3]:
            word = topic_tuple[0]  # vezmeme jen to slovo, např. 'great'
            print(f"--- Kontext pro slovo: '{word}' ---")
            # Vypíšeme max 5 řádků kontextu, ať to není moc dlouhé
            text_obj.concordance(word, lines=5)
            print("---")  # Oddělovač
    except Exception as e:
        print(f"Error calculating Concordance: {e}")