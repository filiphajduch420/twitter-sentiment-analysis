# src/preprocessing.py

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string  # Pro odstranění interpunkce
import ssl  # <-- PŘIDÁNO: Pro opravu SSL chyby


def download_nltk_data():
    """
    Stáhne potřebný data pro NLTK (stačí spustit jednou).
    OBSAHUJE FIX PRO SSL CHYBU.
    """
    print("Downloading NLTK data (stopwords, punkt)...")

    # FIX pro SSL chybu - nastavíme SSL kontext PŘED stahováním
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        # Starší Python, který SSL neověřuje
        pass
    else:
        # Nastavíme neověřený kontext jako výchozí
        ssl._create_default_https_context = _create_unverified_https_context

    # Teď stáhneme data
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('vader_lexicon')
    print("NLTK data downloaded successfully.")


# --- Stáhneme data HNED ---
# Teď už by to mělo projít díky tomu SSL fixu
download_nltk_data()

# --- Tohle se načte jen jednou při startu skriptu ---

# 1. Načtu si anglický "stopwords"
stop_words = set(stopwords.words('english'))

# 2. Přidám vlastní "smetí", co nechci v analýze
custom_stop_words = [
    'rt', 'gopdebate', 'gop', 'debate', 'amp', 'http', 'https', 'co', 'realdonaldtrump',
    # Jména kandidátů
    'donald', 'trump', 'ted', 'cruz', 'ben', 'carson', 'scott', 'walker',
    'jeb', 'bush', 'marco', 'rubio', 'mike', 'huckabee', 'chris', 'christie',
    'rand', 'paul', 'john', 'kasich'
]
stop_words.update(custom_stop_words)

# 3. Připravím si sadu interpunkce k odstranění
punctuation = set(string.punctuation)
punctuation.add("''")
punctuation.add("...")
punctuation.add("``")
punctuation.add("’")
punctuation.add("‘")


# ---------------------------------------------------


def preprocess_text(text: str) -> list:
    """
    Krok 4 (část 1): Vyčistí jeden textový řetězec.
    Vrátí seznam čistých tokenů (slov).
    """
    if not isinstance(text, str):
        return []

        # 1. Převedu na malý písmena
    text_lower = text.lower()

    # 2. Tokenizace (rozseká větu na slova)
    try:
        tokens = word_tokenize(text_lower)
    except Exception:
        return []

    # 3. Čištění


    cleaned_tokens = []
    for token in tokens:
        if token in punctuation:
            continue
        if token in stop_words:
            continue
        if token.isalpha() and len(token) > 2:
            cleaned_tokens.append(token)

    return cleaned_tokens

# ---- Kód pro testování ----
if __name__ == "__main__":
    print("--- Testing preprocessing.py ---")

    # Stahování už proběhlo nahoře

    test_tweet = "RT @User1: Donald Trump was GREAT in the #GOPDebate! So much better than Jeb Bush... http://t.co.Network/abc"

    print(f"\nOriginal text: {test_tweet}")
    cleaned = preprocess_text(test_tweet)
    print(f"Cleaned tokens: {cleaned}")

    print("\n--- Test complete ---")