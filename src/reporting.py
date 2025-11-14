# src/reporting.py

import pandas as pd
import matplotlib.pyplot as plt
import os  # Potřebujeme pro vytvoření složky


def save_sentiment_bar_chart(sentiment_data: pd.DataFrame,
                             filepath: str = "results/images/sentiment_overview.png"):
    """
    Uloží skládaný sloupcový graf (Graf 1: Celkový přehled sentimentu).

    Očekává DataFrame, kde:
    - index = jména kandidátů
    - sloupce = ['positive', 'negative', 'neutral']
    """
    print(f"Ukládám graf celkového sentimentu do {filepath}...")

    # Ujistím se, že složka /results/images/ existuje
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
    except OSError as e:
        print(f"Chyba při vytváření složky: {e}")
        return

    # Seřadím data, aby největší sloupec byl vlevo (je to přehlednější)
    # Spočítám si celkový součet pro řazení
    sentiment_data['total'] = sentiment_data.sum(axis=1)
    sentiment_data.sort_values(by='total', ascending=False, inplace=True)
    sentiment_data.drop(columns='total', inplace=True)

    # Vytvořím graf
    # 'stacked=True' je to kouzlo, co je dá na sebe
    # 'figsize' je velikost obrázku (šířka, výška v palcích)
    ax = sentiment_data.plot(
        kind='bar',
        stacked=True,
        figsize=(12, 7),
        color=['#2ca02c', '#d62728', '#8c8c8c'],  # Zelená (pozitivní), Červená (negativní), Šedá (neutrální)
        title="Celkový poměr sentimentu pro kandidáty"
    )

    # Popisky
    ax.set_xlabel("Kandidát")
    ax.set_ylabel("Počet tweetů")
    ax.legend(["Pozitivní", "Negativní", "Neutrální"])

    # Otočím popisky na ose X, aby se daly přečíst (když je jich moc)
    plt.xticks(rotation=45, ha='right')

    # Zajistím, že se vše vejde do obrázku
    plt.tight_layout()

    # Uložím soubor
    try:
        plt.savefig(filepath)
        print(f"Graf úspěšně uložen: {filepath}")
    except Exception as e:
        print(f"Nastala chyba při ukládání grafu: {e}")

    # Zavřu "kreslítko", aby se graf nezobrazoval v konzoli
    plt.close()