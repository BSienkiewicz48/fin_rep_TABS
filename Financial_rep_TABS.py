import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import openai
import numpy as np
import plotly.graph_objects as go
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import requests
import datetime

# Pobierz wartość zmiennej środowiskowej API_KEYsd


def get_ticker_for_company(company_name):
    prompt = f"Napisz mi TYLKO, ticker yahoo finance firmy: {company_name}, pamiętaj o końcówkach do giełd, chodzi o to że akcje notowane na przykład na giełdzie w warszawie mogą mieć .WA jak np PKN.WA. Mogą mieć ale nie muszą."
    
    # Utwórz instancję klienta OpenAI
    client = openai.OpenAI(api_key=api_key)
    
    # Użyj klienta do stworzenia zapytania
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Only response, short as possible, no dot on end"},
            {"role": "user", "content": prompt}
        ]
    )
    
    ticker = response.choices[0].message.content.strip()
    return ticker

# Definiowanie funkcji do analizowania rekomendacji analityków na podstawie tabeli
def summarize_recommendations(df):
    # Konwertuj tabelę do stringa
    table_string = df.to_string(index=True)
    
    # Stwórz prompt z tabelą
    prompt = f"""Przeanalizuj poniższą tabelę, która przedstawia  procentowy udział poszczególnych rekomendacji w łącznej liczbie rekomendacji analityków na przestrzeni ostatnich miesięcy. Określ, czy widoczna jest tendencja wzrostowa lub spadkowa w liczbie rekomendacji dotyczących kupna i sprzedaży, z naciskiem na „Silne rekomendacje kupna” oraz „Silne rekomendacje sprzedaży” (te dwie kolumny mają kluczowe znaczenie). 
Skup się na najnowszych danych (0m) (o ile są dostępne), ale uwzględnij też wcześniejsze okresy: -1m to rekomendacje sprzed misiąca, -2m to sprzed dwóch a -3m to sprzed 3. Zamiast pisać -0m to pisz że aktualne rekomendacje, zamiast pisać -1m pisz że rekomendacje sprzed miesiąca, itp. Aby wychwycić ewentualne zmiany. Na końcu oceń, czy rekomendacje analityków są ogólnie pozytywne, neutralne czy negatywne dla zakupu akcji. Jeśli tabela nie zawiera danych, po prostu stwierdź, że brak jest rekomendacji analityków i nie wydawaj oceny czy jest to negatywne/pozytywne.
Napisz to krotko, 3-4 zdania. Najwazniejsze jest podsumowanie czy ogólnie oceny są Pozytywne/neutralne/negatywne dla inwestycji w akcje tej firmy. Napisz pogrubieniem któryś z wyrazów Pozytywne/neutralne/negatywne.

Poniższa tabela zawiera dane:
{df}
"""
    
    # Utwórz instancję klienta OpenAI
    client = openai.OpenAI(api_key=api_key)

    # Użyj klienta do stworzenia zapytania
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Ma to być fragment do raportu inwestycyjnego, same konkrety, pisz jasno że analitycy sugerują sprzedaż lub kupno akcji"},
            {"role": "user", "content": prompt}
        ]
    )

    summary = response.choices[0].message.content.strip()
    return summary

# Przykład użycia funkcji
# summary = summarize_recommendations(df)


#Definiowanie funkcji do analizowania sytuacji finansowej firmy na podstawie wskaźników fin.
def summarize_indicators(df):
    # Konwertuj tabelę do stringa
    table_string_indc = df.to_string(index=True)
    
    # Stwórz prompt z tabelą
    prompt = f"Przeanalizuj krótko sytuację finansową firmy na podstawie wskaźników:\n\n{table_string_indc}\n\n uwzględnij to w jakiej branży działa firma {company_name} wskaźniki z tej samej kategorii analizuj razem, w jednym punkcie (na przykład płynność) i dodawaj ocenę wskaźników przy każdej kategorii: Negatywne/Neutralne/Pozytywne pod kątem zakupu akcji, na końcu przy podsumowaniu pisz ocenę ogólną pogrubieniem(ale tylko słowa Negatywne/Neutralne/Pozytywne mają być pogrubione). Poszczególne kategorie oddzielaj linią. Bierz pod uwgę też aktualny poziom wskaźników danej kategorii, czy są na odpowiednim poziomie. Nie pisz nagłówka, ale podpunkty z pogrubieniem czego dotyczy pisz. Nie wypisuj wszystkich danych w tekście, użytkownik będzie miał tablkę załączoną. Pamiętaj też że najświeższe dane mogą być międzyokresowe i ewentualne odchylenia mogą wynikać z sezonowości danej firmy."

    # Utwórz instancję klienta OpenAI
    client = openai.OpenAI(api_key=api_key)

    # Użyj klienta do stworzenia zapytania
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Ma to być fragment do raportu inwestycyjnego, same konkrety, pisz jasno czy dobre czy złe wskaźniki finansowe"},
            {"role": "user", "content": prompt}
        ]
    )

    summary_ind = response.choices[0].message.content.strip()
    return summary_ind



#Funkcja do analizy wskazników giełdowych
def summarize_market_indicators(df):
    # Konwertuj tabelę do stringa
    indicators_2_df_str = df.to_string(index=True)
    
    # Stwórz prompt z tabelą
    prompt = f"Przeanalizuj krótko sytuację finansową firmy na podstawie wskaźników:\n\n{indicators_2_df_str}\n\n uwzględnij to w jakiej branży działa firma {company_name} wskaźniki z tej samej kategorii analizuj razem, w jednym punkcie (na przykład płynność) i dodawaj ocenę wskaźników przy każdej kategorii: Negatywne/Neutralne/Pozytywne pod kątem zakupu akcji, na końcu przy podsumowaniu pisz ocenę ogólną pogrubieniem(ale tylko słowa Negatywne/Neutralne/Pozytywne mają być pogrubione). Poszczególne kategorie oddzielaj linią. Bierz pod uwgę też aktualny poziom wskaźników danej kategorii, czy są na odpowiednim poziomie. Nie pisz nagłówka, ale podpunkty z pogrubieniem czego dotyczy pisz. Nie wypisuj wszystkich danych w tekście, użytkownik będzie miał tablkę załączoną. Pamiętaj też że najświeższe dane mogą być międzyokresowe i ewentualne odchylenia mogą wynikać z sezonowości danej firmy."

    # Utwórz instancję klienta OpenAI
    client = openai.OpenAI(api_key=api_key)

    # Użyj klienta do stworzenia zapytania
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Ma to być fragment do raportu inwestycyjnego, same konkrety, pisz jasno czy dobre czy złe wskaźniki finansowe"},
            {"role": "user", "content": prompt}
        ]
    )

    summary_stock_ind = response.choices[0].message.content.strip()
    return summary_stock_ind






# Definiowanie funkcji do analizowania sytuacji finansowej firmy na podstawie wskaźników finansowych i podstawowych danych finansowych
def summarize_financials_with_percent_changes(percent_changes, basic_fin, company_name):
    # Konwertuj tabele do stringa
    table_string_percent_changes = percent_changes.to_string(index=True)
    table_string_basic = basic_fin.to_string(index=True)
    
    # Stwórz prompt z tabelami
    prompt = f"Przeanalizuj krótko sytuację finansową firmy {company_name} na podstawie procentowych zmian danych finansowych i podstawowych danych finansowych:\n\nProcentowe zmiany danych finansowych:\n{table_string_percent_changes}\n\nPodstawowe dane finansowe:\n{table_string_basic}\n\n Uwzględnij to, w jakiej branży działa firma {company_name}. Dane z tej samej kategorii analizuj razem, w jednym punkcie i dodawaj ocenę grupy danych przy każdej kategorii: Negatywne/Neutralne/Pozytywne pod kątem zakupu akcji, Pisz to na końcu z pogrubieniem, na końcu przy podsumowaniu pisz ocenę ogólną pogrubieniem. Poszczególne kategorie oddzielaj linią. Bierz pod uwagę w jakiej branży działa firma, czy taki poziom poszczególnych danych jest charakterystyczny dla danego sektora. Nie pisz nagłówka, ale podpunkty z pogrubieniem czego dotyczy pisz. Nie wypisuj wszystkich danych w tekście, użytkownik będzie miał tabelę załączoną. Pamiętaj też, że najświeższe dane mogą być międzyokresowe i ewentualne odchylenia mogą wynikać z sezonowości danej firmy."

    # Utwórz instancję klienta OpenAI
    client = openai.OpenAI(api_key=api_key)

    # Użyj klienta do stworzenia zapytania
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Ma to być fragment do raportu inwestycyjnego, same konkrety, pisz jasno czy dobre czy złe wskaźniki finansowe"},
            {"role": "user", "content": prompt}
        ]
    )

    summary_financials = response.choices[0].message.content.strip()
    return summary_financials


def Strenghts(company_name):

    # Stwórz prompt z tabelą
    prompt = f"Przeanalizuj krótko sytuację firmy {company_name}, napisz jakie mocne strony ma, staraj się pisać głównie te rzeczy które wyróżniają ją na tle konkurencji. Nie pisz wstępu. Pisz 3 główne punkty - najmocniejsze strony i krótko je rozwiń dlaczego akurat to jest ich mocną stroną na tle konkurencji."

    # Utwórz instancję klienta OpenAI
    client = openai.OpenAI(api_key=api_key)

    # Użyj klienta do stworzenia zapytania
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Ma to być fragment do raportu inwestycyjnego, same konkrety, pisz jasno."},
            {"role": "user", "content": prompt}
        ]
    )

    Strenghts = response.choices[0].message.content.strip()
    return Strenghts


def Weaknesses(company_name):

    # Stwórz prompt z tabelą
    prompt = f"Przeanalizuj krótko sytuację firmy {company_name}, napisz jakie słabe strony ma, jakie słabości, staraj się pisać głównie te rzeczy które wyróżniają ją na tle konkurencji. Nie pisz wstępu. Pisz 3 główne punkty - najsłabsze strony i krótko je rozwiń dlaczego akurat to jest ich słabą stroną na tle konkurencji."

    # Utwórz instancję klienta OpenAI
    client = openai.OpenAI(api_key=api_key)

    # Użyj klienta do stworzenia zapytania
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Ma to być fragment do raportu inwestycyjnego, same konkrety, pisz jasno."},
            {"role": "user", "content": prompt}
        ]
    )

    Weaknesses = response.choices[0].message.content.strip()
    return Weaknesses

def Opportunities(company_name):

    # Stwórz prompt z tabelą
    prompt = f"Przeanalizuj krótko sytuację firmy {company_name}, napisz jakie szanse przed nią stoją, staraj się pisać głównie te okazje dla firmy które wyróżniają ją na tle konkurencji. Nie pisz wstępu. Pisz 3 główne punkty - największe szanse i krótko je rozwiń dlaczego akurat to może być ich szansą na tle konkurencji."

    # Utwórz instancję klienta OpenAI
    client = openai.OpenAI(api_key=api_key)

    # Użyj klienta do stworzenia zapytania
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Ma to być fragment do raportu inwestycyjnego, same konkrety, pisz jasno."},
            {"role": "user", "content": prompt}
        ]
    )

    Opportunities = response.choices[0].message.content.strip()
    return Opportunities


def Threats(company_name):

    # Stwórz prompt z tabelą
    prompt = f"Przeanalizuj krótko sytuację firmy {company_name}, napisz jakie zagrożenia przed nią stoją, staraj się pisać głównie o tych zagrożeniach dla firmy które wyróżniają ją na tle konkurencji. Nie pisz wstępu. Pisz 3 główne punkty - największe zagrożenia i krótko je rozwiń dlaczego akurat to może być ich zagrożeniem na tle konkurencji."

    # Utwórz instancję klienta OpenAI
    client = openai.OpenAI(api_key=api_key)

    # Użyj klienta do stworzenia zapytania
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Ma to być fragment do raportu inwestycyjnego, same konkrety, pisz jasno."},
            {"role": "user", "content": prompt}
        ]
    )

    Threats = response.choices[0].message.content.strip()
    return Threats



def SWOT_summary(S,W,O,T,name_ticker):

    # Stwórz prompt z tabelą
    prompt = f"Napisz krótkie podsumowanie wszystkich składowych analizy SWOT firmy {name_ticker}. Mocnych stron: {S}, Słabych stron: {W}, Okazji: {O} i zagrożeń: {T}. Na końcu podsumowania napisz jedno zdanie w którym POGRUBIENIEM! napiszesz czy na podstawie analizy SWOT sytuacja jest Negatywna lub Neutralna lub pozytywna dla zakupu akcji tej firmy. Nie pisz podpunktów, napisz podsumowanie ciągiem. Ma mieć maksymalnie 5 zdań."

    # Utwórz instancję klienta OpenAI
    client = openai.OpenAI(api_key=api_key)

    # Użyj klienta do stworzenia zapytania
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Ma to być fragment do raportu inwestycyjnego, same konkrety, pisz jasno."},
            {"role": "user", "content": prompt}
        ]
    )

    SWOT_summary = response.choices[0].message.content.strip()
    return SWOT_summary





def Report_summary(Recommendations, Basic_fin, Fin_ind, Market_ind, SWOT, NEWS):

    # Stwórz prompt z tabelą
    prompt = f"Napisz krótkie podsumowanie raportu na temat inwestycji w {company_name}, podsumowanie oprzyj na tych danych: {Recommendations}, {Basic_fin}, {Fin_ind}, {Market_ind}, {SWOT}, {NEWS}. Podsumowanie musi zawierać odniesienie do każdego fragmentu ale nie pisz tego samego, już nie podawaj masy liczb. Na koniec ma być zdanie oceniające czy aktualnie warto kupować czy nie. Staraj sie możliwie rzadko używać określenia że trudno określic. Rekomendację napisz pogrubieniem. Zawsze na koniec musi być czy według stanu na dzień dzisiejszy: Nie warto inwestować, Warto inwestować, Nie da się jednoznacznie określić. Nie pisz nagłówka. "

    # Utwórz instancję klienta OpenAI
    client = openai.OpenAI(api_key=api_key)

    # Użyj klienta do stworzenia zapytania
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Ma to być fragment do raportu inwestycyjnego, same konkrety, pisz jasno."},
            {"role": "user", "content": prompt}
        ]
    )

    Report_summary = response.choices[0].message.content.strip()
    return Report_summary






def get_article_links(ticker):
    # Tworzymy URL na podstawie tickeru
    url = f'https://finance.yahoo.com/quote/{ticker}/latest-news/'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    
    # Pobieramy stronę HTML
    response = requests.get(url, headers=headers)
    
    # Sprawdzamy, czy żądanie zakończyło się sukcesem
    if response.status_code != 200:
        print(f"Nie udało się pobrać strony: {url}")
        return []
    
    # Parsujemy HTML za pomocą BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Znajdujemy wszystkie linki do artykułów
    article_elements = soup.select('a.subtle-link')
    
    # Wyodrębniamy unikalne URL z każdego elementu i sprawdzamy, czy zaczynają się na 'https://finance.yahoo.com/news/'
    article_links = set()  # Używamy zbioru do przechowywania unikalnych linków
    for element in article_elements:
        article_url = element['href']
        if article_url.startswith('/'):
            article_url = f"https://finance.yahoo.com{article_url}"
        
        # Sprawdzenie, czy link zaczyna się na 'https://finance.yahoo.com/news/'
        if article_url.startswith('https://finance.yahoo.com/news/'):
            article_links.add(article_url)  # Zbiór automatycznie ignoruje duplikaty
    
    return list(article_links)  # Konwertujemy z powrotem na listę

def scrape_article_text(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        print(f"Nie udało się pobrać artykułu: {url}")
        return "Brak treści."
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Znajdź div, który zawiera treść artykułu
    article_text = soup.find('div', class_='caas-body')
    if article_text:
        return article_text.get_text().strip()
    else:
        return "Brak treści."

def create_articles_dataframe(ticker):
    links = get_article_links(ticker)
    
    data = {
        "Link": [],
        "Treść": []
    }
    
    for link in links[:10]:  # Limituje do 10 artykułów
        text = scrape_article_text(link)
        data["Link"].append(link)
        data["Treść"].append(text)
    
    # Tworzymy DataFrame
    df = pd.DataFrame(data)
    return df

# Przykładowe użycie



def summarize_news(df_streszczenia_to_AI):
    # Konwertuj tabelę do stringa
    table_string_indc = df_streszczenia_to_AI.to_string(index=True)
    
    # Stwórz prompt z tabelą
    prompt = f"Przeanalizuj najnowsze wiadomości dotyczące {company_name} zawarte w tabeli:\n\n{table_string_indc}\n\n. Napisz podsumowanie wiadomości a w ostatnim zdaniu napisz pogrubieniem czy jest to negatywne lub neutralne lub pozytywne pod kątem inwestycji w akcje firmy {company_name}"

    # Utwórz instancję klienta OpenAI
    client = openai.OpenAI(api_key=api_key)

    # Użyj klienta do stworzenia zapytania
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Ma to być fragment do raportu inwestycyjnego, same konkrety, pisz jasno."},
            {"role": "user", "content": prompt}
        ]
    )

    summary_news = response.choices[0].message.content.strip()
    return summary_news


# Przykładowe użycie

# Utworzenie instancji klienta OpenAI
# client = openai.OpenAI(api_key=api_key)  # Usunięto, używamy openai.ChatCompletion.create bezpośrednio

def streszczenie_artykułów(df):
    # Dodanie kolumny na streszczenia
    df['Streszczenie'] = ''
    
    # Iteracja przez wiersze DataFrame
    for index, row in df.iterrows():
        content = row['Treść']
        client = openai.OpenAI(api_key=api_key)
        # Wykonanie zapytania do API OpenAI
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Streść ten artykuł: {content}. Poszukaj w nim informacji na temat {company_name}. Oceń i napisz pogrubieniem czy w artykule pisano o {company_name} który był pozytywny dla inwestycji w akcje tej firmy, neutralny czy negatywny."}
            ]
        )
        
        # Zbieranie odpowiedzi
        summary = response.choices[0].message.content
        
        # Zapisanie streszczenia do DataFrame
        df.at[index, 'Streszczenie'] = summary
    
    return df


st.title('Raport inwestycyjny')

# Pole tekstowe do wpisania nazwy
company_name = st.text_input("Wprowadź nazwę firmy", "")

if st.button('Wygeneruj raport'):
    # Twoje API key OpenAI

    ticker = get_ticker_for_company(company_name)

    if not ticker:
        st.warning("Please enter a ticker")
    else:
        try:
            stock = yf.Ticker(ticker)
            company_name = stock.info.get('longName')

            # Jeśli nie ma nazwy firmy, wyświetl komunikat "Incorrect ticker"
            if not company_name:
                st.error("Nieprawidłowy ticker: " + ticker + " lub błąd Yahoo Finance. Spróbuj ponownie wprowadzając ticker firmy z Yahoo Finance.")
            else:
                # Pobranie rekomendacji analityków
                recommendations = stock.recommendations
                name_ticker = (company_name + " " + ticker)

                if recommendations is None or recommendations.empty:
                    st.warning("Brak rekomendacji analityków dla tej firmy.")
                else:
                    # Ustawienie kolumny 'period' jako indeks i usunięcie pierwszej pustej kolumny
                    recommendations = recommendations.reset_index().set_index('period')

                    # Usunięcie kolumny 'index'
                    recommendations.drop(columns=['index'], inplace=True)
                    
                    # Usunięcie wierszy, gdzie wszystkie liczby to 0
                    recommendations = recommendations.loc[~(recommendations[['strongBuy', 'buy', 'hold', 'sell', 'strongSell']] == 0).all(axis=1)]
                    recommendations = pd.DataFrame(recommendations)
                    
                    # Wyświetlenie rekomendacji w Streamlit
                    # st.subheader(f'Rekomendacje analityków dla {company_name} ({ticker})')

                    # Pobranie informacji o firmie
                    info = stock.info
                    currency_name = info.get('currency')

                    # Zmiana nazw kolumn na polskie
                    recommendations = recommendations.rename(columns={
                        'period': 'Okres',
                        'strongBuy': 'Silne rekom. kupna',
                        'buy': 'Rekom. kupna',
                        'hold': 'Rekom. trzymaj',
                        'sell': 'Rekom. sprzedaży',
                        'strongSell': 'Silne rekom. sprzedaży'})
                    
                    # Najpierw wyznaczam sumę rekomendacji dla każdego okresu
                    total_recommendations = recommendations[['Silne rekom. kupna', 'Rekom. kupna', 'Rekom. trzymaj', 'Rekom. sprzedaży', 'Silne rekom. sprzedaży']].sum(axis=1)

                    # Tworzę nową tabelę z procentowym udziałem
                    recommendations_percentage = recommendations.copy()

                    # Dzielę każdą kolumnę przez sumę rekomendacji dla danego okresu, mnożę przez 100, zaokrąglam do liczb całkowitych
                    recommendations_percentage[['Silne rekom. kupna', 'Rekom. kupna', 'Rekom. trzymaj', 'Rekom. sprzedaży', 'Silne rekom. sprzedaży']] = \
                    (recommendations[['Silne rekom. kupna', 'Rekom. kupna', 'Rekom. trzymaj', 'Rekom. sprzedaży', 'Silne rekom. sprzedaży']].div(total_recommendations, axis=0)) * 100

                    # Zaokrąglam wyniki do liczb całkowitych
                    recommendations_percentage[['Silne rekom. kupna', 'Rekom. kupna', 'Rekom. trzymaj', 'Rekom. sprzedaży', 'Silne rekom. sprzedaży']] = \
                    recommendations_percentage[['Silne rekom. kupna', 'Rekom. kupna', 'Rekom. trzymaj', 'Rekom. sprzedaży', 'Silne rekom. sprzedaży']].round(0).astype(int)
                    
                    # Dodaję znak '%' do każdej wartości
                    for column in ['Silne rekom. kupna', 'Rekom. kupna', 'Rekom. trzymaj', 'Rekom. sprzedaży', 'Silne rekom. sprzedaży']:
                        recommendations_percentage[column] = recommendations_percentage[column].astype(str) + '%'

                # Pobranie danych o dywidendach
                dividends = stock.dividends

                # Lista do przechowywania wyników
                results = []

                # Iteruj przez daty wypłat dywidend
                for date, dividend in dividends.items():
                    # Pobierz cenę zamknięcia akcji z dnia wypłaty dywidendy
                    history = stock.history(start=date, end=date + pd.Timedelta(days=1))
                    if not history.empty:
                        close_price = history['Close'].iloc[0]  # Użyj .iloc do uzyskania wartości według pozycji
                        
                        # Oblicz dividend yield jako procent
                        dividend_yield = (dividend / close_price) * 100
                        
                        # Dodaj dane do listy
                        results.append({
                            "Date": date,
                            "Dividend": dividend,
                            "Close Price": close_price,
                            "Dividend Yield (%)": dividend_yield
                        })

                # Stwórz DataFrame z wyników
                dividend_df = pd.DataFrame(results)
                # 1. Konwersja kolumny 'date' do formatu dd-mm-rrrr
                dividend_df_desc=dividend_df.sort_values('Date',ascending=False)
                dividend_df_desc['Date'] = pd.to_datetime(dividend_df_desc['Date']).dt.strftime('%d-%m-%Y')
                dividend_df_desc = dividend_df_desc.set_index('Date')
                dividend_df_desc = dividend_df_desc.round(1)

                dividend_df['Date'] = pd.to_datetime(dividend_df['Date']).dt.strftime('%d-%m-%Y')

                # 2. Ustawienie kolumny 'date' jako indeks i usunięcie starego indeksu
                dividend_df = dividend_df.set_index('Date')

                # 3. Zaokrąglenie wszystkich pozostałych kolumn do 1 miejsca po przecinku
                dividend_df = dividend_df.round(1)

                # Pobierz historyczne dane o cenach akcji
                history_df = stock.history(period="10y")[['Close']].reset_index()
                # Tworzymy history_df_close_only z kolumnami 'Date' i 'Close'
                history_df_close_only = history_df[['Date', 'Close']]
                # Zmień kolumnę 'Date' na typ datetime
                history_df_close_only['Date'] = pd.to_datetime(history_df['Date'])
                # Ustawienie indeksu jako 'Date' i zmiana indeksu na daty bez czasu
                history_df_close_only.set_index('Date', inplace=True)
                history_df_close_only.index = history_df_close_only.index.date



                # Pobranie rocznych sprawozdań finansowych
                annual_financials = stock.financials.T.sort_index(ascending=False)
                # Pobranie rocznego bilansu
                annual_balance_sheet = stock.balance_sheet.T.sort_index(ascending=False)

                # Pobranie najnowszych międzyokresowych sprawozdań finansowych
                quarterly_financials = stock.quarterly_financials.T.sort_index(ascending=False)

                # Pobranie najnowszego międzyokresowego bilansu
                quarterly_balance_sheet = stock.quarterly_balance_sheet.T.sort_index(ascending=False)

                # Sprawdzenie, czy najnowsze międzyokresowe dane są nowsze niż najnowsze roczne dane
                latest_annual_financial_date = annual_financials.index[0]
                latest_quarterly_financial_date = quarterly_financials.index[0]

                latest_annual_balance_date = annual_balance_sheet.index[0]
                latest_quarterly_balance_date = quarterly_balance_sheet.index[0]

                # Normalizacja międzyokresowych danych finansowych do rocznych
                def normalize_to_annual(data, period_length_in_months):
                    return data * (12 / period_length_in_months)

                if latest_quarterly_financial_date > latest_annual_financial_date:
                    normalized_quarterly_financials = normalize_to_annual(quarterly_financials.head(1), 3)
                    latest_financials = pd.concat([normalized_quarterly_financials, annual_financials])
                else:
                    latest_financials = annual_financials

                if latest_quarterly_balance_date > latest_annual_balance_date:
                    latest_balance_sheet = pd.concat([quarterly_balance_sheet.head(1), annual_balance_sheet])
                else:
                    latest_balance_sheet = annual_balance_sheet

                
                latest_balance_sheet_2 = latest_balance_sheet
                latest_financials_2=latest_financials

                # Obliczanie wskaźników finansowych dla wszystkich dostępnych lat
                results = []
                for date in latest_financials.index:
                    latest_financials_row = latest_financials.loc[date]
                    latest_balance_sheet_row = latest_balance_sheet.loc[date]
                    
                    current_assets = latest_balance_sheet_row.get('Current Assets')
                    current_liabilities = latest_balance_sheet_row.get('Current Liabilities')
                    inventory = latest_balance_sheet_row.get('Inventory')
                    total_liabilities = latest_balance_sheet_row.get('Total Liabilities Net Minority Interest')
                    total_equity = latest_balance_sheet_row.get('Total Equity Gross Minority Interest')
                    
                    current_ratio = current_assets / current_liabilities if current_assets and current_liabilities else None
                    quick_ratio = (current_assets - inventory) / current_liabilities if current_assets and inventory and current_liabilities else None
                    debt_to_equity_ratio = total_liabilities / total_equity if total_liabilities and total_equity else None
                    
                    total_revenue = latest_financials_row.get('Total Revenue')
                    cost_of_revenue = latest_financials_row.get('Cost Of Revenue')
                    gross_margin = total_revenue - cost_of_revenue if total_revenue and cost_of_revenue else None
                    gross_margin_ratio = (gross_margin / total_revenue * 100) if gross_margin and total_revenue else None
                    ebit = latest_financials_row.get('EBIT')
                    operating_margin = (ebit / total_revenue * 100) if ebit and total_revenue else None
                    net_income = latest_financials_row.get('Net Income')
                    net_profit_margin = (net_income / total_revenue * 100) if net_income and total_revenue else None
                    
                    total_assets = latest_balance_sheet_row.get('Total Assets')
                    return_on_assets = (net_income / total_assets * 100) if net_income and total_assets else None
                    return_on_equity = (net_income / total_equity * 100) if net_income and total_equity else None
                    
                    results.append({
                        "Date": date.strftime("%Y-%m-%d"),
                        "Current Ratio": current_ratio,
                        "Quick Ratio": quick_ratio,
                        "Debt to Equity Ratio": debt_to_equity_ratio,
                        "Gross Margin Ratio (%)": gross_margin_ratio,
                        "Operating Margin (%)": operating_margin,
                        "Net Profit Margin (%)": net_profit_margin,
                        "Return on Assets (ROA) (%)": return_on_assets,
                        "Return on Equity (ROE) (%)": return_on_equity
                    })

                    # Tworzenie Df z najważniejszymi danymi finansowymi
                    # Tworzenie tabelki z najważniejszymi danymi finansowymi
                    financial_metrics = ['Total Revenue', 'Total Revenue (Annualized)', 'EBIT', 'EBIT (Annualized)', 'Net Income', 'Net Income (Annualized)']
                    balance_sheet_metrics = ['Total Assets', 'Total Liabilities Net Minority Interest', 'Total Equity Gross Minority Interest']

                    data_fin = []

                    for date in latest_financials.index:
                        row = {'Date': date.strftime("%d.%m.%Y")}
                        financials_row = latest_financials.loc[date]
                        balance_sheet_row = latest_balance_sheet.loc[date]
                        
                        for metric in financial_metrics:
                            if metric in financials_row:
                                row[metric] = financials_row[metric]
                        
                        for metric in balance_sheet_metrics:
                            if metric in balance_sheet_row:
                                row[metric] = balance_sheet_row[metric]
                        
                        data_fin.append(row)

                    # Konwersja wyników do DataFrame
                    basic_fin = pd.DataFrame(data_fin)

                    # Usuń wiersze, które mają we wszystkich kolumnach NaN
                    basic_fin = basic_fin.dropna(how='all')

                    # Usuń wiersze, które mają we wszystkich kolumnach NaN
                    basic_fin = basic_fin.loc[~(basic_fin.isna().all(axis=1))]

                    # Ustaw kolumnę 'Date' jako indeks
                    basic_fin.set_index('Date', inplace=True)

                    # Słownik mapujący angielskie nazwy kolumn na polskie
                    column_mapping = {
                        'Total Revenue': 'Przychody Ogółem',
                        'EBIT': 'EBIT',
                        'Net Income': 'Zysk Netto',
                        'Total Assets': 'Aktywa Ogółem',
                        'Total Liabilities Net Minority Interest': 'Zobowiązania Ogółem',
                        'Total Equity Gross Minority Interest': 'Kapitał Własny',
                        'Total Revenue (Annualized)': 'Przychody Ogółem (Urocznione)',
                        'EBIT (Annualized)': 'EBIT (Urocznione)',
                        'Net Income (Annualized)': 'Zysk Netto (Uroczniony)'
                    }

                    # Zmiana nazw kolumn na polskie
                    basic_fin.rename(columns=column_mapping, inplace=True)

                    # Formatowanie liczb z użyciem spacji jako separatorów tysięcznych
                    def format_number(x):
                        if pd.isna(x):
                            return ''
                        elif isinstance(x, (int, float)):
                            return f'{x:,.0f}'.replace(',', ' ')
                        else:
                            return x

                    basic_fin = basic_fin.map(format_number)

                # Usuń wiersze, które mają puste wszystkie komórki oprócz indeksu
                basic_fin.dropna(how='all', subset=basic_fin.columns.difference(['Date']), inplace=True)

                # Uzupełnij wszystkie puste komórki wartością 0
                basic_fin.fillna(0, inplace=True)

                # Usuń wiersze, które mają puste wszystkie komórki oprócz indeksu
                basic_fin = basic_fin.loc[~((basic_fin == 0) | (basic_fin == '')).all(axis=1)]

                # Zmień typ danych dla indeksu na datę
                basic_fin.index = pd.to_datetime(basic_fin.index, format='%d.%m.%Y')

                # Posortuj dane od najnowszych do najstarszych
                basic_fin = basic_fin.sort_index(ascending=False)

                basic_fin_1 = basic_fin.apply(pd.to_numeric, errors='coerce')

                # Zmień typ danych dla indeksu na datę z uwzględnieniem formatu 'DD.MM.YYYY'
                basic_fin.index = pd.to_datetime(basic_fin.index, format='%d.%m.%Y')

                # Posortuj dane od najnowszych do najstarszych
                basic_fin = basic_fin.sort_index(ascending=False)

                # Usuń spacje z danych i zamień je na liczby
                basic_fin_1 = basic_fin.replace(r'\s+', '', regex=True)
                basic_fin_1 = basic_fin_1.apply(pd.to_numeric, errors='coerce')

                # Oblicz zmiany procentowe według wzoru:
                # (Wartość z nowszego okresu - Wartość ze starszego okresu) / Wartość ze starszego okresu * 100
                percent_changes = (basic_fin_1 - basic_fin_1.shift(-1)) / basic_fin_1.shift(-1) * 100

                # Dodaj warunek dla wartości ujemnych w starszym okresie
                for col in percent_changes.columns:
                    previous_values = basic_fin_1[col].shift(-1)
                    current_values = basic_fin_1[col]
                    # Zaktualizuj procentową zmianę tylko wtedy, gdy poprzednia wartość jest ujemna, a obecna dodatnia
                    percent_changes[col] = np.where((previous_values < 0) & (current_values > 0),
                                                    (current_values - previous_values) / abs(previous_values) * 100,
                                                    percent_changes[col])

                # Zaokrąglij zmiany do pełnych procentów
                percent_changes = percent_changes.round(0)

                # Usuń wiersz, który powstaje w wyniku pierwszego obliczenia zmian procentowych
                percent_changes = percent_changes.dropna()

                # Dodaj kolumnę z okresami w formacie YYYY/MM - YYYY/MM bez końcówki /31
                percent_changes['Date'] = basic_fin.index.to_series().shift(-1).dt.strftime('%Y/%m') + " - " + basic_fin.index.to_series().dt.strftime('%Y/%m')

                # Usuń starą kolumnę Date i ustaw nową jako indeks
                percent_changes = percent_changes.set_index('Date')

                # Dodaj "%" do wszystkich wartości procentowych
                percent_changes = percent_changes.astype(str) + '%'
                percent_changes = pd.DataFrame(percent_changes)

                #percent_changes = pd.DataFrame(percent_changes)


                # Stworzenie DataFrame z wyników
                indicators_df = pd.DataFrame(results).set_index('Date')
                # Usunięcie wierszy, gdzie wszystkie wartości w danym wierszu są puste
                indicators_df.dropna(how='all', inplace=True)

                # Wyświetlenie wskaźników finansowych w Streamlit
                # Zmiana nazw kolumn na polskie
                indicators_df = indicators_df.rename(columns={
                    'Date': 'Data',
                    'Current Ratio': 'Wskaźnik bież. płynności',
                    'Quick Ratio': 'Wskaźnik szybki',
                    'Debt to Equity Ratio': 'zadłużenie do kap. własnego',
                    'Gross Margin Ratio (%)': 'Marża brutto %',
                    'Operating Margin (%)': 'Marża operacyjna %',
                    'Net Profit Margin (%)': 'Marża zysku netto %',
                    'Return on Assets (ROA) (%)': 'Zwrot z aktywów (ROA) %',
                    'Return on Equity (ROE) (%)': 'Zwrot z kap. własnego (ROE) %'})
                
                 #Generowanie podsumowania wskaźników fin.
                summary_ind = summarize_indicators(indicators_df)
                #generowanie podsumowania podstawowych danych finansowych
                summary_fin = summarize_financials_with_percent_changes(percent_changes, basic_fin, company_name)




                # Upewnij się, że indeks jest typu datetime
                basic_fin.index = pd.to_datetime(basic_fin.index)

                # Usuń część czasu, pozostawiając tylko datę
                basic_fin.index = basic_fin.index.normalize()




                # Generowanie wykresów ze wskaźnikami giełdowymi
                # Konwersja indeksu latest_financials_2 i latest_balance_sheet_2 na daty bez czasu
                latest_financials_2.index = pd.to_datetime(latest_financials_2.index).date
                latest_balance_sheet_2.index = pd.to_datetime(latest_balance_sheet_2.index).date

                # Funkcja do znalezienia najbliższej dostępnej ceny akcji
                def get_closest_stock_price(date, history_df_close_only):
                    if date in history_df_close_only.index:
                        return history_df_close_only.at[date, 'Close']
                    
                    # Znajdź najbliższą datę, jeżeli brak dokładnego dopasowania
                    closest_date = min(history_df_close_only.index, key=lambda d: abs(d - date))
                    return history_df_close_only.at[closest_date, 'Close']

                # Obliczanie wskaźników finansowych historycznie
                indicators = []

                # Przechodzimy przez dane i obliczamy wskaźniki
                for date in latest_financials_2.index:
                    # Wyciągamy dane finansowe na dany okres
                    net_income = latest_financials_2.at[date, 'Net Income']  # Używamy kolumny 'Net Income'
                    total_equity = latest_balance_sheet_2.at[date, 'Total Equity Gross Minority Interest']  # Używamy kolumny 'Total Equity Gross Minority Interest'
                    total_assets = latest_balance_sheet_2.at[date, 'Total Assets']
                    total_debt = latest_balance_sheet_2.at[date, 'Total Debt']
                    ebitda = latest_financials_2.at[date, 'EBITDA']
                    
                    # Pobranie liczby akcji na dany okres z kolumny "Share Issued"
                    shares = latest_balance_sheet_2.at[date, 'Share Issued']
                    
                    # Pobranie najbliższej dostępnej ceny akcji
                    stock_price = get_closest_stock_price(date, history_df_close_only)
                    
                    # Obliczenie kapitalizacji rynkowej
                    market_cap = stock_price * shares if stock_price and shares else None

                    # Obliczanie wskaźników
                    pe_ratio = market_cap / net_income if market_cap and net_income else None
                    book_value_per_share = total_equity / shares if shares else None
                    pb_ratio = market_cap / (book_value_per_share * shares) if market_cap and shares else None
                    roe = net_income / total_equity if total_equity else None
                    roa = net_income / total_assets if total_assets else None
                    de_ratio = total_debt / total_equity if total_equity else None
                    ev = market_cap + total_debt - latest_balance_sheet_2.at[date, 'Cash Cash Equivalents And Short Term Investments'] if market_cap else None
                    ev_to_ebitda = ev / ebitda if ev and ebitda else None

                    # Dodanie wskaźników do listy
                    indicators.append({
                        'Date': date,
                        'Stock Price': stock_price,
                        'Market Cap': market_cap,
                        'P/E Ratio': pe_ratio,
                        'P/B Ratio': pb_ratio,
                        'ROE': roe,
                        'ROA': roa,
                        'Debt-to-Equity Ratio': de_ratio,
                        'EV/EBITDA': ev_to_ebitda,
                        'EPS': latest_financials_2.at[date, 'Basic EPS']  # Używamy 'Basic EPS'
                    })

                # Konwersja do DataFrame i wyświetlenie wyników
                indicators_2_df = pd.DataFrame(indicators)

                # Wybór odpowiednich kolumn z dataframe
                indicators_2_df = indicators_2_df[['Date', 'P/E Ratio', 'P/B Ratio', 'ROE', 'ROA', 'EV/EBITDA', 'EPS']]
               
                indicators_2_df = indicators_2_df.dropna(how='all', subset=indicators_2_df.columns.difference(['Date']))

               # Utworzenie nowego DataFrame na podstawie indicators_2_df
                AI_indicators_df = indicators_2_df.copy()
                # Zastąpienie ujemnych wartości w kolumnie 'P/E Ratio' informacją o ujemnych zyskach
                AI_indicators_df['P/E Ratio'] = AI_indicators_df['P/E Ratio'].apply(lambda x: "Zysk poniżej zera" if x < 0 else x)
                # Zastąpienie ujemnych wartości w kolumnie 'EV/EBITDA' informacją o ujemnych zyskach
                AI_indicators_df['EV/EBITDA'] = AI_indicators_df['EV/EBITDA'].apply(lambda x: "Zysk poniżej zera" if x < 0 else x)


               # Zastąp ujemne wartości w kolumnach P/E Ratio i EV/EBITDA na NaN
                indicators_2_df['P/E Ratio'] = indicators_2_df['P/E Ratio'].apply(lambda x: np.nan if x < 0 else x)
                indicators_2_df['EV/EBITDA'] = indicators_2_df['EV/EBITDA'].apply(lambda x: np.nan if x < 0 else x)






                # Konwersja kolumny 'Date' na format datetime, jeśli jeszcze nie jest w formacie datetime
                indicators_2_df['Date'] = pd.to_datetime(indicators_2_df['Date'])

                # Zastosowanie logiki, aby usunąć dni i pozostawić tylko YYYY/MM
                indicators_2_df['Date'] = indicators_2_df['Date'].dt.strftime('%Y/%m')

                # Ustawienie kolumny 'Date' jako indeksu
                indicators_2_df.set_index('Date', inplace=True)

                # Sortowanie danych według indeksu (czyli daty) rosnąco
                indicators_2_df.sort_index(ascending=True, inplace=True)

                

                # Creating subplots
                fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 16))

                # Flatten axes for easy iteration
                axes = axes.flatten()

                # List of columns to plot
                columns = ['P/E Ratio', 'P/B Ratio', 'ROE', 'ROA', 'EV/EBITDA', 'EPS']

                # Colors for bars
                colors = plt.cm.viridis(np.linspace(0, 1, len(columns)))

                # Plotting each column
                for ax, column, color in zip(axes, columns, colors):
                    bars = indicators_2_df[column].plot(kind='bar', ax=ax, color=color)
                    ax.set_title(column)
                    ax.set_xticklabels(indicators_2_df.index, rotation=45, ha='right')  # Używamy wartości indeksu jako etykiet
                    
                    # Adding values on top of bars
                    for bar in bars.patches:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height(),
                            round(bar.get_height(), 2),
                            ha='center', va='bottom'
                        )

                # Adjust layout
                plt.tight_layout()

                #Generowanie opisów wskaźników przez AI
                summary_stock_indc = summarize_market_indicators(AI_indicators_df)

                # Tworzenie zakładek
                tabs = st.tabs(["Rekom. Analityków, SWOT, wiadomości", "Dane Finansowe i Dywidendy", "Analiza Wskaźnikowa", "Podsumowanie"])

                with tabs[0]:
                    st.subheader(f'Rekomendacje analityków dla {company_name} ({ticker})')
                    st.markdown("Tabela poniżej przedstawia procentowy udział poszczególnych rekomendacji w łącznej liczbie rekomendacji:")
                    st.dataframe(recommendations_percentage)
                    with st.expander("Rozwiń, aby zobaczyć tabelę z liczbą poszczególnych rekomendacji"):
                        st.dataframe(recommendations)
                    # Generowanie podsumowania rekomendacji analityków
                    summary = summarize_recommendations(recommendations_percentage)
                    st.markdown(summary)


                    st.subheader(f"Analiza SWOT firmy {company_name}:")
                    Strenghts_response = Strenghts(name_ticker)
                    Weaknesses_response = Weaknesses(name_ticker)
                    Opportunities_response = Opportunities(name_ticker)
                    Threats_response = Threats(name_ticker)

                    # Jeden expander zawierający wszystkie 4 podpunkty
                    with st.expander("Wszystkie podpunkty analizy SWOT:"):
                        # Mocne strony
                        st.markdown("### Mocne strony:")
                        st.markdown(Strenghts_response)
                        
                        # Słabe strony
                        st.markdown("### Słabe strony:")
                        st.markdown(Weaknesses_response)
                        
                        # Okazje
                        st.markdown("### Okazje:")
                        st.markdown(Opportunities_response)
                        
                        # Zagrożenia
                        st.markdown("### Zagrożenia:")
                        st.markdown(Threats_response)

                    SWOT_summary_response = SWOT_summary(Strenghts_response, Weaknesses_response, Opportunities_response, Threats_response, name_ticker)
                    st.markdown(SWOT_summary_response)

                    # Najnowsze wiadomości
                    Articles = create_articles_dataframe(ticker)
                    Articles = pd.DataFrame(Articles)
                    Articles = Articles[Articles['Link'] != "https://finance.yahoo.com/news/"]
                    Articles_short = streszczenie_artykułów(Articles)
                    df_streszczenia_to_AI = Articles_short[['Streszczenie']]
                    summarized_news = summarize_news(df_streszczenia_to_AI)
                    News_links = Articles_short[['Link']]

                    if not Articles.empty:
                        st.subheader("Najnowsze wiadomości na temat firmy w pigułce:")
                        st.markdown(summarized_news)
                        with st.expander("Źródła wiadomości:"):
                            st.dataframe(News_links, use_container_width=True)

                with tabs[1]:
                    st.subheader(f'Podstawowe dane finansowe i dywidendy {company_name}')
                    # Wyświetlanie danych finansowych
                    st.markdown(f"Poniższa tabela zawiera podstawowe dane finansowe {company_name}, dane międzyokresowe są uśrednione w celu zapewnienia porównywalności do danych rocznych. Dane finansowe są przedstawione w walucie ***{currency_name}***")
                    st.dataframe(basic_fin)
                    st.markdown(f"Zmiany między poszczególnymi okresami:")
                    st.dataframe(percent_changes)
                    st.markdown(summary_fin)

                    # Wyświetlanie dywidend
                    if not dividend_df.empty:
                        st.subheader(f"Wypłaty dywidendy w {company_name}")
                        st.dataframe(dividend_df_desc, use_container_width=True)
                        # Dodanie wykresów dywidend
                        st.subheader('Dywidenda i stopa dywidendy na przestrzeni czasu')

                        fig1 = go.Figure()

                        # Wykres kwoty wypłaconej dywidendy
                        fig1.add_trace(go.Scatter(
                            x=dividend_df.index, y=dividend_df['Dividend'],
                            mode='lines+markers', name='Kwota wypłaconej dywidendy',
                            line=dict(color='red'), marker=dict(color='red', size=4)
                        ))

                        # Wykres stopy dywidendy na osobnej osi Y
                        fig1.add_trace(go.Scatter(
                            x=dividend_df.index, y=dividend_df['Dividend Yield (%)'],
                            mode='lines+markers', name='Stopa dywidendy (%)',
                            yaxis='y2', line=dict(color='blue'), marker=dict(color='blue', size=4)
                        ))

                        # Ustawienia dla podwójnej osi Y
                        fig1.update_layout(
                            xaxis_title='Data',
                            yaxis=dict(title='Kwota wypłaconej dywidendy', showgrid=False),
                            yaxis2=dict(title='Stopa dywidendy (%)', overlaying='y', side='right'),
                            title='Dywidenda i stopa dywidendy na przestrzeni czasu',
                            legend=dict(x=0.01, y=1, orientation="h"),
                            hovermode="x unified",
                            height=600
                        )

                        # Dodanie interaktywności z suwakiem dat
                        fig1.update_xaxes(rangeslider_visible=True)

                        # Automatyczne dopasowanie do zakresu danych widocznych na wykresie
                        fig1.update_yaxes(autorange=True)

                        # Wyświetlenie wykresu w Streamlit
                        st.plotly_chart(fig1)
                        st.markdown("Powyższy wykres ilustruje zmiany w wysokości wypłacanej przez firmę dywidendy oraz stopę dywidendy w dniu jej wypłaty.")



                        # Obliczenie daty 10 lat temu od dziś
                        today = datetime.datetime.today()
                        ten_years_ago = today - datetime.timedelta(days=365 * 10.5)

                        # Sprawdzenie, czy indeks 'Date' jest typu DatetimeIndex
                        if not isinstance(dividend_df.index, pd.DatetimeIndex):
                            # Resetowanie indeksu, aby 'Date' stało się zwykłą kolumną
                            dividend_df_fig2 = dividend_df.reset_index()
                            
                            # Konwersja kolumny 'Date' na typ datetime
                            dividend_df_fig2['Date'] = pd.to_datetime(dividend_df_fig2['Date'], errors='coerce', dayfirst=True)
                        else:
                            # Jeśli indeks jest DatetimeIndex, kopiujemy DataFrame i tworzymy kolumnę 'Date'
                            dividend_df_fig2 = dividend_df.copy()
                            dividend_df_fig2['Date'] = dividend_df_fig2.index

                        # Usuwanie wierszy z nieprawidłowymi datami (opcjonalnie)
                        dividend_df_fig2 = dividend_df_fig2.dropna(subset=['Date'])

                        # Filtracja danych do ostatnich 10 lat
                        dividend_df_fig2 = dividend_df_fig2[dividend_df_fig2['Date'] >= ten_years_ago]

                        # Zaokrąglenie wszystkich kolumn numerycznych do 1 miejsca po przecinku
                        dividend_df_fig2 = dividend_df_fig2.round(1)

                        # Przygotowanie DataFrame do wyświetlania w Streamlit (z sformatowaną datą jako string)
                        dividend_df_display = dividend_df_fig2.copy()
                        dividend_df_display['Date'] = dividend_df_display['Date'].dt.strftime('%d-%m-%Y')
                        dividend_df_display = dividend_df_display.set_index('Date')


                        # Tworzenie wykresu z użyciem danych filtrowanych (Date jako datetime)
                        fig2 = go.Figure()

                        # Wykres ceny zamknięcia akcji
                        fig2.add_trace(go.Scatter(
                            x=history_df['Date'], y=history_df['Close'],
                            mode='lines', name='Cena zamknięcia akcji',
                            line=dict(color='green')
                        ))

                        # Wykres kwoty wypłaconej dywidendy na osobnej osi Y
                        fig2.add_trace(go.Scatter(
                            x=dividend_df_fig2['Date'], y=dividend_df_fig2['Dividend'],
                            mode='lines+markers', name='Kwota wypłaconej dywidendy',
                            yaxis='y2', line=dict(color='red'), marker=dict(color='red', size=4)
                        ))

                        # Ustawienia dla podwójnej osi Y
                        fig2.update_layout(
                            xaxis_title='Data',
                            yaxis=dict(title='Cena zamknięcia akcji'),
                            yaxis2=dict(title='Kwota wypłaconej dywidendy', overlaying='y', side='right', showgrid=False),
                            title='Cena akcji i wypłacona dywidenda na przestrzeni czasu',
                            legend=dict(x=0.01, y=1, orientation="h"),  # Legenda na górze
                            hovermode="x unified",
                            height=600  # Zwiększamy wysokość wykresu
                        )

                        # Dodanie interaktywności z suwakiem dat
                        fig2.update_xaxes(rangeslider_visible=True)

                        # Automatyczne dopasowanie do zakresu danych widocznych na wykresie
                        fig2.update_yaxes(autorange=True)

                        # Wyświetlenie wykresu w Streamlit
                        st.plotly_chart(fig2)

                    else:
                        st.warning("Brak danych o dywidendach dla tej firmy.")

                with tabs[2]:
                    st.subheader(f'Analiza wskaźnikowa {company_name}')
                    st.dataframe(indicators_df)
                    st.markdown(summary_ind)
                    st.subheader('Podstawowe wskaźniki giełdowe')
                    st.pyplot(fig)
                    st.markdown(summary_stock_indc)

                with tabs[3]:
                    Report_summary_response = Report_summary(summary, summary_fin, summary_ind, summary_stock_indc, SWOT_summary_response, summarized_news)
                    st.subheader(f"Podsumowanie raportu inwestycyjnego dotyczącego {company_name}")
                    st.markdown(Report_summary_response)

        except Exception as e:
            st.error("Nieprawidłowy ticker")
            st.error(f"Error: {e}")
