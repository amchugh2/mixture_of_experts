import yfinance as yf
import pandas as pd
import numpy as np

# 1. Market Data (OHLCV)
def fetch_ohlcv(ticker, start_date, end_date):
    try:
        df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
        if df.empty:
            raise ValueError("Empty DataFrame")
        df.reset_index(inplace=True)
        return df
    except Exception as e:
        print(f"Failed to fetch OHLCV for {ticker}: {e}")
        return None  # No fallback data

# 2. Alpha Factors (examples)
def compute_alpha_factors(df):
    if df is None or df.empty:
        print("No OHLCV data available for alpha factors.")
        return None
    factors = {}
    try:
        factors['momentum_30'] = df['Close'].iloc[-1] - df['Close'].iloc[-30] if len(df) >= 30 else np.nan
        factors['volatility_10'] = df['Close'].iloc[-10:].std() if len(df) >= 10 else np.nan
        factors['mean_return_5'] = df['Close'].pct_change().iloc[-5:].mean() if len(df) >= 5 else np.nan
    except Exception as e:
        print(f"Error computing alpha factors: {e}")
        return None
    return factors

# 3. Fundamental Data (basic via yfinance)
def fetch_fundamentals(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        fundamentals = {
            'trailingPE': info.get('trailingPE'),
            'forwardPE': info.get('forwardPE'),
            'bookValue': info.get('bookValue'),
            'earningsGrowth': info.get('earningsGrowth'),
            'dividendYield': info.get('dividendYield'),
            'marketCap': info.get('marketCap'),
        }
        if all(v is None for v in fundamentals.values()):
            raise ValueError("No fundamentals fetched")
        return fundamentals
    except Exception as e:
        print(f"Failed to fetch fundamentals for {ticker}: {e}")
        return None  # No fallback data

# 4. News (now using stocknews for Yahoo news and sentiment)
import os
import requests

def fetch_news(ticker, api_key=None):
    if api_key is None:
        api_key = os.getenv("NEWSAPI_KEY")  # Make sure you export this

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": ticker,
        "sortBy": "publishedAt",
        "language": "en",
        "apiKey": api_key
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        articles = response.json().get("articles", [])
        return [
            f"{article['title']}. {article.get('description', '')}"
            for article in articles if article.get('description')
        ]
    except Exception as e:
        print(f"Failed to fetch news with NewsAPI for {ticker}: {e}")
        return ["No news available."]

# --- Example Usage ---
def main():
    ticker = 'AAPL'
    start_date = '2023-01-01'
    end_date = '2023-01-31'

    print(f"Fetching OHLCV for {ticker}...")
    ohlcv = fetch_ohlcv(ticker, start_date, end_date)
    print(ohlcv.head() if ohlcv is not None else "No OHLCV data.")

    print(f"Computing alpha factors for {ticker}...")
    alpha_factors = compute_alpha_factors(ohlcv)
    print(alpha_factors if alpha_factors is not None else "No alpha factors.")

    print(f"Fetching fundamentals for {ticker}...")
    fundamentals = fetch_fundamentals(ticker)
    print(fundamentals if fundamentals is not None else "No fundamentals.")

    print(f"Fetching news for {ticker}...")
    news = fetch_news(ticker)
    if news is not None:
        for article in news:
            print(article)
    else:
        print("No news.")

if __name__ == "__main__":
    main() 