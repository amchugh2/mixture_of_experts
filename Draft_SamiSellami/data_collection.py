import yfinance as yf
import pandas as pd
import numpy as np

# Try to import StockNews for Yahoo news and sentiment
try:
    from stocknews import StockNews
    STOCKNEWS_AVAILABLE = True
except ImportError:
    STOCKNEWS_AVAILABLE = False
    print("stocknews package not found. Install with: pip install stocknews")

# 1. Market Data (OHLCV)
def fetch_ohlcv(ticker, start_date, end_date):
    try:
        df = yf.download(ticker, start=start_date, end=end_date)
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
def fetch_news(ticker):
    if STOCKNEWS_AVAILABLE:
        try:
            sn = StockNews([ticker])
            df = sn.summarize()
            if df.empty:
                print(f"No news found for {ticker}.")
                return None
            # Return top 3 news headlines with sentiment
            news_list = []
            for _, row in df.iterrows():
                news_list.append(f"{row['stock']} | {row['date']} | {row['title']} | Sentiment: {row['sentiment']} | Summary: {row['summary']}")
            return news_list[:3] if news_list else None
        except Exception as e:
            print(f"Failed to fetch news with stocknews for {ticker}: {e}")
            return None
    # Fallback if stocknews not available
    print(f"stocknews not available or failed for {ticker}.")
    return None

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