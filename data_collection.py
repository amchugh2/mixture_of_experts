import pandas as pd
import numpy as np
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData
import os
from datetime import datetime, timedelta
import requests

# Alpha Vantage API Key - Set this as environment variable
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', 'YOUR_API_KEY_HERE')

if not ALPHA_VANTAGE_API_KEY:
    print("Warning: ALPHA_VANTAGE_API_KEY not set. Please set it as an environment variable.")
    print("You can get a free API key from: https://www.alphavantage.co/support/#api-key")

# Initialize Alpha Vantage clients
ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas') if ALPHA_VANTAGE_API_KEY else None
fd = FundamentalData(key=ALPHA_VANTAGE_API_KEY, output_format='pandas') if ALPHA_VANTAGE_API_KEY else None

# 1. Market Data (OHLCV)
def fetch_ohlcv(ticker, start_date=None, end_date=None):
    """
    Fetch OHLCV data using Alpha Vantage
    Note: Alpha Vantage free tier has limited daily requests
    """
    if not ts:
        print("Alpha Vantage API key not configured")
        return None
    
    try:
        # Alpha Vantage provides daily data by default
        data, meta_data = ts.get_daily(symbol=ticker, outputsize='full')
        
        if data.empty:
            raise ValueError("Empty DataFrame")
        
        # Reset index to make date a column
        data.reset_index(inplace=True)
        data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        
        # Filter by date range if provided
        if start_date:
            data = data[data['Date'] >= start_date]
        if end_date:
            data = data[data['Date'] <= end_date]
        
        # Sort by date (newest first)
        data = data.sort_values('Date', ascending=False).reset_index(drop=True)
        
        return data
        
    except Exception as e:
        print(f"Failed to fetch OHLCV for {ticker}: {e}")
        return None

# 2. Alpha Factors (computed from OHLCV data)
def compute_alpha_factors(df, ticker=None):
    """Compute technical indicators and alpha factors"""
    if df is None or df.empty:
        print("No OHLCV data available for alpha factors.")
        return None
    
    factors = {}
    try:
        # Ensure we have enough data
        if len(df) < 10:
            ticker_str = ticker if ticker else "the stock"
            print(f"Insufficient data for {ticker_str}. Need at least 10 days, got {len(df)}")
            return None
        
        # Price-based factors
        factors['current_price'] = df['Close'].iloc[0]  # Most recent price
        factors['price_change_1d'] = df['Close'].iloc[0] - df['Close'].iloc[1] if len(df) > 1 else 0
        factors['price_change_5d'] = df['Close'].iloc[0] - df['Close'].iloc[5] if len(df) > 5 else 0
        factors['price_change_10d'] = df['Close'].iloc[0] - df['Close'].iloc[10] if len(df) > 10 else 0
        
        # Return-based factors
        factors['return_1d'] = (df['Close'].iloc[0] / df['Close'].iloc[1] - 1) * 100 if len(df) > 1 else 0
        factors['return_5d'] = (df['Close'].iloc[0] / df['Close'].iloc[5] - 1) * 100 if len(df) > 5 else 0
        factors['return_10d'] = (df['Close'].iloc[0] / df['Close'].iloc[10] - 1) * 100 if len(df) > 10 else 0
        
        # Volatility factors
        factors['volatility_5d'] = df['Close'].iloc[:5].pct_change().std() * 100 if len(df) >= 5 else 0
        factors['volatility_10d'] = df['Close'].iloc[:10].pct_change().std() * 100 if len(df) >= 10 else 0
        
        # Volume factors
        factors['avg_volume_5d'] = df['Volume'].iloc[:5].mean() if len(df) >= 5 else df['Volume'].mean()
        factors['volume_ratio'] = df['Volume'].iloc[0] / factors['avg_volume_5d'] if factors['avg_volume_5d'] > 0 else 1
        
        # Technical indicators
        factors['rsi_14'] = calculate_rsi(df['Close'], min(14, len(df)))
        factors['sma_10'] = df['Close'].iloc[:10].mean() if len(df) >= 10 else df['Close'].mean()
        factors['price_vs_sma10'] = (df['Close'].iloc[0] / factors['sma_10'] - 1) * 100
        
    except Exception as e:
        print(f"Error computing alpha factors: {e}")
        return None
    
    return factors

def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index"""
    try:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[0]
    except:
        return np.nan

# 3. Fundamental Data
def fetch_fundamentals(ticker):
    """Fetch fundamental data using Alpha Vantage"""
    if not fd:
        print("Alpha Vantage API key not configured")
        return None
    
    try:
        # Get company overview
        overview, meta_data = fd.get_company_overview(symbol=ticker)
        
        if overview.empty:
            raise ValueError("No fundamental data available")
        
        # Extract key metrics with proper type conversion
        fundamentals = {}
        
        # Helper function to safely convert values
        def safe_convert(value, convert_type=float):
            if value is None or value == '':
                return None
            try:
                return convert_type(value)
            except (ValueError, TypeError):
                return None
        
        val = overview.get('Symbol', [None])
        fundamentals['symbol'] = val.iloc[0] if hasattr(val, 'iloc') else val[0]
        val = overview.get('Name', [None])
        fundamentals['name'] = val.iloc[0] if hasattr(val, 'iloc') else val[0]
        val = overview.get('Sector', [None])
        fundamentals['sector'] = val.iloc[0] if hasattr(val, 'iloc') else val[0]
        val = overview.get('Industry', [None])
        fundamentals['industry'] = val.iloc[0] if hasattr(val, 'iloc') else val[0]
        val = overview.get('MarketCapitalization', [None])
        fundamentals['market_cap'] = safe_convert(val.iloc[0] if hasattr(val, 'iloc') else val[0], float)
        val = overview.get('PERatio', [None])
        fundamentals['pe_ratio'] = safe_convert(val.iloc[0] if hasattr(val, 'iloc') else val[0], float)
        val = overview.get('ForwardPE', [None])
        fundamentals['forward_pe'] = safe_convert(val.iloc[0] if hasattr(val, 'iloc') else val[0], float)
        val = overview.get('PriceToBookRatio', [None])
        fundamentals['price_to_book'] = safe_convert(val.iloc[0] if hasattr(val, 'iloc') else val[0], float)
        val = overview.get('DividendYield', [None])
        fundamentals['dividend_yield'] = safe_convert(val.iloc[0] if hasattr(val, 'iloc') else val[0], float)
        val = overview.get('EPS', [None])
        fundamentals['eps'] = safe_convert(val.iloc[0] if hasattr(val, 'iloc') else val[0], float)
        val = overview.get('Revenue', [None])
        fundamentals['revenue'] = safe_convert(val.iloc[0] if hasattr(val, 'iloc') else val[0], float)
        val = overview.get('ProfitMargin', [None])
        fundamentals['profit_margin'] = safe_convert(val.iloc[0] if hasattr(val, 'iloc') else val[0], float)
        val = overview.get('ReturnOnEquityTTM', [None])
        fundamentals['return_on_equity'] = safe_convert(val.iloc[0] if hasattr(val, 'iloc') else val[0], float)
        val = overview.get('Beta', [None])
        fundamentals['beta'] = safe_convert(val.iloc[0] if hasattr(val, 'iloc') else val[0], float)
        
        # Clean up None values
        fundamentals = {k: v for k, v in fundamentals.items() if v is not None}
        
        return fundamentals
        
    except Exception as e:
        print(f"Failed to fetch fundamentals for {ticker}: {e}")
        return None

# 4. News and Sentiment (using direct HTTP request)
def fetch_news(ticker):
    """Fetch news and sentiment using Alpha Vantage NEWS_SENTIMENT endpoint (direct HTTP)"""
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={ALPHA_VANTAGE_API_KEY}"
    try:
        response = requests.get(url)
        data = response.json()
        if "feed" in data:
            news_list = []
            for article in data["feed"]:
                news_list.append({
                    'title': article.get('title'),
                    'summary': article.get('summary'),
                    'time_published': article.get('time_published'),
                    'sentiment': article.get('overall_sentiment_label'),
                    'sentiment_score': article.get('overall_sentiment_score'),
                    'url': article.get('url'),
                })
            print(f"✅ Fetched {len(news_list)} news articles for {ticker}")
            return news_list
        else:
            print(f"❌ No news data found for {ticker} or API limit reached.")
            return []
    except Exception as e:
        print(f"Failed to fetch news for {ticker}: {e}")
        return []

# 5. Additional Alpha Vantage data sources
def fetch_technical_indicators(ticker):
    """Fetch technical indicators from Alpha Vantage"""
    if not ts:
        print("Alpha Vantage API key not configured")
        return None
    
    try:
        indicators = {}
        
        # RSI
        rsi_data, meta_data = ts.get_rsi(symbol=ticker, interval='daily', time_period=14, series_type='close')
        if not rsi_data.empty:
            indicators['rsi'] = rsi_data.iloc[0, 0]
        
        # MACD
        macd_data, meta_data = ts.get_macd(symbol=ticker, interval='daily', series_type='close')
        if not macd_data.empty:
            indicators['macd'] = macd_data.iloc[0, 0]
            indicators['macd_signal'] = macd_data.iloc[0, 1]
            indicators['macd_hist'] = macd_data.iloc[0, 2]
        
        # Bollinger Bands
        bb_data, meta_data = ts.get_bbands(symbol=ticker, interval='daily', time_period=20, series_type='close')
        if not bb_data.empty:
            indicators['bb_upper'] = bb_data.iloc[0, 0]
            indicators['bb_middle'] = bb_data.iloc[0, 1]
            indicators['bb_lower'] = bb_data.iloc[0, 2]
        
        return indicators
        
    except Exception as e:
        print(f"Failed to fetch technical indicators for {ticker}: {e}")
        return None

# --- Example Usage ---
def main():
    # Set your API key here
    os.environ['ALPHA_VANTAGE_API_KEY'] = 'YOUR_API_KEY_HERE'
    
    ticker = 'AAPL'
    start_date = '2024-01-01'
    end_date = '2024-01-31'
    
    print(f"Fetching OHLCV for {ticker}...")
    ohlcv = fetch_ohlcv(ticker, start_date, end_date)
    if ohlcv is not None:
        print(f"OHLCV data shape: {ohlcv.shape}")
        print(ohlcv.head())
    else:
        print("No OHLCV data.")
    
    print(f"\nComputing alpha factors for {ticker}...")
    alpha_factors = compute_alpha_factors(ohlcv, ticker)
    if alpha_factors:
        print("Alpha Factors:")
        for factor, value in alpha_factors.items():
            print(f"  {factor}: {value}")
    else:
        print("No alpha factors.")
    
    print(f"\nFetching fundamentals for {ticker}...")
    fundamentals = fetch_fundamentals(ticker)
    if fundamentals:
        print("Fundamentals:")
        for metric, value in fundamentals.items():
            print(f"  {metric}: {value}")
    else:
        print("No fundamentals.")
    
    print(f"\nFetching news for {ticker}...")
    news = fetch_news(ticker)
    if news:
        print("News Articles:")
        for i, article in enumerate(news, 1):
            print(f"  {i}. {article['title']}")
            print(f"     Sentiment: {article['sentiment']}")
            print(f"     Time: {article['time_published']}")
            print()
    else:
        print("News functionality not available in current Alpha Vantage version.")
    
    print(f"\nFetching technical indicators for {ticker}...")
    tech_indicators = fetch_technical_indicators(ticker)
    if tech_indicators:
        print("Technical Indicators:")
        for indicator, value in tech_indicators.items():
            print(f"  {indicator}: {value}")
    else:
        print("No technical indicators.")

if __name__ == "__main__":
    main() 