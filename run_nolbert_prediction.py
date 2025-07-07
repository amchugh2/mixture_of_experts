from data_collection import fetch_ohlcv, compute_alpha_factors, fetch_fundamentals, fetch_news
from nolbert_experts import NewsExpert, MarketDataExpert, AlphaFactorExpert, FundamentalsExpert, model, tokenizer, device
import warnings
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Helper functions to summarize data for each expert
def summarize_ohlcv(df):
    try:
        min_close = df['Close'].min()
        max_close = df['Close'].max()
        median_close = df['Close'].median()
        trend = 'upward' if df['Close'].iloc[-1].item() > df['Close'].iloc[0].item() else 'downward'
        return f"Minimum close: {min_close}, Maximum close: {max_close}, Median close: {median_close}, Trend: {trend}."
    except Exception as e:
        print(f"[summarize_ohlcv] Error summarizing OHLCV: {e}")
        return "OHLCV summary unavailable."

def summarize_alpha_factors(factors):
    summary = []
    for k, v in factors.items():
        if isinstance(v, pd.Series):
            summary.append(f"{k}: {v.mean():.4f}")
        elif isinstance(v, (float, int)):  # covers NaN, float, int
            val = 'nan' if pd.isna(v) else f"{v:.4f}"
            summary.append(f"{k}: {val}")
        else:
            summary.append(f"{k}: unsupported type")
    return ', '.join(summary)

def summarize_fundamentals(fundamentals):
    if not fundamentals:
        return "No fundamentals."
    return ', '.join([f"{k}: {v}" for k, v in fundamentals.items()])

def main():
    ticker = 'MSFT'
    start_date = '2010-12-01'
    end_date = '2023-12-16'

    print(f"\n=== Fetching OHLCV for {ticker} ===")
    ohlcv = fetch_ohlcv(ticker, start_date, end_date)
    if ohlcv is None:
        print("No OHLCV data available. Exiting.")
        return

    print("[DEBUG] OHLCV raw data:")
    print(ohlcv.head())

    # Make sure we are working with a DataFrame
    if isinstance(ohlcv, dict):
        ohlcv_df = ohlcv[ticker]
    else:
        ohlcv_df = ohlcv

    print(f"\n=== Computing Alpha Factors ===")
    alpha_factors = compute_alpha_factors(ohlcv_df)
    print("[DEBUG] Alpha Factors:", alpha_factors)

    print(f"\n=== Fetching Fundamentals ===")
    fundamentals = fetch_fundamentals(ticker)
    print("[DEBUG] Fundamentals:", fundamentals)

    print(f"\n=== Fetching News ===")
    news_list = fetch_news(ticker)
    if news_list:
        print("[DEBUG] News sample:", news_list[0])
        news_article = news_list[0]
    else:
        news_article = "No news available."

    # Prepare summaries
    ohlcv_stats = summarize_ohlcv(ohlcv_df)
    alpha_factors_desc = summarize_alpha_factors(alpha_factors)
    fundamentals_desc = summarize_fundamentals(fundamentals)

    print(f"\n=== Summary Inputs ===")
    print("[DEBUG] OHLCV Summary:", ohlcv_stats)
    print("[DEBUG] Alpha Summary:", alpha_factors_desc)
    print("[DEBUG] Fundamental Summary:", fundamentals_desc)

    # Instantiate experts
    news_expert = NewsExpert(model, tokenizer, device)
    market_expert = MarketDataExpert(model, tokenizer, device)
    alpha_expert = AlphaFactorExpert(model, tokenizer, device)
    fundamentals_expert = FundamentalsExpert(model, tokenizer, device)

    # Get predictions
    print("\n=== Expert Predictions ===")
    print("📰 NewsExpert:", news_expert.analyze(news_article))
    print("📈 MarketDataExpert:", market_expert.analyze(ohlcv_stats))
    print("📊 AlphaFactorExpert:", alpha_expert.analyze(alpha_factors_desc))
    print("💰 FundamentalsExpert:", fundamentals_expert.analyze(fundamentals_desc))

if __name__ == "__main__":
    main()
