from data_collection import fetch_ohlcv, compute_alpha_factors, fetch_fundamentals, fetch_news
from nolbert_experts import NewsExpert, MarketDataExpert, AlphaFactorExpert, FundamentalsExpert, model, tokenizer, device

# Helper functions to summarize data for each expert

def summarize_ohlcv(df):
    min_close = df['Close'].min()
    max_close = df['Close'].max()
    median_close = df['Close'].median()
    trend = 'upward' if df['Close'].iloc[-1] > df['Close'].iloc[0] else 'downward'
    return f"Minimum close: {min_close}, Maximum close: {max_close}, Median close: {median_close}, Trend: {trend}."

def summarize_alpha_factors(factors):
    return ', '.join([f"{k}: {v:.4f}" for k, v in factors.items()])

def summarize_fundamentals(fundamentals):
    return ', '.join([f"{k}: {v}" for k, v in fundamentals.items()])

def main():
    ticker = 'MSFT'
    start_date = '2023-01-01'
    end_date = '2023-12-31'

    # Fetch data
    ohlcv = fetch_ohlcv(ticker, start_date, end_date)
    alpha_factors = compute_alpha_factors(ohlcv)
    fundamentals = fetch_fundamentals(ticker)
    news_list = fetch_news(ticker)

    # Prepare summaries
    ohlcv_stats = summarize_ohlcv(ohlcv)
    alpha_factors_desc = summarize_alpha_factors(alpha_factors)
    fundamentals_desc = summarize_fundamentals(fundamentals)
    news_article = news_list[0]  # Use the first news article

    # Instantiate experts
    news_expert = NewsExpert(model, tokenizer, device)
    market_expert = MarketDataExpert(model, tokenizer, device)
    alpha_expert = AlphaFactorExpert(model, tokenizer, device)
    fundamentals_expert = FundamentalsExpert(model, tokenizer, device)

    # Get predictions
    print("NewsExpert prediction:", news_expert.analyze(news_article))
    print("MarketDataExpert prediction:", market_expert.analyze(ohlcv_stats))
    print("AlphaFactorExpert prediction:", alpha_expert.analyze(alpha_factors_desc))
    print("FundamentalsExpert prediction:", fundamentals_expert.analyze(fundamentals_desc))

if __name__ == "__main__":
    main() 