from data_collection import fetch_ohlcv, compute_alpha_factors, fetch_fundamentals, fetch_news
from nolbert_experts import NewsExpert, MarketDataExpert, AlphaFactorExpert, FundamentalsExpert, model, tokenizer, device
import warnings
import pandas as pd
import numpy as np
import torch

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Summary functions ---

def summarize_ohlcv(df):
    try:
        min_val = df['Close'].min()
        max_val = df['Close'].max()
        median_val = df['Close'].median()
        min_D = (df['Close'].idxmin() - df.index[0]).days
        max_D = (df['Close'].idxmax() - df.index[0]).days
        median_D = (df['Close'].sub(median_val).abs().idxmin() - df.index[0]).days
        trend = 1 if df['Close'].iloc[-1] > df['Close'].iloc[0] else -1

        return {
            "min_val": round(min_val, 2),
            "max_val": round(max_val, 2),
            "median_val": round(median_val, 2),
            "min_D": min_D,
            "max_D": max_D,
            "median_D": median_D,
            "trend": trend
        }
    except Exception as e:
        print(f"[summarize_ohlcv] Error summarizing OHLCV: {e}")
        return {
            "min_val": "N/A", "max_val": "N/A", "median_val": "N/A",
            "min_D": "N/A", "max_D": "N/A", "median_D": "N/A", "trend": 0
        }

def summarize_alpha_factors(factors):
    summary = []
    for k, v in factors.items():
        if isinstance(v, pd.Series):
            summary.append(f"{k}: {v.mean():.4f}")
        elif isinstance(v, (float, int)):
            val = 'nan' if pd.isna(v) else f"{v:.4f}"
            summary.append(f"{k}: {val}")
        else:
            summary.append(f"{k}: unsupported type")
    return ', '.join(summary)

def summarize_fundamentals(fundamentals):
    if not fundamentals:
        return "No fundamentals."
    return ', '.join([f"{k}: {v}" for k, v in fundamentals.items()])

# --- NEW: Just return patches, not embeddings

def get_market_patches(ohlcv_df, patch_len=10):
    print("[DEBUG] Starting get_market_patches()")

    X = ohlcv_df[['Open', 'High', 'Low', 'Close', 'Volume']].values.T
    N, T = X.shape
    print(f"[DEBUG] OHLCV data shape: {X.shape} (features={N}, timesteps={T})")

    L_P = T // patch_len
    if L_P == 0:
        raise ValueError(f"Not enough data to create at least one patch of length {patch_len}")
    print(f"[DEBUG] Creating {L_P} patches with patch_len={patch_len}")

    patches = np.stack([
        X[:, i * patch_len:(i + 1) * patch_len].flatten()
        for i in range(L_P)
    ], axis=0)  # (L_P, patch_size)

    patch_tensor = patches[np.newaxis, :, :]  # (1, L_P, patch_size)
    print(f"[DEBUG] Patch tensor shape: {patch_tensor.shape}")
    return patch_tensor

# --- Main logic ---

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

    ohlcv_df = ohlcv[ticker] if isinstance(ohlcv, dict) else ohlcv

    print(f"\n=== Computing Alpha Factors ===")
    alpha_factors = compute_alpha_factors(ohlcv_df)
    print("[DEBUG] Alpha Factors:", alpha_factors)

    print(f"\n=== Fetching Fundamentals ===")
    fundamentals = fetch_fundamentals(ticker)
    print("[DEBUG] Fundamentals:", fundamentals)

    print(f"\n=== Fetching News ===")
    news_list = fetch_news(ticker)
    news_article = news_list[0] if news_list else "No news available."
    print("[DEBUG] News sample:", news_article)

    # Summarize for LLM prompts
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

    # Prepare patch tensor (raw input to reprogrammer)
    ohlcv_patches = get_market_patches(ohlcv_df, patch_len=10)
    patch_size = ohlcv_patches.shape[-1]
    market_expert.set_reprogrammer(input_dim=patch_size)

    # --- Final predictions ---
    print("\n=== Expert Predictions ===")
    print("📰 NewsExpert:", news_expert.analyze(news_article))
    print("📈 MarketDataExpert:", market_expert.analyze(ohlcv_patches, ohlcv_stats))
    print("📊 AlphaFactorExpert:", alpha_expert.analyze(alpha_factors_desc))
    print("💰 FundamentalsExpert:", fundamentals_expert.analyze(fundamentals_desc))

if __name__ == "__main__":
    main()
