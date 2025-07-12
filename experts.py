import random
import yfinance as yf
import requests
from openai import OpenAI

class GeneralExpert:
    """
    In prediction mode, used for stock movement prediction, the summarized reports are used to 
    construct a prompt with a prediction prefix. Given the summarized reports, the General Expert LLM 
    outputs a binary prediction indicating whether the stock will rise or fall. In ranking mode, used 
    for stock trading, the General Expert LLM functions as a comparator to establish the ranking ability.    
    """
    
    def __init__(self, mode="prediction"):
        self.mode = mode

    def predict(self, reports):
        """
        Aggregates reports and returns a directional signal (buy/sell/hold).

        Parameters:
            reports (dict): keys are expert names, values are expert outputs

        Returns:
            dict: summary and signal
        """
        positive_signals = sum(1 for r in reports.values() if r.get("signal") == "buy")
        negative_signals = sum(1 for r in reports.values() if r.get("signal") == "sell")

        if positive_signals > negative_signals:
            signal = "buy"
        elif negative_signals > positive_signals:
            signal = "sell"
        else:
            signal = "hold"

        summary = f"Buy: {positive_signals}, Sell: {negative_signals}"
        return {"summary": summary, "signal": signal}

    def rank(self, reports_a, reports_b):
        """
        Compares two sets of reports and returns which is stronger.

        Returns:
            str: 'A' or 'B'
        """
        score_a = sum(1 for r in reports_a.values() if r.get("signal") == "buy")
        score_b = sum(1 for r in reports_b.values() if r.get("signal") == "buy")

        return 'A' if score_a >= score_b else 'B'

# (1) News Expert - Siddanth wants to do this
# CHATGPT API?
class NewsExpert:
    def analyze(self, news_texts):
        positive = sum(1 for text in news_texts if "gain" in text.lower() or "rise" in text.lower())
        negative = sum(1 for text in news_texts if "fall" in text.lower() or "drop" in text.lower())
        signal = "buy" if positive > negative else "sell" if negative > positive else "hold"
        return {"summary": f"Positive: {positive}, Negative: {negative}", "signal": signal}

# Need to build this out more
def fetch_news(ticker):
    url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey=37d004d771fa096e35801095719233e8"
    resp = requests.get(url)
    if resp.status_code == 200:
        articles = resp.json().get("articles", [])
        return [article["title"] + ". " + article.get("description", "") for article in articles]
    return []

# Market Data consists of historical daily OHLCV records for
# stocks on S&P 500 list. This dataset includes a total of 481,484 records, 
# offering a detailed view of the stocks’ trading activity over the specified period.

# Need to set up chat gpt api maybe or something else
class MarketDataExpert:
    client = OpenAI()
    def analyze(self, ohlcv_df):
        min_close = ohlcv_df['close'].min()
        min_day = ohlcv_df['close'].idxmin()
        max_close = ohlcv_df['close'].max()
        max_day = ohlcv_df['close'].idxmax()
        median_close = ohlcv_df['close'].median()
        trend = 'upward' if ohlcv_df['close'].iloc[-1] > ohlcv_df['close'].iloc[0] else 'downward'

        prompt = (
            f"Statistics: The historical prices have a minimum close of {min_close:.2f} on day {min_day}, "
            f"a maximum close of {max_close:.2f} on day {max_day}, and a median close of {median_close:.2f}. "
            f"The overall trend is {trend}.\n"
            f"Question: Given the reprogrammed OHLCV data and its statistics, how is the stock expected to perform in the next 5 days?"
        )

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response.choices[0].message.content.strip()
        signal = "buy" if "Rise" in answer else "sell" if "Fall" in answer else "hold"
        return {"summary": answer, "signal": signal}

# incorporates 108 technical in- dicators and factors with their expressions, 
# which are believed to possess predictive power regarding stock price movements.
class AlphaFactorExpert:
    def analyze(self, alpha_factors):
        score = sum(alpha_factors.values())
        signal = "buy" if score > 0 else "sell" if score < 0 else "hold"
        return {"summary": f"Alpha Score: {score}", "signal": signal}

# includes earnings call, transcripts, financial statements, and fundamental metrics. 
# The earnings call transcripts are sourced from Seeking Alpha, with 16 transcripts (4 years, 
# quaterly updated) available for each stock. Funda- mental metrics include Earnings Per Share (EPS),
# Price-to-Earnings Ratio (P/E Ratio), Book Value Per Share (BVPS), etc.

# To do the market expert, they tokeninize the continuous stock data into bits, but since 
# we can use databento data, which is already discrete, we just need to tokenize t
def compute_alpha_factors(price_series):
    factors = {}
    if price_series:
        factors['momentum'] = price_series[-1] - price_series[0]
        factors['volatility'] = (max(price_series) - min(price_series)) / price_series[0] if price_series[0] != 0 else 0
    return factors

class FundamentalsExpert:
    def analyze(self, fundamentals):
        growth = fundamentals.get("earnings_growth", 0)
        debt = fundamentals.get("debt_ratio", 1)
        score = growth - debt
        signal = "buy" if score > 0 else "sell" if score < 0 else "hold"
        return {"summary": f"Growth: {growth}, Debt: {debt}", "signal": signal}

def fetch_fundamentals(ticker):
    # Placeholder static data — replace with real API calls
    return {"earnings_growth": random.uniform(-0.1, 0.3), "debt_ratio": random.uniform(0.1, 0.5)}