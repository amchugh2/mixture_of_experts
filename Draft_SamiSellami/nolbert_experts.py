import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load NoLBERT model and tokenizer once
checkpoint = "alikLab/NoLBERT"
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(device)

# --- Expert Classes ---

class NewsExpert:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def analyze(self, news_article, days=1):
        prompt = (
            f"Instruction: You are provided with a news article. Please predict how the stock will perform in the next {days} day(s).\n"
            f"Format your response as follows: Reasoning: [Your reasoning here] Prediction: [Rise or Fall]\n"
            f"News Article: {news_article}\n"
            f"Question: Given the information in the news article above, how is the stock expected to perform in the next {days} day(s)?"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        pred = torch.argmax(logits, dim=-1).item()
        return "Rise" if pred == 1 else "Fall"

class MarketDataExpert:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def analyze(self, ohlcv_stats, days=1):
        prompt = (
            f"Instruction: You are provided with historical OHLCV data statistics. Please predict how the stock will perform in the next {days} day(s). Your response should be 'Rise' or 'Fall'.\n"
            f"Prompt: Statistics: {ohlcv_stats}\n"
            f"Question: Given the reprogrammed OHLCV data and its statistics, how is the stock expected to perform in the next {days} day(s)?"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        pred = torch.argmax(logits, dim=-1).item()
        return "Rise" if pred == 1 else "Fall"

class AlphaFactorExpert:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def analyze(self, alpha_factors_desc, days=1):
        prompt = (
            f"Instruction: You are provided with alpha factors derived from OHLCV data. Please predict the stock's movement based on the top contributing alpha factors. Your response should be 'Rise' or 'Fall'.\n"
            f"Prompt: Alpha Factors: {alpha_factors_desc}\n"
            f"Question: Based on the provided alpha factors, how is the stock expected to perform in the next {days} day(s)?"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        pred = torch.argmax(logits, dim=-1).item()
        return "Rise" if pred == 1 else "Fall"

class FundamentalsExpert:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def analyze(self, fundamentals_desc, days=1):
        prompt = (
            f"Instruction: You are provided with a summarized report of the stock's earnings call transcript and fundamental metrics. Please predict whether the stock will rise or fall in the next quarter. Your response should include a prediction in one of the following five categories: 'Strong Rise', 'Moderate Rise', 'No Change', 'Moderate Fall', or 'Strong Fall', followed by reasoning.\n"
            f"Prompt: {fundamentals_desc}\n"
            f"Question: Based on the fundamental information, will the stock rise or fall in the next quarter?"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        pred = torch.argmax(logits, dim=-1).item()
        # For demonstration, map binary output to 'Rise'/'Fall' (can be extended for 5-class if fine-tuned)
        return "Rise" if pred == 1 else "Fall"

# --- Example Usage ---
def main():
    # Example inputs (replace with real data in production)
    news_article = "Apple's Q4 earnings beat expectations, driven by strong iPhone sales and growth in services."
    ohlcv_stats = "Minimum close: 188.93, Maximum close: 197.59, Median close: 194.20, Trend: downward."
    alpha_factors_desc = "ID 26: close to min/max, ID 5: close - vwap, ID 29: stddev(high), ID 27: corr(low, adv15), ID 25: corr(high, volume)"
    fundamentals_desc = "Q4 earnings show strong performance, record iPhone and services revenue, positive outlook for next quarter."

    news_expert = NewsExpert(model, tokenizer, device)
    market_expert = MarketDataExpert(model, tokenizer, device)
    alpha_expert = AlphaFactorExpert(model, tokenizer, device)
    fundamentals_expert = FundamentalsExpert(model, tokenizer, device)

    print("NewsExpert prediction:", news_expert.analyze(news_article))
    print("MarketDataExpert prediction:", market_expert.analyze(ohlcv_stats))
    print("AlphaFactorExpert prediction:", alpha_expert.analyze(alpha_factors_desc))
    print("FundamentalsExpert prediction:", fundamentals_expert.analyze(fundamentals_desc))

if __name__ == "__main__":
    torch.manual_seed(42)
    main() 