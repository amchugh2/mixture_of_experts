import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn as nn
import numpy as np
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


# --- Added Reprogrammer & Patching ---
class MarketReprogrammer(nn.Module):
    def __init__(self, input_dim=50, prototype_count=32, embed_dim=128):
        super().__init__()
        self.E_prime = nn.Parameter(torch.randn(prototype_count, embed_dim))
        self.query_proj = nn.Linear(input_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, X_patch):
        Q = self.query_proj(X_patch)
        K = self.key_proj(self.E_prime)
        V = self.value_proj(self.E_prime)
        attn_scores = torch.matmul(Q, K.T) / (K.shape[-1] ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        return torch.matmul(attn_weights, V)

def create_patches(X, patch_len=10):
    N, T = X.shape
    L_P = T // patch_len
    patches = np.stack([
        X[:, i * patch_len:(i + 1) * patch_len].flatten()
        for i in range(L_P)
    ], axis=0)
    return patches[np.newaxis, :, :]

class MarketDataExpert:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.reprogrammer = MarketReprogrammer()

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
        return "Rise" if pred == 1 else "Fall"
