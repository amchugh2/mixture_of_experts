#!/usr/bin/env python3
"""
NoLBERT Prediction Script
Uses trained NoLBERT model to predict buy/sell signals for new data
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
from data_collection import fetch_ohlcv, compute_alpha_factors, fetch_fundamentals, fetch_news
from data_pipeline import DataPipeline
import json
from typing import Dict, List, Tuple
import os

class NoLBERTPredictor:
    """NoLBERT model for financial prediction"""
    
    def __init__(self, model_path: str = "./trained_nolbert", device: str = None):
        """
        Initialize the predictor
        
        Args:
            model_path: Path to trained model
            device: Device to use (cuda/cpu)
        """
        self.model_path = model_path
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"🚀 Loading trained NoLBERT model from {model_path}")
        print(f"💻 Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        print("✅ Model loaded successfully")
    
    def prepare_features(self, ticker: str, start_date: str = None, end_date: str = None) -> str:
        """
        Prepare features for a single stock (same as training pipeline)
        
        Args:
            ticker: Stock symbol
            start_date: Start date for data collection
            end_date: End date for data collection
            
        Returns:
            Text representation of features
        """
        try:
            print(f"📊 Preparing features for {ticker}...")
            
            # Fetch data using Alpha Vantage
            ohlcv = fetch_ohlcv(ticker, start_date, end_date)
            if ohlcv is None or ohlcv.empty:
                raise ValueError(f"No OHLCV data available for {ticker}")
            
            fundamentals = fetch_fundamentals(ticker)
            news_data = fetch_news(ticker)
            alpha_factors = compute_alpha_factors(ohlcv, ticker)
            
            # Create text features (same format as training)
            text_features = self._create_text_features(
                ticker, ohlcv, alpha_factors, fundamentals, news_data
            )
            
            return text_features
            
        except Exception as e:
            print(f"❌ Error preparing features for {ticker}: {e}")
            return None
    
    def _create_text_features(self, ticker: str, ohlcv: pd.DataFrame, 
                            alpha_factors: Dict, fundamentals: Dict, 
                            news_data: List) -> str:
        """Create text representation of features"""
        text_parts = []
        
        # Basic info
        text_parts.append(f"Stock: {ticker}")
        text_parts.append(f"Current Date: {pd.Timestamp.now().strftime('%Y-%m')}")
        
        # OHLCV features (most recent data)
        if not ohlcv.empty:
            recent_data = ohlcv.iloc[:10]  # Last 10 days
            text_parts.append(f"Recent OHLCV - Open: ${recent_data['Open'].iloc[0]:.2f}, High: ${recent_data['High'].max():.2f}, Low: ${recent_data['Low'].min():.2f}, Close: ${recent_data['Close'].iloc[0]:.2f}")
            text_parts.append(f"Volume: {recent_data['Volume'].sum():,.0f}, Avg Volume: {recent_data['Volume'].mean():,.0f}")
            text_parts.append(f"Price Volatility: {recent_data['Close'].std():.4f}")
        
        # Alpha factors
        if alpha_factors:
            alpha_text = "Alpha Factors: "
            alpha_parts = []
            for factor, value in alpha_factors.items():
                if isinstance(value, (int, float)) and not pd.isna(value):
                    alpha_parts.append(f"{factor}: {value:.4f}")
            if alpha_parts:
                text_parts.append(alpha_text + ", ".join(alpha_parts[:5]))
        
        # Fundamentals
        if fundamentals:
            fund_text = "Fundamentals: "
            fund_parts = []
            for metric, value in fundamentals.items():
                if isinstance(value, (int, float)) and not pd.isna(value):
                    fund_parts.append(f"{metric}: {value}")
            if fund_parts:
                text_parts.append(fund_text + ", ".join(fund_parts[:5]))
        
        # News sentiment
        if news_data:
            positive_count = sum(1 for article in news_data if isinstance(article, dict) and 
                               'positive' in article.get('sentiment', '').lower())
            negative_count = sum(1 for article in news_data if isinstance(article, dict) and 
                               'negative' in article.get('sentiment', '').lower())
            text_parts.append(f"News - Positive: {positive_count}, Negative: {negative_count}, Total: {len(news_data)}")
        
        # Technical indicators
        if not ohlcv.empty:
            price_momentum = (ohlcv['Close'].iloc[0] - ohlcv['Close'].iloc[5]) / ohlcv['Close'].iloc[5] if len(ohlcv) > 5 else 0
            volume_momentum = (ohlcv['Volume'].iloc[0] - ohlcv['Volume'].iloc[5]) / ohlcv['Volume'].iloc[5] if len(ohlcv) > 5 and ohlcv['Volume'].iloc[5] > 0 else 0
            text_parts.append(f"Price Momentum: {price_momentum:.4f}")
            text_parts.append(f"Volume Momentum: {volume_momentum:.4f}")
        
        return ". ".join(text_parts)
    
    def predict(self, text_features: str) -> Dict:
        """
        Make prediction using the trained model
        
        Args:
            text_features: Text representation of features
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Tokenize the text
            inputs = self.tokenizer(
                text_features,
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors='pt'
            )
            
            # Move inputs to device
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                
                # Get prediction and confidence
                prediction = torch.argmax(logits, dim=-1).item()
                confidence = probabilities[0][prediction].item()
                
                # Get probabilities for both classes
                buy_prob = probabilities[0][1].item()
                sell_prob = probabilities[0][0].item()
            
            result = {
                'prediction': 'BUY' if prediction == 1 else 'SELL',
                'confidence': confidence,
                'buy_probability': buy_prob,
                'sell_probability': sell_prob,
                'prediction_code': prediction
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Error making prediction: {e}")
            return None
    
    def predict_stock(self, ticker: str, start_date: str = None, end_date: str = None) -> Dict:
        """
        Complete prediction pipeline for a single stock
        
        Args:
            ticker: Stock symbol
            start_date: Start date for data collection
            end_date: End date for data collection
            
        Returns:
            Dictionary with prediction results and data summary
        """
        print(f"🔍 Making prediction for {ticker}")
        print("=" * 50)
        
        # Prepare features
        text_features = self.prepare_features(ticker, start_date, end_date)
        
        if text_features is None:
            return {'error': 'Failed to prepare features'}
        
        # Make prediction
        prediction_result = self.predict(text_features)
        
        if prediction_result is None:
            return {'error': 'Failed to make prediction'}
        
        # Add additional info
        result = {
            'ticker': ticker,
            'prediction_date': pd.Timestamp.now().strftime('%Y-%m-%d'),
            'text_features': text_features[:200] + "..." if len(text_features) > 200 else text_features,
            **prediction_result
        }
        
        # Print results
        print(f"📊 Prediction Results for {ticker}:")
        print(f"   Signal: {result['prediction']}")
        print(f"   Confidence: {result['confidence']:.2%}")
        print(f"   Buy Probability: {result['buy_probability']:.2%}")
        print(f"   Sell Probability: {result['sell_probability']:.2%}")
        
        # Provide recommendation
        if result['confidence'] > 0.7:
            confidence_level = "HIGH"
        elif result['confidence'] > 0.5:
            confidence_level = "MEDIUM"
        else:
            confidence_level = "LOW"
        
        print(f"\n🎯 RECOMMENDATION: {result['prediction']} (Confidence: {confidence_level})")
        
        return result
    
    def predict_multiple_stocks(self, tickers: List[str], start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Predict for multiple stocks
        
        Args:
            tickers: List of stock symbols
            start_date: Start date for data collection
            end_date: End date for data collection
            
        Returns:
            DataFrame with predictions for all stocks
        """
        print(f"🚀 Making predictions for {len(tickers)} stocks")
        print("=" * 60)
        
        results = []
        
        for i, ticker in enumerate(tickers, 1):
            print(f"\n[{i}/{len(tickers)}] Processing {ticker}...")
            
            result = self.predict_stock(ticker, start_date, end_date)
            
            if 'error' not in result:
                results.append({
                    'ticker': result['ticker'],
                    'prediction': result['prediction'],
                    'confidence': result['confidence'],
                    'buy_probability': result['buy_probability'],
                    'sell_probability': result['sell_probability'],
                    'prediction_date': result['prediction_date']
                })
            
            # Add delay to respect API limits
            import time
            time.sleep(1)
        
        # Create DataFrame
        if results:
            df = pd.DataFrame(results)
            
            # Sort by confidence
            df = df.sort_values('confidence', ascending=False)
            
            print(f"\n📊 Summary of Predictions:")
            print(f"   Total stocks analyzed: {len(df)}")
            print(f"   Buy signals: {len(df[df['prediction'] == 'BUY'])}")
            print(f"   Sell signals: {len(df[df['prediction'] == 'SELL'])}")
            print(f"   Average confidence: {df['confidence'].mean():.2%}")
            
            return df
        else:
            print("❌ No predictions made")
            return pd.DataFrame()

def main():
    """Example usage"""
    # Initialize predictor
    model_path = "./trained_nolbert"
    
    if not os.path.exists(model_path):
        print(f"❌ Trained model not found at {model_path}")
        print("Please run train_nolbert.py first to train the model.")
        return
    
    predictor = NoLBERTPredictor(model_path)
    
    # Example 1: Predict single stock
    print("\n" + "="*60)
    print("EXAMPLE 1: Single Stock Prediction")
    print("="*60)
    
    result = predictor.predict_stock('AAPL')
    if 'error' not in result:
        print(f"\n✅ Prediction completed for AAPL")
    
    # Example 2: Predict multiple stocks
    print("\n" + "="*60)
    print("EXAMPLE 2: Multiple Stocks Prediction")
    print("="*60)
    
    test_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
    results_df = predictor.predict_multiple_stocks(test_stocks)
    
    if not results_df.empty:
        print(f"\n📋 Detailed Results:")
        print(results_df.to_string(index=False))
        
        # Save results
        results_df.to_csv('nolbert_predictions.csv', index=False)
        print(f"\n💾 Results saved to nolbert_predictions.csv")

if __name__ == "__main__":
    main() 