#!/usr/bin/env python3
"""
Data Pipeline for NoLBERT Training
Fetches monthly data from Alpha Vantage and prepares it for training
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data_collection import fetch_ohlcv, compute_alpha_factors, fetch_fundamentals, fetch_news
import os
import json
from typing import List, Dict, Tuple, Optional

class DataPipeline:
    def __init__(self, api_key: str = None):
        """
        Initialize the data pipeline
        
        Args:
            api_key: Alpha Vantage API key (optional, can be set as env var)
        """
        if api_key:
            os.environ['ALPHA_VANTAGE_API_KEY'] = api_key
        
        # List of stocks to analyze (reduced to 5 for memory efficiency)
        self.stocks = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'
        ]
        
        # Training data storage
        self.training_data = []
        
    def get_monthly_data(self, ticker: str, start_date: str, end_date: str) -> Optional[Dict]:
        """
        Fetch and prepare monthly data for a single stock
        
        Args:
            ticker: Stock symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Dictionary with monthly data and features
        """
        try:
            print(f"📊 Fetching data for {ticker}...")
            
            # Fetch OHLCV data
            ohlcv = fetch_ohlcv(ticker, start_date, end_date)
            if ohlcv is None or ohlcv.empty:
                print(f"❌ No OHLCV data for {ticker}")
                return None
            
            # Fetch fundamental data
            fundamentals = fetch_fundamentals(ticker)
            
            # Fetch news data (if available)
            news_data = fetch_news(ticker)
            
            # Compute alpha factors
            alpha_factors = compute_alpha_factors(ohlcv, ticker)
            
            # Prepare monthly features
            monthly_features = self._prepare_monthly_features(
                ohlcv, alpha_factors, fundamentals, news_data, ticker
            )
            
            if monthly_features:
                print(f"✅ Successfully prepared data for {ticker}")
                return monthly_features
            else:
                print(f"❌ Failed to prepare features for {ticker}")
                return None
                
        except Exception as e:
            print(f"❌ Error processing {ticker}: {e}")
            return None
    
    def _prepare_monthly_features(self, ohlcv: pd.DataFrame, alpha_factors: Dict, 
                                fundamentals: Dict, news_data: List, ticker: str) -> Optional[Dict]:
        """
        Prepare monthly features from raw data
        """
        try:
            # Group OHLCV data by month
            ohlcv['Date'] = pd.to_datetime(ohlcv['Date'])
            ohlcv['YearMonth'] = ohlcv['Date'].dt.to_period('M')
            
            monthly_data = []
            
            for year_month in ohlcv['YearMonth'].unique():
                month_data = ohlcv[ohlcv['YearMonth'] == year_month]
                
                if len(month_data) < 5:  # Need at least 5 trading days
                    continue
                
                # Calculate monthly return (for labeling)
                month_start_price = month_data['Close'].iloc[-1]  # First day of month
                month_end_price = month_data['Close'].iloc[0]     # Last day of month
                monthly_return = (month_end_price - month_start_price) / month_start_price
                
                # Create label: 1 for buy (positive return), 0 for sell (negative return)
                label = 1 if monthly_return > 0 else 0
                
                # Prepare features
                features = {
                    'ticker': ticker,
                    'year_month': str(year_month),
                    'monthly_return': monthly_return,
                    'label': label,
                    
                    # OHLCV features
                    'month_open': month_data['Open'].iloc[-1],
                    'month_high': month_data['High'].max(),
                    'month_low': month_data['Low'].min(),
                    'month_close': month_data['Close'].iloc[0],
                    'month_volume': month_data['Volume'].sum(),
                    'month_avg_volume': month_data['Volume'].mean(),
                    'month_price_volatility': month_data['Close'].std(),
                    
                    # Alpha factors (if available)
                    'alpha_factors': alpha_factors if alpha_factors else {},
                    
                    # Fundamental features (if available)
                    'fundamentals': fundamentals if fundamentals else {},
                    
                    # News features (if available)
                    'news_count': len(news_data) if news_data else 0,
                    'news_sentiment': self._calculate_news_sentiment(news_data),
                    
                    # Technical indicators
                    'price_momentum': (month_data['Close'].iloc[0] - month_data['Close'].iloc[-1]) / month_data['Close'].iloc[-1],
                    'volume_momentum': (month_data['Volume'].iloc[0] - month_data['Volume'].iloc[-1]) / month_data['Volume'].iloc[-1] if month_data['Volume'].iloc[-1] > 0 else 0,
                }
                
                monthly_data.append(features)
            
            return monthly_data
            
        except Exception as e:
            print(f"Error preparing monthly features for {ticker}: {e}")
            return None
    
    def _calculate_news_sentiment(self, news_data: List) -> Dict:
        """Calculate news sentiment features"""
        if not news_data:
            return {'positive': 0, 'negative': 0, 'neutral': 0, 'avg_sentiment': 0}
        
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        sentiment_scores = []
        
        for article in news_data:
            if isinstance(article, dict):
                sentiment = article.get('sentiment', 'Neutral').lower()
                sentiment_score = article.get('sentiment_score', 0)
                
                if 'positive' in sentiment or 'bullish' in sentiment:
                    positive_count += 1
                elif 'negative' in sentiment or 'bearish' in sentiment:
                    negative_count += 1
                else:
                    neutral_count += 1
                
                if isinstance(sentiment_score, (int, float)):
                    sentiment_scores.append(sentiment_score)
        
        return {
            'positive': positive_count,
            'negative': negative_count,
            'neutral': neutral_count,
            'avg_sentiment': np.mean(sentiment_scores) if sentiment_scores else 0
        }
    
    def build_training_dataset(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Build complete training dataset for all stocks
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with all training data
        """
        print(f"🚀 Building training dataset from {start_date} to {end_date}")
        print(f"📈 Processing {len(self.stocks)} stocks...")
        
        all_data = []
        
        for i, ticker in enumerate(self.stocks, 1):
            print(f"\n[{i}/{len(self.stocks)}] Processing {ticker}...")
            
            monthly_data = self.get_monthly_data(ticker, start_date, end_date)
            
            if monthly_data:
                all_data.extend(monthly_data)
            
            # Add delay to respect API limits
            import time
            time.sleep(0.5)
        
        # Convert to DataFrame
        if all_data:
            df = pd.DataFrame(all_data)
            print(f"\n✅ Training dataset created with {len(df)} samples")
            print(f"📊 Label distribution:")
            print(f"   Buy (1): {len(df[df['label'] == 1])} samples")
            print(f"   Sell (0): {len(df[df['label'] == 0])} samples")
            return df
        else:
            print("❌ No data collected")
            return pd.DataFrame()
    
    def save_dataset(self, df: pd.DataFrame, filename: str = "training_dataset.csv"):
        """Save the training dataset"""
        df.to_csv(filename, index=False)
        print(f"💾 Dataset saved to {filename}")
        
        # Also save as JSON for NoLBERT
        json_filename = filename.replace('.csv', '.json')
        self._save_as_json(df, json_filename)
    
    def _save_as_json(self, df: pd.DataFrame, filename: str):
        """Save dataset in JSON format for NoLBERT"""
        data = []
        
        for _, row in df.iterrows():
            # Prepare text representation for NoLBERT
            text_features = self._create_text_features(row)
            
            data.append({
                'text': text_features,
                'label': int(row['label']),
                'ticker': row['ticker'],
                'year_month': row['year_month'],
                'monthly_return': float(row['monthly_return'])
            })
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"💾 JSON dataset saved to {filename}")
    
    def _create_text_features(self, row: pd.Series) -> str:
        """Create text representation of features for NoLBERT"""
        text_parts = []
        
        # Basic info
        text_parts.append(f"Stock: {row['ticker']}")
        text_parts.append(f"Period: {row['year_month']}")
        
        # OHLCV features
        text_parts.append(f"Monthly OHLCV - Open: ${row['month_open']:.2f}, High: ${row['month_high']:.2f}, Low: ${row['month_low']:.2f}, Close: ${row['month_close']:.2f}")
        text_parts.append(f"Volume: {row['month_volume']:,.0f}, Avg Volume: {row['month_avg_volume']:,.0f}")
        text_parts.append(f"Price Volatility: {row['month_price_volatility']:.4f}")
        
        # Alpha factors
        if row['alpha_factors']:
            alpha_text = "Alpha Factors: "
            alpha_parts = []
            for factor, value in row['alpha_factors'].items():
                if isinstance(value, (int, float)) and not pd.isna(value):
                    alpha_parts.append(f"{factor}: {value:.4f}")
            if alpha_parts:
                text_parts.append(alpha_text + ", ".join(alpha_parts[:5]))  # Limit to 5 factors
        
        # Fundamentals
        if row['fundamentals']:
            fund_text = "Fundamentals: "
            fund_parts = []
            for metric, value in row['fundamentals'].items():
                if isinstance(value, (int, float)) and not pd.isna(value):
                    fund_parts.append(f"{metric}: {value}")
            if fund_parts:
                text_parts.append(fund_text + ", ".join(fund_parts[:5]))  # Limit to 5 metrics
        
        # News sentiment
        if row['news_sentiment']:
            sentiment = row['news_sentiment']
            text_parts.append(f"News - Positive: {sentiment['positive']}, Negative: {sentiment['negative']}, Neutral: {sentiment['neutral']}")
        
        # Technical indicators
        text_parts.append(f"Price Momentum: {row['price_momentum']:.4f}")
        text_parts.append(f"Volume Momentum: {row['volume_momentum']:.4f}")
        
        return ". ".join(text_parts)

def main():
    """Example usage"""
    # Initialize pipeline
    pipeline = DataPipeline()
    
    # Build dataset for the last 2 years
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
    
    # Build training dataset
    df = pipeline.build_training_dataset(start_date, end_date)
    
    if not df.empty:
        # Save dataset
        pipeline.save_dataset(df, "nolbert_training_data.csv")
        
        # Display sample
        print("\n📋 Sample data:")
        print(df.head())
        
        print(f"\n📊 Dataset Statistics:")
        print(f"Total samples: {len(df)}")
        print(f"Unique stocks: {df['ticker'].nunique()}")
        print(f"Date range: {df['year_month'].min()} to {df['year_month'].max()}")
        print(f"Average monthly return: {df['monthly_return'].mean():.4f}")
        print(f"Return std: {df['monthly_return'].std():.4f}")

if __name__ == "__main__":
    main() 