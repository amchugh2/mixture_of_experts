#!/usr/bin/env python3
"""
Complete NoLBERT Pipeline
Runs the entire process: data collection → training → prediction
"""

import os
import sys
from datetime import datetime, timedelta
from data_pipeline import DataPipeline
from train_nolbert import NoLBERTTrainer
from predict_nolbert import NoLBERTPredictor

def run_complete_pipeline():
    """Run the complete NoLBERT pipeline"""
    
    print("🚀 NoLBERT Financial Prediction Pipeline")
    print("=" * 60)
    print("This pipeline will:")
    print("1. 📊 Collect data from Alpha Vantage")
    print("2. 🎯 Train NoLBERT model")
    print("3. 🔮 Make predictions on new data")
    print("=" * 60)
    
    # Step 1: Data Collection
    print("\n📊 STEP 1: Data Collection")
    print("-" * 30)
    
    # Initialize data pipeline
    pipeline = DataPipeline()
    
    # Set date range (last 2 years)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
    
    print(f"📅 Collecting data from {start_date} to {end_date}")
    
    # Build training dataset
    df = pipeline.build_training_dataset(start_date, end_date)
    
    if df.empty:
        print("❌ No data collected. Exiting.")
        return
    
    # Save dataset
    pipeline.save_dataset(df, "nolbert_training_data.csv")
    
    # Step 2: Model Training
    print("\n🎯 STEP 2: Model Training")
    print("-" * 30)
    
    # Check if training data exists
    json_data_path = "nolbert_training_data.json"
    if not os.path.exists(json_data_path):
        print(f"❌ Training data not found at {json_data_path}")
        return
    
    # Initialize trainer
    trainer = NoLBERTTrainer()
    
    # Load and prepare data
    texts, labels = trainer.load_data(json_data_path)
    train_dataset, val_dataset, test_dataset = trainer.prepare_datasets(texts, labels)
    
    # Train model
    print("🔥 Starting model training...")
    model_trainer = trainer.train(
        train_dataset, 
        val_dataset,
        output_dir="./trained_nolbert",
        epochs=3,
        batch_size=2,  # Reduced for memory efficiency
        warmup_steps=100,
        logging_steps=5
    )
    
    # Evaluate model
    print("📊 Evaluating trained model...")
    results = trainer.evaluate(test_dataset)
    
    # Plot results
    trainer.plot_results(results, "training_results.png")
    
    # Step 3: Prediction
    print("\n🔮 STEP 3: Making Predictions")
    print("-" * 30)
    
    # Check if trained model exists
    model_path = "./trained_nolbert"
    if not os.path.exists(model_path):
        print(f"❌ Trained model not found at {model_path}")
        return
    
    # Initialize predictor
    predictor = NoLBERTPredictor(model_path)
    
    # Test stocks for prediction
    test_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'NFLX']
    
    print(f"📈 Making predictions for {len(test_stocks)} stocks...")
    
    # Make predictions
    results_df = predictor.predict_multiple_stocks(test_stocks)
    
    if not results_df.empty:
        # Save predictions
        results_df.to_csv('final_predictions.csv', index=False)
        
        print(f"\n📋 Final Predictions Summary:")
        print(results_df.to_string(index=False))
        
        # Show top recommendations
        print(f"\n🏆 TOP RECOMMENDATIONS:")
        high_confidence = results_df[results_df['confidence'] > 0.7]
        if not high_confidence.empty:
            print("High Confidence (>70%):")
            for _, row in high_confidence.iterrows():
                print(f"  {row['ticker']}: {row['prediction']} (Confidence: {row['confidence']:.1%})")
        
        # Summary statistics
        buy_signals = results_df[results_df['prediction'] == 'BUY']
        sell_signals = results_df[results_df['prediction'] == 'SELL']
        
        print(f"\n📊 Summary:")
        print(f"  Total stocks analyzed: {len(results_df)}")
        print(f"  Buy signals: {len(buy_signals)}")
        print(f"  Sell signals: {len(sell_signals)}")
        print(f"  Average confidence: {results_df['confidence'].mean():.1%}")
        
        print(f"\n💾 Results saved to 'final_predictions.csv'")
    
    print("\n🎉 Pipeline completed successfully!")
    print("=" * 60)

def run_individual_steps():
    """Run individual steps of the pipeline"""
    
    print("🔧 Individual Pipeline Steps")
    print("=" * 40)
    print("1. Collect data only")
    print("2. Train model only")
    print("3. Make predictions only")
    print("4. Run complete pipeline")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        print("\n📊 Running data collection only...")
        pipeline = DataPipeline()
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
        df = pipeline.build_training_dataset(start_date, end_date)
        if not df.empty:
            pipeline.save_dataset(df, "nolbert_training_data.csv")
    
    elif choice == "2":
        print("\n🎯 Running model training only...")
        trainer = NoLBERTTrainer()
        texts, labels = trainer.load_data("nolbert_training_data.json")
        train_dataset, val_dataset, test_dataset = trainer.prepare_datasets(texts, labels)
        trainer.train(train_dataset, val_dataset, output_dir="./trained_nolbert")
        results = trainer.evaluate(test_dataset)
        trainer.plot_results(results)
    
    elif choice == "3":
        print("\n🔮 Running predictions only...")
        predictor = NoLBERTPredictor("./trained_nolbert")
        test_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
        results_df = predictor.predict_multiple_stocks(test_stocks)
        if not results_df.empty:
            results_df.to_csv('predictions_only.csv', index=False)
            print(results_df.to_string(index=False))
    
    elif choice == "4":
        run_complete_pipeline()
    
    else:
        print("❌ Invalid choice")

def main():
    """Main function"""
    
    print("Welcome to NoLBERT Financial Prediction System!")
    print("=" * 60)
    
    # Check if Alpha Vantage API key is set
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    if not api_key:
        print("⚠️  Warning: ALPHA_VANTAGE_API_KEY not set!")
        print("Please set your Alpha Vantage API key before running the pipeline.")
        print("You can set it in data_collection.py or as an environment variable.")
        
        proceed = input("\nDo you want to proceed anyway? (y/n): ").lower().strip()
        if proceed != 'y':
            return
    
    # Ask user what they want to do
    print("\nWhat would you like to do?")
    print("1. Run complete pipeline (data collection → training → prediction)")
    print("2. Run individual steps")
    
    choice = input("\nEnter your choice (1-2): ").strip()
    
    if choice == "1":
        run_complete_pipeline()
    elif choice == "2":
        run_individual_steps()
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main() 