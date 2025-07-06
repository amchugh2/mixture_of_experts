#!/usr/bin/env python3
"""
NoLBERT Training Script
Trains NoLBERT model on financial data for buy/sell prediction
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import os

def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    
    return {
        'eval_accuracy': accuracy,
    }

class FinancialDataset(Dataset):
    """Custom dataset for financial data"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class NoLBERTTrainer:
    """NoLBERT model trainer for financial prediction"""
    
    def __init__(self, model_name: str = "alikLab/NoLBERT", device: str = None):
        """
        Initialize the trainer
        
        Args:
            model_name: HuggingFace model name
            device: Device to use (cuda/cpu)
        """
        self.model_name = model_name
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"🚀 Initializing NoLBERT trainer on {self.device}")
        print(f"📦 Using model: {model_name}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=2  # Binary classification: buy/sell
        ).to(self.device)
        
        print(f"✅ Model loaded successfully")
    
    def load_data(self, data_path: str) -> Tuple[List[str], List[int]]:
        """
        Load training data from JSON file
        
        Args:
            data_path: Path to JSON data file
            
        Returns:
            Tuple of (texts, labels)
        """
        print(f"📂 Loading data from {data_path}")
        
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        texts = [item['text'] for item in data]
        labels = [item['label'] for item in data]
        
        print(f"✅ Loaded {len(texts)} samples")
        print(f"📊 Label distribution:")
        print(f"   Buy (1): {sum(labels)} samples")
        print(f"   Sell (0): {len(labels) - sum(labels)} samples")
        
        return texts, labels
    
    def prepare_datasets(self, texts: List[str], labels: List[int], 
                        test_size: float = 0.2, val_size: float = 0.1) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Prepare train, validation, and test datasets
        
        Args:
            texts: List of text features
            labels: List of labels
            test_size: Fraction of data for testing
            val_size: Fraction of remaining data for validation
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        print("🔧 Preparing datasets...")
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
        )
        
        # Create datasets
        train_dataset = FinancialDataset(X_train, y_train, self.tokenizer)
        val_dataset = FinancialDataset(X_val, y_val, self.tokenizer)
        test_dataset = FinancialDataset(X_test, y_test, self.tokenizer)
        
        print(f"✅ Datasets prepared:")
        print(f"   Train: {len(train_dataset)} samples")
        print(f"   Validation: {len(val_dataset)} samples")
        print(f"   Test: {len(test_dataset)} samples")
        
        return train_dataset, val_dataset, test_dataset
    
    def train(self, train_dataset: Dataset, val_dataset: Dataset, 
              output_dir: str = "./nolbert_model", **kwargs):
        """
        Train the NoLBERT model
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            output_dir: Directory to save the model
            **kwargs: Additional training arguments
        """
        print("🎯 Starting training...")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=kwargs.get('epochs', 3),
            per_device_train_batch_size=kwargs.get('batch_size', 2),  # Reduced for memory
            per_device_eval_batch_size=kwargs.get('batch_size', 2),   # Reduced for memory
            warmup_steps=kwargs.get('warmup_steps', 500),
            weight_decay=kwargs.get('weight_decay', 0.01),
            logging_dir=f"{output_dir}/logs",
            logging_steps=kwargs.get('logging_steps', 10),
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy",
            greater_is_better=True,
            save_total_limit=2,
            dataloader_num_workers=kwargs.get('num_workers', 0),  # Reduced for memory
            fp16=False,  # Disable mixed precision for MPS
            report_to=None,  # Disable wandb/tensorboard
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics,
        )
        
        # Train the model
        print("🔥 Training started...")
        trainer.train()
        
        # Save the model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"✅ Training completed! Model saved to {output_dir}")
        
        return trainer
    
    def evaluate(self, test_dataset: Dataset) -> Dict:
        """
        Evaluate the trained model
        
        Args:
            test_dataset: Test dataset
            
        Returns:
            Dictionary with evaluation metrics
        """
        print("📊 Evaluating model...")
        
        # Create data loader
        test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)  # Reduced for memory
        
        # Set model to evaluation mode
        self.model.eval()
        
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                # Force CPU for evaluation to avoid MPS issues
                input_ids = batch['input_ids'].cpu()
                attention_mask = batch['attention_mask'].cpu()
                labels = batch['labels'].cpu()
                
                # Forward pass (force CPU)
                self.model = self.model.cpu()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                # Get predictions
                preds = torch.argmax(logits, dim=-1)
                
                predictions.extend(preds.numpy())
                true_labels.extend(labels.numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        report = classification_report(true_labels, predictions, output_dict=True)
        conf_matrix = confusion_matrix(true_labels, predictions)
        
        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix,
            'predictions': predictions,
            'true_labels': true_labels
        }
        
        # Print results
        print(f"📈 Evaluation Results:")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Precision (Buy): {report['1']['precision']:.4f}")
        print(f"   Recall (Buy): {report['1']['recall']:.4f}")
        print(f"   F1-Score (Buy): {report['1']['f1-score']:.4f}")
        
        return results
    
    def plot_results(self, results: Dict, save_path: str = "training_results.png"):
        """Plot training results"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Confusion matrix
        sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', 
                   cmap='Blues', ax=ax1, cbar=False)
        ax1.set_title('Confusion Matrix')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')
        ax1.set_xticklabels(['Sell', 'Buy'])
        ax1.set_yticklabels(['Sell', 'Buy'])
        
        # Metrics bar plot
        metrics = ['Precision', 'Recall', 'F1-Score']
        buy_scores = [
            results['classification_report']['1']['precision'],
            results['classification_report']['1']['recall'],
            results['classification_report']['1']['f1-score']
        ]
        sell_scores = [
            results['classification_report']['0']['precision'],
            results['classification_report']['0']['recall'],
            results['classification_report']['0']['f1-score']
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax2.bar(x - width/2, buy_scores, width, label='Buy', color='green', alpha=0.7)
        ax2.bar(x + width/2, sell_scores, width, label='Sell', color='red', alpha=0.7)
        
        ax2.set_xlabel('Metrics')
        ax2.set_ylabel('Score')
        ax2.set_title('Classification Metrics')
        ax2.set_xticks(x)
        ax2.set_xticklabels(metrics)
        ax2.legend()
        ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Results plot saved to {save_path}")
        plt.show()

def main():
    """Main training function"""
    # Initialize trainer
    trainer = NoLBERTTrainer()
    
    # Load data
    data_path = "nolbert_training_data.json"
    if not os.path.exists(data_path):
        print(f"❌ Data file {data_path} not found!")
        print("Please run data_pipeline.py first to generate training data.")
        return
    
    texts, labels = trainer.load_data(data_path)
    
    # Prepare datasets
    train_dataset, val_dataset, test_dataset = trainer.prepare_datasets(texts, labels)
    
    # Train model
    model_trainer = trainer.train(
        train_dataset, 
        val_dataset,
        output_dir="./trained_nolbert",
        epochs=3,
        batch_size=8,
        warmup_steps=100,
        logging_steps=5
    )
    
    # Evaluate model
    results = trainer.evaluate(test_dataset)
    
    # Plot results
    trainer.plot_results(results)
    
    print("🎉 Training pipeline completed!")

if __name__ == "__main__":
    main() 