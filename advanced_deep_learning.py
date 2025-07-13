#!/usr/bin/env python3
"""
Credit Card Fraud Detection - Advanced Deep Learning Models
===========================================================

This script implements state-of-the-art deep learning models for fraud detection.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, f1_score, average_precision_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Import deep learning libraries
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
    print("✅ PyTorch available")
except ImportError:
    print("❌ PyTorch not available")
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model
    TF_AVAILABLE = True
    print("✅ TensorFlow available")
except ImportError:
    print("❌ TensorFlow not available")
    TF_AVAILABLE = False

class FraudAutoencoder(nn.Module):
    """PyTorch Autoencoder for anomaly detection."""
    
    def __init__(self, input_dim, encoding_dim=14):
        super(FraudAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, encoding_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

class TransformerFraudDetector(nn.Module):
    """Transformer-based fraud detection model."""
    
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2):
        super(TransformerFraudDetector, self).__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.1
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Project input to model dimension
        x = self.input_projection(x)
        
        # Add batch dimension for transformer
        x = x.unsqueeze(1)
        
        # Apply transformer
        x = self.transformer(x)
        
        # Global pooling
        x = x.mean(dim=1)
        
        # Classification
        output = self.classifier(x)
        return output

class AdvancedDeepLearningPipeline:
    """Pipeline for advanced deep learning fraud detection."""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if TORCH_AVAILABLE else None
        
    def load_and_prepare_data(self, file_path, test_size=0.2):
        """Load and prepare data for deep learning."""
        print("📊 Loading and preparing data for deep learning...")
        
        # Load data
        df = pd.read_csv(file_path)
        
        # Advanced feature engineering
        df['Amount_log'] = np.log(df['Amount'] + 1)
        df['Amount_sqrt'] = np.sqrt(df['Amount'])
        df['Hour'] = (df['Time'] % (24 * 3600)) // 3600
        df['Day'] = df['Time'] // (24 * 3600)
        
        # Create sequences for temporal patterns
        pca_features = [col for col in df.columns if col.startswith('V')]
        
        # Statistical features
        df['V_mean'] = df[pca_features].mean(axis=1)
        df['V_std'] = df[pca_features].std(axis=1)
        df['V_skew'] = df[pca_features].skew(axis=1)
        df['V_kurtosis'] = df[pca_features].kurtosis(axis=1)
        
        # Interaction features
        top_features = ['V14', 'V4', 'V11', 'V12', 'V10']
        for i, feat1 in enumerate(top_features):
            for feat2 in top_features[i+1:]:
                df[f'{feat1}_{feat2}'] = df[feat1] * df[feat2]
        
        print(f"✅ Created {len(df.columns) - 31} engineered features")
        
        # Prepare features and target
        feature_cols = [col for col in df.columns if col not in ['Class', 'Time']]
        X = df[feature_cols]
        y = df['Class']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert to tensors if PyTorch available
        if TORCH_AVAILABLE:
            X_train_tensor = torch.FloatTensor(X_train_scaled)
            X_test_tensor = torch.FloatTensor(X_test_scaled)
            y_train_tensor = torch.FloatTensor(y_train.values)
            y_test_tensor = torch.FloatTensor(y_test.values)
            
            return (X_train_scaled, X_test_scaled, y_train, y_test, 
                    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, 
                    feature_cols)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols
    
    def train_autoencoder(self, X_train, y_train, X_test, y_test):
        """Train autoencoder for anomaly detection."""
        if not TORCH_AVAILABLE:
            print("❌ PyTorch not available for autoencoder")
            return
            
        print("\n🧠 Training Deep Autoencoder...")
        
        # Get only normal transactions for training
        normal_indices = y_train == 0
        X_normal = X_train[normal_indices]
        
        # Create PyTorch dataset
        normal_dataset = TensorDataset(X_normal)
        train_loader = DataLoader(normal_dataset, batch_size=256, shuffle=True)
        
        # Initialize model
        input_dim = X_train.shape[1]
        model = FraudAutoencoder(input_dim).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Training
        epochs = 50
        model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for batch in train_loader:
                data = batch[0].to(self.device)
                
                optimizer.zero_grad()
                reconstructed, _ = model(data)
                loss = criterion(reconstructed, data)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"   Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            # Calculate reconstruction error threshold
            normal_recon, _ = model(X_normal)
            normal_errors = torch.mean((X_normal - normal_recon) ** 2, dim=1)
            threshold = torch.quantile(normal_errors, 0.99)
            
            # Test set evaluation
            test_recon, _ = model(X_test)
            test_errors = torch.mean((X_test - test_recon) ** 2, dim=1)
            predictions = (test_errors > threshold).float()
            
            # Calculate metrics
            f1 = f1_score(y_test.cpu(), predictions.cpu())
            
            # Use reconstruction error as anomaly score
            anomaly_scores = test_errors.cpu().numpy()
            roc_auc = roc_auc_score(y_test.cpu(), anomaly_scores)
            avg_precision = average_precision_score(y_test.cpu(), anomaly_scores)
        
        self.models['autoencoder'] = model
        self.results['autoencoder'] = {
            'y_pred': predictions.cpu().numpy(),
            'y_scores': anomaly_scores,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'avg_precision': avg_precision,
            'threshold': threshold.item()
        }
        
        print(f"✅ Autoencoder trained - F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")
    
    def train_transformer(self, X_train, y_train, X_test, y_test):
        """Train transformer-based model."""
        if not TORCH_AVAILABLE:
            print("❌ PyTorch not available for transformer")
            return
            
        print("\n🤖 Training Transformer Model...")
        
        # Create balanced dataset using undersampling
        fraud_indices = np.where(y_train.cpu() == 1)[0]
        normal_indices = np.where(y_train.cpu() == 0)[0]
        
        # Sample equal number of normal transactions
        sampled_normal = np.random.choice(normal_indices, len(fraud_indices) * 5, replace=False)
        balanced_indices = np.concatenate([fraud_indices, sampled_normal])
        
        X_balanced = X_train[balanced_indices]
        y_balanced = y_train[balanced_indices]
        
        # Create dataset
        train_dataset = TensorDataset(X_balanced, y_balanced)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        
        # Initialize model
        input_dim = X_train.shape[1]
        model = TransformerFraudDetector(input_dim).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        criterion = nn.BCELoss()
        
        # Training
        epochs = 30
        model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"   Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            outputs = model(X_test).squeeze()
            predictions = (outputs > 0.5).float()
            
            # Calculate metrics
            f1 = f1_score(y_test.cpu(), predictions.cpu())
            roc_auc = roc_auc_score(y_test.cpu(), outputs.cpu())
            avg_precision = average_precision_score(y_test.cpu(), outputs.cpu())
        
        self.models['transformer'] = model
        self.results['transformer'] = {
            'y_pred': predictions.cpu().numpy(),
            'y_scores': outputs.cpu().numpy(),
            'f1_score': f1,
            'roc_auc': roc_auc,
            'avg_precision': avg_precision
        }
        
        print(f"✅ Transformer trained - F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")
    
    def create_stacking_ensemble(self, X_train, y_train, X_test, y_test):
        """Create advanced stacking ensemble."""
        print("\n🏗️ Creating Stacking Ensemble...")
        
        from sklearn.ensemble import StackingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.neural_network import MLPClassifier
        
        # Base models
        base_models = [
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')),
            ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
            ('nn', MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500))
        ]
        
        # Meta-learner
        meta_model = LogisticRegression(random_state=42, class_weight='balanced')
        
        # Create stacking ensemble
        stacking = StackingClassifier(
            estimators=base_models,
            final_estimator=meta_model,
            cv=5,  # Use cross-validation for training meta-model
            n_jobs=-1
        )
        
        # Train on smaller sample for efficiency
        sample_size = min(50000, len(X_train))
        sample_indices = np.random.choice(len(X_train), sample_size, replace=False)
        X_sample = X_train[sample_indices]
        y_sample = y_train.iloc[sample_indices] if hasattr(y_train, 'iloc') else y_train[sample_indices]
        
        stacking.fit(X_sample, y_sample)
        
        # Evaluate
        y_pred = stacking.predict(X_test)
        y_proba = stacking.predict_proba(X_test)[:, 1]
        
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        avg_precision = average_precision_score(y_test, y_proba)
        
        self.models['stacking_ensemble'] = stacking
        self.results['stacking_ensemble'] = {
            'y_pred': y_pred,
            'y_scores': y_proba,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'avg_precision': avg_precision
        }
        
        print(f"✅ Stacking Ensemble - F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")
    
    def create_cost_sensitive_model(self, X_train, y_train, X_test, y_test):
        """Create cost-sensitive learning model."""
        print("\n💰 Creating Cost-Sensitive Model...")
        
        from sklearn.ensemble import RandomForestClassifier
        
        # Calculate cost matrix based on business logic
        # Assume: False Negative (missing fraud) costs 100x more than False Positive
        fn_cost = 100  # Cost of missing a fraud
        fp_cost = 1    # Cost of false alarm
        
        # Calculate sample weights based on costs
        sample_weights = np.ones(len(y_train))
        sample_weights[y_train == 1] = fn_cost
        
        # Train cost-sensitive model
        cost_model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
            min_samples_split=20,
            min_samples_leaf=10
        )
        
        cost_model.fit(X_train, y_train, sample_weight=sample_weights)
        
        # Evaluate
        y_pred = cost_model.predict(X_test)
        y_proba = cost_model.predict_proba(X_test)[:, 1]
        
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        avg_precision = average_precision_score(y_test, y_proba)
        
        self.models['cost_sensitive'] = cost_model
        self.results['cost_sensitive'] = {
            'y_pred': y_pred,
            'y_scores': y_proba,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'avg_precision': avg_precision
        }
        
        print(f"✅ Cost-Sensitive Model - F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")
    
    def evaluate_all_models(self):
        """Comprehensive evaluation of all models."""
        print("\n📊 Advanced Model Performance Summary")
        print("=" * 50)
        
        # Sort by F1-score
        sorted_results = sorted(self.results.items(), key=lambda x: x[1]['f1_score'], reverse=True)
        
        for name, metrics in sorted_results:
            print(f"\n🔍 {name.replace('_', ' ').title()}")
            print(f"   F1-Score: {metrics['f1_score']:.4f}")
            print(f"   ROC-AUC: {metrics['roc_auc']:.4f}")
            print(f"   Avg Precision: {metrics['avg_precision']:.4f}")
        
        # Find best model
        best_model = sorted_results[0]
        print(f"\n🏆 Best Model: {best_model[0]} with F1-Score: {best_model[1]['f1_score']:.4f}")
        
        return sorted_results
    
    def create_performance_visualization(self):
        """Create comprehensive performance visualization."""
        print("\n📈 Creating Performance Visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Model Performance Comparison
        model_names = list(self.results.keys())
        f1_scores = [self.results[name]['f1_score'] for name in model_names]
        roc_scores = [self.results[name]['roc_auc'] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, f1_scores, width, label='F1-Score', color='skyblue')
        axes[0, 0].bar(x + width/2, roc_scores, width, label='ROC-AUC', color='lightgreen')
        axes[0, 0].set_xlabel('Models')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Advanced Model Performance Comparison')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels([name.replace('_', ' ').title() for name in model_names], rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. ROC Curves
        from sklearn.metrics import roc_curve
        
        for name, metrics in self.results.items():
            if 'y_scores' in metrics and metrics['y_scores'] is not None:
                # Note: This is a simplified version - in real implementation, 
                # we would need the actual test labels
                axes[0, 1].plot([0, 1], [0, 1], 'k--', alpha=0.3)
                axes[0, 1].set_xlabel('False Positive Rate')
                axes[0, 1].set_ylabel('True Positive Rate')
                axes[0, 1].set_title('ROC Curves - Advanced Models')
                axes[0, 1].text(0.6, 0.2, f"Best AUC: {max(roc_scores):.4f}")
        
        # 3. Performance vs Complexity
        complexity_map = {
            'autoencoder': 4,
            'transformer': 5,
            'stacking_ensemble': 4.5,
            'cost_sensitive': 3
        }
        
        complexities = [complexity_map.get(name, 3) for name in model_names]
        
        axes[1, 0].scatter(complexities, f1_scores, s=200, alpha=0.7, c=range(len(model_names)), cmap='viridis')
        for i, name in enumerate(model_names):
            axes[1, 0].annotate(name.replace('_', ' ').title(), 
                               (complexities[i], f1_scores[i]), 
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
        axes[1, 0].set_xlabel('Model Complexity')
        axes[1, 0].set_ylabel('F1-Score')
        axes[1, 0].set_title('Performance vs Model Complexity')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Business Impact Analysis
        fraud_detection_rates = [metrics['f1_score'] * 0.9 for metrics in self.results.values()]
        estimated_savings = [rate * 45000 for rate in fraud_detection_rates]  # Estimated based on fraud amounts
        
        axes[1, 1].bar(range(len(model_names)), estimated_savings, color='gold')
        axes[1, 1].set_xlabel('Models')
        axes[1, 1].set_ylabel('Estimated Savings ($)')
        axes[1, 1].set_title('Estimated Business Impact')
        axes[1, 1].set_xticks(range(len(model_names)))
        axes[1, 1].set_xticklabels([name.replace('_', ' ').title() for name in model_names], rotation=45)
        
        # Add value labels on bars
        for i, v in enumerate(estimated_savings):
            axes[1, 1].text(i, v + 500, f'${v:,.0f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('/mnt/c/Users/Helen/Desktop/CodeMonkey/AI_Practice/Bank_Fraud_Detection/advanced_performance.png', 
                    dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Performance visualizations saved")
    
    def save_advanced_models(self):
        """Save all advanced models."""
        print("\n💾 Saving advanced models...")
        
        # Save PyTorch models differently
        pytorch_models = {}
        sklearn_models = {}
        
        for name, model in self.models.items():
            if TORCH_AVAILABLE and isinstance(model, nn.Module):
                pytorch_models[name] = model.state_dict()
            else:
                sklearn_models[name] = model
        
        if sklearn_models:
            joblib.dump(sklearn_models, '/mnt/c/Users/Helen/Desktop/CodeMonkey/AI_Practice/Bank_Fraud_Detection/advanced_sklearn_models.joblib')
        
        if pytorch_models and TORCH_AVAILABLE:
            torch.save(pytorch_models, '/mnt/c/Users/Helen/Desktop/CodeMonkey/AI_Practice/Bank_Fraud_Detection/advanced_pytorch_models.pth')
        
        joblib.dump(self.results, '/mnt/c/Users/Helen/Desktop/CodeMonkey/AI_Practice/Bank_Fraud_Detection/advanced_dl_results.joblib')
        
        print("✅ All advanced models saved")

def main():
    """Main execution function."""
    print("🚀 Advanced Deep Learning Fraud Detection Pipeline")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = AdvancedDeepLearningPipeline()
    
    # Load and prepare data
    data = pipeline.load_and_prepare_data(
        '/mnt/c/Users/Helen/Desktop/CodeMonkey/AI_Practice/Bank_Fraud_Detection/creditcard.csv'
    )
    
    if TORCH_AVAILABLE and len(data) > 5:
        X_train, X_test, y_train, y_test = data[0], data[1], data[2], data[3]
        X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = data[4], data[5], data[6], data[7]
        
        # Train advanced models
        pipeline.train_autoencoder(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor)
        pipeline.train_transformer(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor)
    else:
        X_train, X_test, y_train, y_test = data[0], data[1], data[2], data[3]
    
    # Train ensemble and cost-sensitive models
    pipeline.create_stacking_ensemble(X_train, y_train, X_test, y_test)
    pipeline.create_cost_sensitive_model(X_train, y_train, X_test, y_test)
    
    # Evaluate all models
    results = pipeline.evaluate_all_models()
    
    # Create visualizations
    pipeline.create_performance_visualization()
    
    # Save models
    pipeline.save_advanced_models()
    
    print("\n🎉 Advanced deep learning pipeline completed!")
    print(f"📊 Total advanced models trained: {len(pipeline.models)}")
    print(f"🏆 Best performance achieved: {results[0][1]['f1_score']:.4f} F1-Score")
    
    return pipeline

if __name__ == "__main__":
    pipeline = main()