#!/usr/bin/env python3
"""
Enhanced Deep Learning Models with Focal Loss and Weighted BCE
=============================================================

Implementation of feedback suggestions:
1. Focal loss for imbalanced data
2. Weighted binary cross-entropy
3. Advanced training techniques for fraud detection
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import logging
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FocalLoss(nn.Module):
    """
    Focal Loss implementation for imbalanced classification.
    
    Focal Loss: FL(p_t) = -α(1-p_t)^γ log(p_t)
    
    This addresses class imbalance by down-weighting easy examples
    and focusing learning on hard examples.
    """
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        """
        Args:
            alpha: Weighting factor for rare class (typically 0.25-1.0)
            gamma: Focusing parameter (typically 1-5)
            reduction: 'mean', 'sum', or 'none'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Predictions from model (before sigmoid)
            targets: Ground truth binary labels
        """
        # Apply sigmoid to get probabilities
        p = torch.sigmoid(inputs)
        
        # Calculate cross entropy
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Calculate p_t
        p_t = p * targets + (1 - p) * (1 - targets)
        
        # Calculate focal weight
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Calculate focal loss
        focal_loss = alpha_t * focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross-Entropy Loss for imbalanced data.
    
    Applies different weights to positive and negative samples.
    """
    
    def __init__(self, pos_weight=1.0):
        """
        Args:
            pos_weight: Weight for positive class (fraud)
        """
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Predictions from model (before sigmoid)
            targets: Ground truth binary labels
        """
        return F.binary_cross_entropy_with_logits(
            inputs, targets, 
            pos_weight=torch.tensor(self.pos_weight)
        )

class FraudDetectionNN(nn.Module):
    """
    Enhanced neural network for fraud detection with dropout and batch normalization.
    """
    
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout_rate=0.3):
        """
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout probability
        """
        super(FraudDetectionNN, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x).squeeze()

class FraudAutoencoder(nn.Module):
    """
    Autoencoder for anomaly detection in fraud data.
    
    Trained on normal transactions only, high reconstruction error
    indicates potential fraud.
    """
    
    def __init__(self, input_dim, encoding_dims=[64, 32, 16]):
        """
        Args:
            input_dim: Number of input features
            encoding_dims: Dimensions for encoder layers
        """
        super(FraudAutoencoder, self).__init__()
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for dim in encoding_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        for dim in reversed(encoding_dims[:-1]):
            decoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)

class EnhancedFraudDetector:
    """
    Enhanced fraud detector with multiple deep learning approaches.
    
    Implements:
    1. Focal Loss for imbalanced data
    2. Weighted BCE
    3. Autoencoder for anomaly detection
    4. Comprehensive evaluation
    """
    
    def __init__(self, device='cpu'):
        """
        Args:
            device: 'cpu' or 'cuda'
        """
        self.device = device
        self.models = {}
        self.scalers = {}
        self.training_history = {}
        self.results = {}
        
        logger.info(f"EnhancedFraudDetector initialized on device: {device}")
    
    def prepare_data(self, df, test_size=0.2):
        """
        Prepare data for training with proper scaling.
        
        Args:
            df: DataFrame with features and 'Class' column
            test_size: Fraction for test set
        """
        logger.info("Preparing data for deep learning models")
        
        # Separate features and target
        feature_columns = [col for col in df.columns if col != 'Class']
        X = df[feature_columns].values
        y = df['Class'].values
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['standard'] = scaler
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        y_test_tensor = torch.FloatTensor(y_test).to(self.device)
        
        # Calculate class weights for imbalanced data
        fraud_ratio = np.mean(y_train)
        pos_weight = (1 - fraud_ratio) / fraud_ratio
        
        logger.info(f"Training set: {len(X_train):,} samples")
        logger.info(f"Test set: {len(X_test):,} samples")
        logger.info(f"Fraud ratio: {fraud_ratio:.4f}")
        logger.info(f"Positive weight: {pos_weight:.2f}")
        
        return {
            'X_train': X_train_tensor,
            'X_test': X_test_tensor,
            'y_train': y_train_tensor,
            'y_test': y_test_tensor,
            'pos_weight': pos_weight,
            'input_dim': X_train_scaled.shape[1]
        }
    
    def train_with_focal_loss(self, data, epochs=100, batch_size=256, lr=0.001):
        """
        Train neural network with Focal Loss.
        
        Implements feedback suggestion for focal loss integration.
        """
        logger.info("Training neural network with Focal Loss")
        
        # Create model
        model = FraudDetectionNN(
            input_dim=data['input_dim'],
            hidden_dims=[256, 128, 64],
            dropout_rate=0.3
        ).to(self.device)
        
        # Focal loss with parameters tuned for fraud detection
        criterion = FocalLoss(alpha=0.75, gamma=2.0)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
        
        # Create data loaders
        train_dataset = TensorDataset(data['X_train'], data['y_train'])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        model.train()
        train_losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_loss)
            
            # Validation and scheduling
            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    val_outputs = model(data['X_test'])
                    val_loss = criterion(val_outputs, data['y_test'])
                
                scheduler.step(val_loss)
                model.train()
                
                logger.info(f"Epoch {epoch}: Train Loss = {avg_loss:.6f}, Val Loss = {val_loss:.6f}")
        
        self.models['focal_loss_nn'] = model
        self.training_history['focal_loss_nn'] = {
            'train_losses': train_losses,
            'epochs': epochs
        }
        
        # Evaluate
        self._evaluate_model(model, data, 'focal_loss_nn')
        
        logger.info("Focal Loss training completed")
    
    def train_with_weighted_bce(self, data, epochs=100, batch_size=256, lr=0.001):
        """
        Train neural network with Weighted Binary Cross-Entropy.
        
        Implements feedback suggestion for weighted BCE.
        """
        logger.info("Training neural network with Weighted BCE")
        
        # Create model
        model = FraudDetectionNN(
            input_dim=data['input_dim'],
            hidden_dims=[256, 128, 64],
            dropout_rate=0.3
        ).to(self.device)
        
        # Weighted BCE loss
        criterion = WeightedBCELoss(pos_weight=data['pos_weight'])
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
        
        # Create data loaders
        train_dataset = TensorDataset(data['X_train'], data['y_train'])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        model.train()
        train_losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_loss)
            
            # Validation and scheduling
            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    val_outputs = model(data['X_test'])
                    val_loss = criterion(val_outputs, data['y_test'])
                
                scheduler.step(val_loss)
                model.train()
                
                logger.info(f"Epoch {epoch}: Train Loss = {avg_loss:.6f}, Val Loss = {val_loss:.6f}")
        
        self.models['weighted_bce_nn'] = model
        self.training_history['weighted_bce_nn'] = {
            'train_losses': train_losses,
            'epochs': epochs
        }
        
        # Evaluate
        self._evaluate_model(model, data, 'weighted_bce_nn')
        
        logger.info("Weighted BCE training completed")
    
    def train_autoencoder(self, data, epochs=100, batch_size=256, lr=0.001):
        """
        Train autoencoder for anomaly detection.
        
        Uses only normal transactions for training.
        """
        logger.info("Training autoencoder for anomaly detection")
        
        # Filter normal transactions for training
        normal_mask = data['y_train'] == 0
        X_normal = data['X_train'][normal_mask]
        
        logger.info(f"Training autoencoder on {len(X_normal):,} normal transactions")
        
        # Create autoencoder
        autoencoder = FraudAutoencoder(
            input_dim=data['input_dim'],
            encoding_dims=[64, 32, 16]
        ).to(self.device)
        
        # MSE loss for reconstruction
        criterion = nn.MSELoss()
        optimizer = optim.Adam(autoencoder.parameters(), lr=lr, weight_decay=1e-5)
        
        # Create data loader
        normal_dataset = TensorDataset(X_normal, X_normal)  # Input = Output for autoencoder
        normal_loader = DataLoader(normal_dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        autoencoder.train()
        train_losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for batch_X, _ in normal_loader:
                optimizer.zero_grad()
                
                reconstructed = autoencoder(batch_X)
                loss = criterion(reconstructed, batch_X)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(normal_loader)
            train_losses.append(avg_loss)
            
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}: Reconstruction Loss = {avg_loss:.6f}")
        
        self.models['autoencoder'] = autoencoder
        self.training_history['autoencoder'] = {
            'train_losses': train_losses,
            'epochs': epochs
        }
        
        # Evaluate autoencoder
        self._evaluate_autoencoder(autoencoder, data)
        
        logger.info("Autoencoder training completed")
    
    def _evaluate_model(self, model, data, model_name):
        """Evaluate classification model."""
        model.eval()
        
        with torch.no_grad():
            # Predictions
            logits = model(data['X_test'])
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities > 0.5).float()
            
            # Convert to numpy
            y_true = data['y_test'].cpu().numpy()
            y_pred = predictions.cpu().numpy()
            y_prob = probabilities.cpu().numpy()
            
            # Calculate metrics
            from sklearn.metrics import (classification_report, confusion_matrix, 
                                       roc_auc_score, f1_score, precision_score, 
                                       recall_score, accuracy_score)
            
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            roc_auc = roc_auc_score(y_true, y_prob)
            
            # Confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            self.results[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'true_positives': int(tp),
                'false_positives': int(fp),
                'true_negatives': int(tn),
                'false_negatives': int(fn),
                'predictions': y_pred,
                'probabilities': y_prob
            }
            
            logger.info(f"{model_name} Results:")
            logger.info(f"  Accuracy: {accuracy:.4f}")
            logger.info(f"  Precision: {precision:.4f}")
            logger.info(f"  Recall: {recall:.4f}")
            logger.info(f"  F1-Score: {f1:.4f}")
            logger.info(f"  ROC-AUC: {roc_auc:.4f}")
    
    def _evaluate_autoencoder(self, autoencoder, data):
        """Evaluate autoencoder for anomaly detection."""
        autoencoder.eval()
        
        with torch.no_grad():
            # Reconstruction errors for all test data
            reconstructed = autoencoder(data['X_test'])
            reconstruction_errors = torch.mean((data['X_test'] - reconstructed) ** 2, dim=1)
            
            # Convert to numpy
            errors = reconstruction_errors.cpu().numpy()
            y_true = data['y_test'].cpu().numpy()
            
            # Find optimal threshold using ROC curve
            from sklearn.metrics import roc_curve, roc_auc_score
            
            fpr, tpr, thresholds = roc_curve(y_true, errors)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            
            # Predictions based on threshold
            y_pred = (errors > optimal_threshold).astype(int)
            
            # Calculate metrics
            from sklearn.metrics import (classification_report, confusion_matrix,
                                       f1_score, precision_score, recall_score, 
                                       accuracy_score)
            
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            roc_auc = roc_auc_score(y_true, errors)
            
            # Confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            self.results['autoencoder'] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'true_positives': int(tp),
                'false_positives': int(fp),
                'true_negatives': int(tn),
                'false_negatives': int(fn),
                'optimal_threshold': optimal_threshold,
                'reconstruction_errors': errors,
                'predictions': y_pred
            }
            
            logger.info("Autoencoder Results:")
            logger.info(f"  Optimal Threshold: {optimal_threshold:.6f}")
            logger.info(f"  Accuracy: {accuracy:.4f}")
            logger.info(f"  Precision: {precision:.4f}")
            logger.info(f"  Recall: {recall:.4f}")
            logger.info(f"  F1-Score: {f1:.4f}")
            logger.info(f"  ROC-AUC: {roc_auc:.4f}")
    
    def create_comparison_report(self):
        """Create comparison report for all deep learning models."""
        if not self.results:
            logger.warning("No models trained yet")
            return None
        
        comparison_data = []
        
        for model_name, metrics in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1_score']:.4f}",
                'ROC-AUC': f"{metrics['roc_auc']:.4f}",
                'True_Positives': metrics['true_positives'],
                'False_Positives': metrics['false_positives'],
                'False_Negatives': metrics['false_negatives']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by F1-score
        comparison_df['_f1_numeric'] = [float(f1) for f1 in comparison_df['F1-Score']]
        comparison_df = comparison_df.sort_values('_f1_numeric', ascending=False).drop('_f1_numeric', axis=1)
        
        logger.info("Deep Learning Model Comparison:")
        print("\n" + "="*80)
        print("DEEP LEARNING MODEL COMPARISON")
        print("="*80)
        print(comparison_df.to_string(index=False))
        print("="*80)
        
        return comparison_df
    
    def save_models(self, save_path="enhanced_deep_learning_models.pt"):
        """Save all trained models."""
        save_data = {
            'models': {name: model.state_dict() for name, model in self.models.items()},
            'scalers': self.scalers,
            'results': self.results,
            'training_history': self.training_history
        }
        
        torch.save(save_data, save_path)
        logger.info(f"Models saved to: {save_path}")

def main():
    """Main training pipeline for enhanced deep learning models."""
    logger.info("Starting Enhanced Deep Learning Fraud Detection")
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Load data
    try:
        df = pd.read_csv('creditcard.csv')
        logger.info(f"Dataset loaded: {len(df):,} transactions")
        logger.info(f"Fraud rate: {df['Class'].mean()*100:.3f}%")
    except FileNotFoundError:
        logger.error("creditcard.csv not found. Please ensure the dataset is available.")
        return
    
    # Initialize detector
    detector = EnhancedFraudDetector(device=device)
    
    # Prepare data
    data = detector.prepare_data(df, test_size=0.2)
    
    # Train models with different loss functions
    detector.train_with_focal_loss(data, epochs=80, batch_size=512, lr=0.001)
    detector.train_with_weighted_bce(data, epochs=80, batch_size=512, lr=0.001)
    detector.train_autoencoder(data, epochs=60, batch_size=512, lr=0.001)
    
    # Create comparison report
    detector.create_comparison_report()
    
    # Save models
    detector.save_models()
    
    logger.info("Enhanced deep learning training completed!")

if __name__ == "__main__":
    main()