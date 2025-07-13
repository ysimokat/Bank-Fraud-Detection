#!/usr/bin/env python3
"""
Credit Card Fraud Detection - Complete ML Pipeline
=================================================

This script implements the complete machine learning pipeline with multiple models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, precision_recall_curve, roc_curve,
                           average_precision_score, f1_score)
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

class FraudDetectionPipeline:
    """Complete fraud detection pipeline with multiple models."""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        
    def load_and_preprocess_data(self, file_path, test_size=0.2):
        """Load and preprocess the data efficiently."""
        print("📊 Loading and preprocessing data...")
        
        # Load data
        df = pd.read_csv(file_path)
        print(f"✅ Loaded {len(df):,} transactions")
        
        # Basic feature engineering
        df['Amount_log'] = np.log(df['Amount'] + 1)
        df['Hour'] = (df['Time'] % (24 * 3600)) // 3600
        
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
        
        # Convert back to DataFrames for easier handling
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols, index=X_test.index)
        
        print(f"✅ Train set: {len(X_train_scaled):,} ({y_train.sum()} frauds)")
        print(f"✅ Test set: {len(X_test_scaled):,} ({y_test.sum()} frauds)")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_baseline_models(self, X_train, y_train):
        """Train baseline models."""
        print("\n🏗️ Training Baseline Models...")
        print("=" * 50)
        
        # 1. Logistic Regression
        print("📊 Training Logistic Regression...")
        lr = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
        lr.fit(X_train, y_train)
        self.models['logistic_regression'] = lr
        print("✅ Logistic Regression trained")
        
        # 2. Random Forest
        print("🌳 Training Random Forest...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42, 
                                  class_weight='balanced', n_jobs=-1)
        rf.fit(X_train, y_train)
        self.models['random_forest'] = rf
        print("✅ Random Forest trained")
        
        # 3. Neural Network
        print("🧠 Training Neural Network...")
        nn = MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, 
                          max_iter=500, early_stopping=True)
        nn.fit(X_train, y_train)
        self.models['neural_network'] = nn
        print("✅ Neural Network trained")
    
    def train_anomaly_models(self, X_train, y_train):
        """Train anomaly detection models."""
        print("\n🔍 Training Anomaly Detection Models...")
        print("=" * 50)
        
        # Get only normal transactions for anomaly detection
        X_normal = X_train[y_train == 0]
        
        # 1. Isolation Forest
        print("🌲 Training Isolation Forest...")
        iso_forest = IsolationForest(contamination=0.002, random_state=42, n_jobs=-1)
        iso_forest.fit(X_normal)
        self.models['isolation_forest'] = iso_forest
        print("✅ Isolation Forest trained")
        
        # 2. One-Class SVM (on a sample for efficiency)
        print("🎯 Training One-Class SVM...")
        sample_size = min(5000, len(X_normal))
        X_sample = X_normal.sample(n=sample_size, random_state=42)
        
        ocsvm = OneClassSVM(gamma='scale', nu=0.002)
        ocsvm.fit(X_sample)
        self.models['one_class_svm'] = ocsvm
        print("✅ One-Class SVM trained")
    
    def train_with_smote(self, X_train, y_train):
        """Train models with SMOTE balanced data."""
        print("\n⚖️ Training with SMOTE balancing...")
        print("=" * 50)
        
        # Apply SMOTE
        smote = SMOTE(random_state=42)
        X_smote, y_smote = smote.fit_resample(X_train, y_train)
        
        print(f"📊 Original: {len(y_train)} samples ({y_train.sum()} frauds)")
        print(f"📊 SMOTE: {len(y_smote)} samples ({y_smote.sum()} frauds)")
        
        # Train models on balanced data
        rf_smote = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_smote.fit(X_smote, y_smote)
        self.models['random_forest_smote'] = rf_smote
        
        lr_smote = LogisticRegression(random_state=42, max_iter=1000)
        lr_smote.fit(X_smote, y_smote)
        self.models['logistic_regression_smote'] = lr_smote
        
        print("✅ SMOTE models trained")
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all trained models."""
        print("\n📊 Evaluating Models...")
        print("=" * 50)
        
        self.results = {}
        
        for name, model in self.models.items():
            print(f"🔍 Evaluating {name}...")
            
            try:
                if 'anomaly' in name or 'isolation' in name or 'svm' in name:
                    # Anomaly detection models
                    predictions = model.predict(X_test)
                    # Convert anomaly scores to binary predictions (1 = normal, -1 = anomaly)
                    y_pred = np.where(predictions == -1, 1, 0)
                    y_scores = None
                    if hasattr(model, 'decision_function'):
                        y_scores = -model.decision_function(X_test)  # Flip sign for anomaly scores
                else:
                    # Regular classifiers
                    y_pred = model.predict(X_test)
                    if hasattr(model, 'predict_proba'):
                        y_scores = model.predict_proba(X_test)[:, 1]
                    else:
                        y_scores = model.decision_function(X_test)
                
                # Calculate metrics
                f1 = f1_score(y_test, y_pred)
                
                if y_scores is not None:
                    auc_score = roc_auc_score(y_test, y_scores)
                    avg_precision = average_precision_score(y_test, y_scores)
                else:
                    auc_score = 0
                    avg_precision = 0
                
                # Store results
                self.results[name] = {
                    'y_pred': y_pred,
                    'y_scores': y_scores,
                    'f1_score': f1,
                    'roc_auc': auc_score,
                    'avg_precision': avg_precision
                }
                
                print(f"   F1-Score: {f1:.4f}")
                print(f"   ROC-AUC: {auc_score:.4f}")
                print(f"   Avg Precision: {avg_precision:.4f}")
                
            except Exception as e:
                print(f"   ❌ Error evaluating {name}: {str(e)}")
                continue
        
        print("✅ Model evaluation completed")
    
    def create_performance_comparison(self):
        """Create a comprehensive performance comparison."""
        print("\n📈 Creating Performance Comparison...")
        
        # Prepare data for comparison
        comparison_data = []
        for name, metrics in self.results.items():
            comparison_data.append({
                'Model': name.replace('_', ' ').title(),
                'F1-Score': metrics['f1_score'],
                'ROC-AUC': metrics['roc_auc'],
                'Avg Precision': metrics['avg_precision']
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        df_comparison = df_comparison.sort_values('F1-Score', ascending=False)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # F1-Score comparison
        sns.barplot(data=df_comparison, x='F1-Score', y='Model', ax=axes[0], palette='viridis')
        axes[0].set_title('F1-Score Comparison', fontweight='bold')
        axes[0].set_xlim(0, 1)
        
        # ROC-AUC comparison
        sns.barplot(data=df_comparison, x='ROC-AUC', y='Model', ax=axes[1], palette='plasma')
        axes[1].set_title('ROC-AUC Comparison', fontweight='bold')
        axes[1].set_xlim(0, 1)
        
        # Average Precision comparison
        sns.barplot(data=df_comparison, x='Avg Precision', y='Model', ax=axes[2], palette='cividis')
        axes[2].set_title('Average Precision Comparison', fontweight='bold')
        axes[2].set_xlim(0, 1)
        
        plt.tight_layout()
        plt.savefig('/mnt/c/Users/Helen/Desktop/CodeMonkey/AI_Practice/Bank_Fraud_Detection/model_comparison.png', 
                    dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 Performance comparison table:")
        print(df_comparison.to_string(index=False))
        
        return df_comparison
    
    def create_detailed_analysis(self, X_test, y_test):
        """Create detailed analysis plots."""
        print("\n📊 Creating Detailed Analysis...")
        
        # Get the best performing model
        best_model_name = max(self.results.keys(), 
                             key=lambda x: self.results[x]['f1_score'])
        best_model = self.models[best_model_name]
        best_results = self.results[best_model_name]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Confusion Matrix
        cm = confusion_matrix(y_test, best_results['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0])
        axes[0,0].set_title(f'Confusion Matrix - {best_model_name.title()}')
        axes[0,0].set_ylabel('True Label')
        axes[0,0].set_xlabel('Predicted Label')
        
        # 2. ROC Curve
        if best_results['y_scores'] is not None:
            fpr, tpr, _ = roc_curve(y_test, best_results['y_scores'])
            axes[0,1].plot(fpr, tpr, color='darkorange', lw=2, 
                          label=f'ROC curve (AUC = {best_results["roc_auc"]:.3f})')
            axes[0,1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            axes[0,1].set_xlim([0.0, 1.0])
            axes[0,1].set_ylim([0.0, 1.05])
            axes[0,1].set_xlabel('False Positive Rate')
            axes[0,1].set_ylabel('True Positive Rate')
            axes[0,1].set_title('ROC Curve')
            axes[0,1].legend(loc="lower right")
        
        # 3. Precision-Recall Curve
        if best_results['y_scores'] is not None:
            precision, recall, _ = precision_recall_curve(y_test, best_results['y_scores'])
            axes[1,0].plot(recall, precision, color='red', lw=2,
                          label=f'PR curve (AP = {best_results["avg_precision"]:.3f})')
            axes[1,0].set_xlabel('Recall')
            axes[1,0].set_ylabel('Precision')
            axes[1,0].set_title('Precision-Recall Curve')
            axes[1,0].legend()
        
        # 4. Feature Importance (if available)
        if hasattr(best_model, 'feature_importances_'):
            feature_names = X_test.columns
            importances = best_model.feature_importances_
            indices = np.argsort(importances)[-10:]  # Top 10 features
            
            axes[1,1].barh(range(len(indices)), importances[indices])
            axes[1,1].set_yticks(range(len(indices)))
            axes[1,1].set_yticklabels([feature_names[i] for i in indices])
            axes[1,1].set_title('Top 10 Feature Importances')
            axes[1,1].set_xlabel('Importance')
        
        plt.tight_layout()
        plt.savefig('/mnt/c/Users/Helen/Desktop/CodeMonkey/AI_Practice/Bank_Fraud_Detection/detailed_analysis.png', 
                    dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"🏆 Best performing model: {best_model_name}")
        print(f"📊 F1-Score: {best_results['f1_score']:.4f}")
        
    def save_models(self):
        """Save all trained models."""
        print("\n💾 Saving models...")
        
        # Save all models
        joblib.dump(self.models, '/mnt/c/Users/Helen/Desktop/CodeMonkey/AI_Practice/Bank_Fraud_Detection/fraud_models.joblib')
        joblib.dump(self.scaler, '/mnt/c/Users/Helen/Desktop/CodeMonkey/AI_Practice/Bank_Fraud_Detection/scaler.joblib')
        joblib.dump(self.results, '/mnt/c/Users/Helen/Desktop/CodeMonkey/AI_Practice/Bank_Fraud_Detection/model_results.joblib')
        
        print("✅ Models saved successfully")

def main():
    """Main pipeline execution."""
    print("🚀 Credit Card Fraud Detection - Complete ML Pipeline")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = FraudDetectionPipeline()
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test = pipeline.load_and_preprocess_data(
        '/mnt/c/Users/Helen/Desktop/CodeMonkey/AI_Practice/Bank_Fraud_Detection/creditcard.csv'
    )
    
    # Train all models
    pipeline.train_baseline_models(X_train, y_train)
    pipeline.train_anomaly_models(X_train, y_train)
    pipeline.train_with_smote(X_train, y_train)
    
    # Evaluate models
    pipeline.evaluate_models(X_test, y_test)
    
    # Create comparisons and analysis
    comparison_df = pipeline.create_performance_comparison()
    pipeline.create_detailed_analysis(X_test, y_test)
    
    # Save everything
    pipeline.save_models()
    
    print("\n🎉 Pipeline completed successfully!")
    print(f"📊 Total models trained: {len(pipeline.models)}")
    
    return pipeline, comparison_df

if __name__ == "__main__":
    pipeline, comparison_df = main()