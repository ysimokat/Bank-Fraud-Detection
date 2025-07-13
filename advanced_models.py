#!/usr/bin/env python3
"""
Credit Card Fraud Detection - Advanced Models & Explainability
=============================================================

This script implements advanced models (XGBoost, LightGBM) and explainability features.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, f1_score, average_precision_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Try to import advanced libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    print("✅ XGBoost available")
except ImportError:
    print("❌ XGBoost not available")
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
    print("✅ LightGBM available")
except ImportError:
    print("❌ LightGBM not available")
    LIGHTGBM_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
    print("✅ SHAP available")
except ImportError:
    print("❌ SHAP not available")
    SHAP_AVAILABLE = False

class AdvancedFraudDetection:
    """Advanced fraud detection with XGBoost, LightGBM, and explainability."""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        self.explainers = {}
        
    def load_and_prepare_data(self, file_path, test_size=0.2):
        """Load and prepare data with advanced feature engineering."""
        print("📊 Loading and preparing data with advanced features...")
        
        # Load data
        df = pd.read_csv(file_path)
        
        # Advanced feature engineering
        df['Amount_log'] = np.log(df['Amount'] + 1)
        df['Amount_sqrt'] = np.sqrt(df['Amount'])
        df['Hour'] = (df['Time'] % (24 * 3600)) // 3600
        df['Day'] = df['Time'] // (24 * 3600)
        
        # Interaction features with top PCA components
        df['V1_V2'] = df['V1'] * df['V2']
        df['V1_V4'] = df['V1'] * df['V4']
        df['V2_V3'] = df['V2'] * df['V3']
        df['V4_V11'] = df['V4'] * df['V11']
        df['V14_Amount'] = df['V14'] * df['Amount_log']
        
        # Statistical features
        pca_features = [col for col in df.columns if col.startswith('V')]
        df['V_mean'] = df[pca_features].mean(axis=1)
        df['V_std'] = df[pca_features].std(axis=1)
        df['V_max'] = df[pca_features].max(axis=1)
        df['V_min'] = df[pca_features].min(axis=1)
        df['V_range'] = df['V_max'] - df['V_min']
        
        print(f"✅ Created {len(df.columns) - 31} new features")
        
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
        
        # Convert back to DataFrames
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols, index=X_test.index)
        
        print(f"✅ Train set: {len(X_train_scaled):,} ({y_train.sum()} frauds)")
        print(f"✅ Test set: {len(X_test_scaled):,} ({y_test.sum()} frauds)")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols
    
    def train_xgboost(self, X_train, y_train, X_test, y_test):
        """Train XGBoost with hyperparameter optimization."""
        print("\n🚀 Training XGBoost...")
        
        # Calculate scale_pos_weight for imbalanced data
        scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
        
        # XGBoost parameters
        xgb_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'scale_pos_weight': scale_pos_weight,
            'random_state': 42,
            'n_estimators': 300,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }
        
        # Train model
        xgb_model = xgb.XGBClassifier(**xgb_params)
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=50,
            verbose=False
        )
        
        self.models['xgboost'] = xgb_model
        print("✅ XGBoost training completed")
        
        # Hyperparameter tuning (simplified for demo)
        print("🔧 XGBoost hyperparameter tuning...")
        param_grid = {
            'max_depth': [4, 6, 8],
            'learning_rate': [0.05, 0.1, 0.2],
            'n_estimators': [200, 300, 400]
        }
        
        # Use a smaller sample for grid search to save time
        sample_size = min(50000, len(X_train))
        X_sample = X_train.sample(n=sample_size, random_state=42)
        y_sample = y_train.loc[X_sample.index]
        
        grid_search = GridSearchCV(
            xgb.XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42),
            param_grid,
            cv=3,
            scoring='f1',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_sample, y_sample)
        
        # Train best model on full dataset
        best_xgb = xgb.XGBClassifier(**grid_search.best_params_, 
                                    scale_pos_weight=scale_pos_weight, 
                                    random_state=42)
        best_xgb.fit(X_train, y_train)
        
        self.models['xgboost_tuned'] = best_xgb
        print(f"✅ Best XGBoost parameters: {grid_search.best_params_}")
    
    def train_lightgbm(self, X_train, y_train):
        """Train LightGBM with optimized parameters."""
        print("\n💡 Training LightGBM...")
        
        # LightGBM parameters
        lgb_params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'random_state': 42,
            'n_estimators': 300,
            'class_weight': 'balanced'
        }
        
        # Train model
        lgb_model = lgb.LGBMClassifier(**lgb_params)
        lgb_model.fit(X_train, y_train)
        
        self.models['lightgbm'] = lgb_model
        print("✅ LightGBM training completed")
    
    def create_ensemble_model(self, X_train, y_train):
        """Create ensemble model combining multiple approaches."""
        print("\n🎭 Creating Ensemble Model...")
        
        from sklearn.ensemble import VotingClassifier
        
        # Use the best individual models
        ensemble_models = [
            ('xgboost', self.models.get('xgboost_tuned', self.models['xgboost'])),
            ('lightgbm', self.models['lightgbm'])
        ]
        
        # Create voting classifier
        ensemble = VotingClassifier(
            estimators=ensemble_models,
            voting='soft'  # Use probabilities
        )
        
        ensemble.fit(X_train, y_train)
        self.models['ensemble'] = ensemble
        print("✅ Ensemble model created")
    
    def evaluate_advanced_models(self, X_test, y_test):
        """Evaluate all advanced models."""
        print("\n📊 Evaluating Advanced Models...")
        
        advanced_models = ['xgboost', 'xgboost_tuned', 'lightgbm', 'ensemble']
        
        for name in advanced_models:
            if name in self.models:
                print(f"🔍 Evaluating {name}...")
                
                model = self.models[name]
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1]
                
                # Calculate metrics
                f1 = f1_score(y_test, y_pred)
                roc_auc = roc_auc_score(y_test, y_proba)
                avg_precision = average_precision_score(y_test, y_proba)
                
                self.results[name] = {
                    'y_pred': y_pred,
                    'y_proba': y_proba,
                    'f1_score': f1,
                    'roc_auc': roc_auc,
                    'avg_precision': avg_precision
                }
                
                print(f"   F1-Score: {f1:.4f}")
                print(f"   ROC-AUC: {roc_auc:.4f}")
                print(f"   Avg Precision: {avg_precision:.4f}")
    
    def create_shap_explanations(self, X_train, X_test, feature_cols):
        """Create SHAP explanations for model interpretability."""
        print("\n🔍 Creating SHAP Explanations...")
        
        if not SHAP_AVAILABLE:
            print("❌ SHAP not available")
            return
        
        # Focus on the best performing model
        best_model_name = 'xgboost_tuned' if 'xgboost_tuned' in self.models else 'xgboost'
        best_model = self.models[best_model_name]
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(best_model)
        
        # Calculate SHAP values for a sample of test data
        sample_size = min(1000, len(X_test))
        X_sample = X_test.sample(n=sample_size, random_state=42)
        
        print(f"📊 Calculating SHAP values for {sample_size} samples...")
        shap_values = explainer.shap_values(X_sample)
        
        self.explainers[best_model_name] = {
            'explainer': explainer,
            'shap_values': shap_values,
            'X_sample': X_sample,
            'feature_names': feature_cols
        }
        
        # Create SHAP visualizations
        plt.figure(figsize=(15, 10))
        
        # Summary plot
        plt.subplot(2, 2, 1)
        shap.summary_plot(shap_values, X_sample, feature_names=feature_cols, 
                         plot_type="bar", show=False, max_display=15)
        plt.title("SHAP Feature Importance")
        
        # Summary plot (detailed)
        plt.subplot(2, 2, 2)
        shap.summary_plot(shap_values, X_sample, feature_names=feature_cols, 
                         show=False, max_display=15)
        plt.title("SHAP Summary Plot")
        
        plt.tight_layout()
        plt.savefig('/mnt/c/Users/Helen/Desktop/CodeMonkey/AI_Practice/Bank_Fraud_Detection/shap_analysis.png', 
                    dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ SHAP explanations created")
    
    def create_feature_importance_analysis(self, feature_cols):
        """Create comprehensive feature importance analysis."""
        print("\n📈 Creating Feature Importance Analysis...")
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        
        # XGBoost feature importance
        if 'xgboost_tuned' in self.models:
            model = self.models['xgboost_tuned']
            importance = model.feature_importances_
            indices = np.argsort(importance)[-15:]
            
            axes[0, 0].barh(range(len(indices)), importance[indices])
            axes[0, 0].set_yticks(range(len(indices)))
            axes[0, 0].set_yticklabels([feature_cols[i] for i in indices])
            axes[0, 0].set_title('XGBoost Feature Importance (Top 15)')
            axes[0, 0].set_xlabel('Importance')
        
        # LightGBM feature importance
        if 'lightgbm' in self.models:
            model = self.models['lightgbm']
            importance = model.feature_importances_
            indices = np.argsort(importance)[-15:]
            
            axes[0, 1].barh(range(len(indices)), importance[indices])
            axes[0, 1].set_yticks(range(len(indices)))
            axes[0, 1].set_yticklabels([feature_cols[i] for i in indices])
            axes[0, 1].set_title('LightGBM Feature Importance (Top 15)')
            axes[0, 1].set_xlabel('Importance')
        
        # Model comparison
        comparison_data = []
        for name, metrics in self.results.items():
            comparison_data.append({
                'Model': name.replace('_', ' ').title(),
                'F1-Score': metrics['f1_score'],
                'ROC-AUC': metrics['roc_auc']
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # F1-Score comparison
        sns.barplot(data=df_comparison, x='F1-Score', y='Model', ax=axes[1, 0])
        axes[1, 0].set_title('Advanced Models F1-Score Comparison')
        
        # ROC-AUC comparison
        sns.barplot(data=df_comparison, x='ROC-AUC', y='Model', ax=axes[1, 1])
        axes[1, 1].set_title('Advanced Models ROC-AUC Comparison')
        
        plt.tight_layout()
        plt.savefig('/mnt/c/Users/Helen/Desktop/CodeMonkey/AI_Practice/Bank_Fraud_Detection/advanced_models_analysis.png', 
                    dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Feature importance analysis completed")
    
    def save_advanced_models(self):
        """Save all advanced models and explanations."""
        print("\n💾 Saving advanced models...")
        
        # Save models
        joblib.dump(self.models, '/mnt/c/Users/Helen/Desktop/CodeMonkey/AI_Practice/Bank_Fraud_Detection/advanced_models.joblib')
        joblib.dump(self.results, '/mnt/c/Users/Helen/Desktop/CodeMonkey/AI_Practice/Bank_Fraud_Detection/advanced_results.joblib')
        joblib.dump(self.explainers, '/mnt/c/Users/Helen/Desktop/CodeMonkey/AI_Practice/Bank_Fraud_Detection/explainers.joblib')
        
        print("✅ Advanced models saved successfully")

def main():
    """Main advanced modeling pipeline."""
    print("🚀 Advanced Credit Card Fraud Detection Pipeline")
    print("=" * 60)
    
    # Initialize advanced pipeline
    advanced_pipeline = AdvancedFraudDetection()
    
    # Load and prepare data
    X_train, X_test, y_train, y_test, feature_cols = advanced_pipeline.load_and_prepare_data(
        '/mnt/c/Users/Helen/Desktop/CodeMonkey/AI_Practice/Bank_Fraud_Detection/creditcard.csv'
    )
    
    # Train advanced models
    if XGBOOST_AVAILABLE:
        advanced_pipeline.train_xgboost(X_train, y_train, X_test, y_test)
    
    if LIGHTGBM_AVAILABLE:
        advanced_pipeline.train_lightgbm(X_train, y_train)
    
    # Create ensemble
    advanced_pipeline.create_ensemble_model(X_train, y_train)
    
    # Evaluate models
    advanced_pipeline.evaluate_advanced_models(X_test, y_test)
    
    # Create explanations
    if SHAP_AVAILABLE:
        advanced_pipeline.create_shap_explanations(X_train, X_test, feature_cols)
    
    # Feature importance analysis
    advanced_pipeline.create_feature_importance_analysis(feature_cols)
    
    # Save everything
    advanced_pipeline.save_advanced_models()
    
    print("\n🎉 Advanced pipeline completed successfully!")
    print(f"📊 Total advanced models trained: {len(advanced_pipeline.models)}")
    
    return advanced_pipeline

if __name__ == "__main__":
    advanced_pipeline = main()