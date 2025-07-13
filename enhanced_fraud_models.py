#!/usr/bin/env python3
"""
Enhanced Fraud Detection Models with Cost-Sensitive Metrics
===========================================================

Implementation of feedback suggestions:
1. Cost-sensitive metrics for fraud vs non-fraud
2. Early stopping & cross-validation metrics in logs
3. Comprehensive model comparison with cost-benefit analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import (train_test_split, cross_val_score, 
                                   StratifiedKFold, validation_curve)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve, 
                           precision_recall_curve, roc_auc_score, f1_score, 
                           precision_score, recall_score, accuracy_score,
                           make_scorer, fbeta_score, cohen_kappa_score)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
import xgboost as xgb
import lightgbm as lgb

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CostSensitiveFraudDetector:
    """
    Enhanced fraud detector with cost-sensitive metrics and comprehensive evaluation.
    
    Implements feedback suggestions:
    - Cost-sensitive metrics (different costs for FP vs FN)
    - Early stopping with cross-validation
    - Detailed logging and model comparison
    """
    
    def __init__(self, fraud_cost=100, false_positive_cost=5, investigation_cost=25):
        """
        Initialize with business costs.
        
        Args:
            fraud_cost: Cost of missing a fraud (false negative)
            false_positive_cost: Cost of false alarm (false positive)  
            investigation_cost: Cost to investigate each alert
        """
        self.fraud_cost = fraud_cost
        self.false_positive_cost = false_positive_cost
        self.investigation_cost = investigation_cost
        
        self.models = {}
        self.scaler = None
        self.results = {}
        self.cost_matrices = {}
        self.cv_scores = {}
        
        # Setup cost-sensitive scoring
        self.cost_sensitive_scorer = make_scorer(self._cost_sensitive_score, greater_is_better=False)
        
        logger.info(f"Initialized CostSensitiveFraudDetector:")
        logger.info(f"  - Fraud cost (FN): ${fraud_cost}")
        logger.info(f"  - False positive cost (FP): ${false_positive_cost}")
        logger.info(f"  - Investigation cost: ${investigation_cost}")
    
    def _cost_sensitive_score(self, y_true, y_pred):
        """
        Calculate cost-sensitive score based on business costs.
        
        Returns negative cost (lower is better for sklearn scorers).
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Calculate total cost
        fraud_loss_cost = fn * self.fraud_cost  # Missed frauds
        false_alarm_cost = fp * self.false_positive_cost  # False alarms
        investigation_cost = (tp + fp) * self.investigation_cost  # All alerts
        
        total_cost = fraud_loss_cost + false_alarm_cost + investigation_cost
        
        return -total_cost  # Negative because sklearn maximizes scores
    
    def _calculate_business_metrics(self, y_true, y_pred, y_pred_proba=None):
        """Calculate comprehensive business and performance metrics."""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Standard metrics
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # F-beta scores (different emphasis on precision vs recall)
        f2_score = fbeta_score(y_true, y_pred, beta=2)  # Emphasizes recall
        f05_score = fbeta_score(y_true, y_pred, beta=0.5)  # Emphasizes precision
        
        # Business cost metrics
        fraud_loss_cost = fn * self.fraud_cost
        false_alarm_cost = fp * self.false_positive_cost
        investigation_cost = (tp + fp) * self.investigation_cost
        total_cost = fraud_loss_cost + false_alarm_cost + investigation_cost
        
        # Cost per transaction
        cost_per_transaction = total_cost / len(y_true)
        
        # Fraud prevented value
        fraud_prevented_value = tp * self.fraud_cost
        net_benefit = fraud_prevented_value - total_cost
        roi = (net_benefit / total_cost) * 100 if total_cost > 0 else 0
        
        # Alert efficiency
        alert_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        alert_volume = tp + fp
        
        # Additional metrics
        kappa = cohen_kappa_score(y_true, y_pred)
        
        metrics = {
            # Standard ML metrics
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1_score': f1,
            'f2_score': f2_score,
            'f05_score': f05_score,
            'kappa': kappa,
            
            # Confusion matrix elements
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            
            # Business metrics
            'fraud_loss_cost': fraud_loss_cost,
            'false_alarm_cost': false_alarm_cost,
            'investigation_cost': investigation_cost,
            'total_cost': total_cost,
            'cost_per_transaction': cost_per_transaction,
            'fraud_prevented_value': fraud_prevented_value,
            'net_benefit': net_benefit,
            'roi_percentage': roi,
            'alert_precision': alert_precision,
            'alert_volume': int(alert_volume),
            
            # Cost-sensitive score
            'cost_sensitive_score': -self._cost_sensitive_score(y_true, y_pred)
        }
        
        # Add ROC-AUC if probabilities available
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        
        return metrics
    
    def _perform_cross_validation(self, model, X, y, model_name, cv_folds=5):
        """
        Perform comprehensive cross-validation with multiple metrics.
        
        Implements feedback suggestion for cross-validation metrics in logs.
        """
        logger.info(f"Performing {cv_folds}-fold cross-validation for {model_name}")
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Multiple scoring metrics
        scoring_metrics = {
            'f1': 'f1',
            'precision': 'precision',
            'recall': 'recall',
            'roc_auc': 'roc_auc',
            'cost_sensitive': self.cost_sensitive_scorer
        }
        
        cv_results = {}
        
        for metric_name, scorer in scoring_metrics.items():
            scores = cross_val_score(model, X, y, cv=cv, scoring=scorer, n_jobs=-1)
            
            cv_results[f'{metric_name}_mean'] = scores.mean()
            cv_results[f'{metric_name}_std'] = scores.std()
            cv_results[f'{metric_name}_scores'] = scores.tolist()
            
            logger.info(f"  {metric_name}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        return cv_results
    
    def train_baseline_models(self, X_train, X_test, y_train, y_test):
        """Train baseline models with cost-sensitive approach."""
        logger.info("Training baseline models with cost-sensitive metrics")
        
        # Calculate class weights for imbalanced data
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
        
        logger.info(f"Class weights: {class_weight_dict}")
        
        # Define models with cost-sensitive configurations
        baseline_models = {
            'Logistic Regression': LogisticRegression(
                class_weight='balanced', 
                random_state=42, 
                max_iter=1000
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100, 
                class_weight='balanced', 
                random_state=42,
                n_jobs=-1
            ),
            'Decision Tree': DecisionTreeClassifier(
                class_weight='balanced', 
                random_state=42,
                max_depth=10
            ),
            'SVM': SVC(
                class_weight='balanced', 
                probability=True, 
                random_state=42
            ),
            'Naive Bayes': GaussianNB(),
            'Neural Network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                random_state=42,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1
            )
        }
        
        for name, model in baseline_models.items():
            logger.info(f"Training {name}")
            
            # Cross-validation before final training
            cv_results = self._perform_cross_validation(model, X_train, y_train, name)
            self.cv_scores[name] = cv_results
            
            # Train final model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate comprehensive metrics
            metrics = self._calculate_business_metrics(y_test, y_pred, y_pred_proba)
            
            # Add cross-validation results
            metrics.update({f'cv_{k}': v for k, v in cv_results.items()})
            
            self.models[name] = model
            self.results[name] = metrics
            
            logger.info(f"  {name} - F1: {metrics['f1_score']:.4f}, "
                       f"Cost: ${metrics['total_cost']:.0f}, ROI: {metrics['roi_percentage']:.1f}%")
    
    def train_advanced_models(self, X_train, X_test, y_train, y_test):
        """
        Train advanced models with early stopping and validation.
        
        Implements feedback suggestion for early stopping & CV metrics.
        """
        logger.info("Training advanced models with early stopping")
        
        # XGBoost with early stopping
        logger.info("Training XGBoost with early stopping")
        
        # Split training data for early stopping
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
        )
        
        # Calculate scale_pos_weight for imbalanced data
        scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
        
        xgb_model = xgb.XGBClassifier(
            n_estimators=1000,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            early_stopping_rounds=20,
            eval_metric='logloss'
        )
        
        # Fit with evaluation set
        xgb_model.fit(
            X_train_split, y_train_split,
            eval_set=[(X_val_split, y_val_split)],
            verbose=False
        )
        
        logger.info(f"  XGBoost stopped at iteration: {xgb_model.best_iteration}")
        
        # Cross-validation
        cv_results = self._perform_cross_validation(xgb_model, X_train, y_train, 'XGBoost')
        self.cv_scores['XGBoost'] = cv_results
        
        # Predictions
        y_pred = xgb_model.predict(X_test)
        y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
        
        # Metrics
        metrics = self._calculate_business_metrics(y_test, y_pred, y_pred_proba)
        metrics.update({f'cv_{k}': v for k, v in cv_results.items()})
        metrics['best_iteration'] = xgb_model.best_iteration
        
        self.models['XGBoost'] = xgb_model
        self.results['XGBoost'] = metrics
        
        # LightGBM with early stopping
        logger.info("Training LightGBM with early stopping")
        
        lgb_model = lgb.LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight='balanced',
            random_state=42,
            early_stopping_rounds=20,
            verbose=-1
        )
        
        # Fit with evaluation set
        lgb_model.fit(
            X_train_split, y_train_split,
            eval_set=[(X_val_split, y_val_split)],
            callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
        )
        
        logger.info(f"  LightGBM stopped at iteration: {lgb_model.best_iteration_}")
        
        # Cross-validation
        cv_results = self._perform_cross_validation(lgb_model, X_train, y_train, 'LightGBM')
        self.cv_scores['LightGBM'] = cv_results
        
        # Predictions
        y_pred = lgb_model.predict(X_test)
        y_pred_proba = lgb_model.predict_proba(X_test)[:, 1]
        
        # Metrics
        metrics = self._calculate_business_metrics(y_test, y_pred, y_pred_proba)
        metrics.update({f'cv_{k}': v for k, v in cv_results.items()})
        metrics['best_iteration'] = lgb_model.best_iteration_
        
        self.models['LightGBM'] = lgb_model
        self.results['LightGBM'] = metrics
        
        logger.info(f"  XGBoost - F1: {self.results['XGBoost']['f1_score']:.4f}, "
                   f"Cost: ${self.results['XGBoost']['total_cost']:.0f}")
        logger.info(f"  LightGBM - F1: {self.results['LightGBM']['f1_score']:.4f}, "
                   f"Cost: ${self.results['LightGBM']['total_cost']:.0f}")
    
    def create_comprehensive_comparison(self):
        """
        Create comprehensive model comparison table with cost-benefit analysis.
        
        Implements feedback suggestion for consolidated performance comparison.
        """
        logger.info("Creating comprehensive model comparison")
        
        if not self.results:
            logger.warning("No models trained yet")
            return None
        
        # Prepare comparison data
        comparison_data = []
        
        for model_name, metrics in self.results.items():
            row = {
                'Model': model_name,
                
                # Performance Metrics
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1_score']:.4f}",
                'F2-Score': f"{metrics['f2_score']:.4f}",
                'Specificity': f"{metrics['specificity']:.4f}",
                'ROC-AUC': f"{metrics.get('roc_auc', 0):.4f}",
                'Kappa': f"{metrics['kappa']:.4f}",
                
                # Business Metrics
                'Total_Cost': f"${metrics['total_cost']:.0f}",
                'Cost_Per_Transaction': f"${metrics['cost_per_transaction']:.2f}",
                'Net_Benefit': f"${metrics['net_benefit']:.0f}",
                'ROI_Percentage': f"{metrics['roi_percentage']:.1f}%",
                'Alert_Volume': metrics['alert_volume'],
                'Alert_Precision': f"{metrics['alert_precision']:.4f}",
                
                # Confusion Matrix
                'True_Positives': metrics['true_positives'],
                'False_Positives': metrics['false_positives'],
                'True_Negatives': metrics['true_negatives'],
                'False_Negatives': metrics['false_negatives'],
                
                # Cross-validation scores
                'CV_F1_Mean': f"{metrics.get('cv_f1_mean', 0):.4f}",
                'CV_F1_Std': f"{metrics.get('cv_f1_std', 0):.4f}",
                'CV_Cost_Mean': f"{metrics.get('cv_cost_sensitive_mean', 0):.0f}",
                'CV_Cost_Std': f"{metrics.get('cv_cost_sensitive_std', 0):.0f}",
                
                # Cost breakdown
                'Fraud_Loss_Cost': f"${metrics['fraud_loss_cost']:.0f}",
                'False_Alarm_Cost': f"${metrics['false_alarm_cost']:.0f}",
                'Investigation_Cost': f"${metrics['investigation_cost']:.0f}",
            }
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by best cost-sensitive performance (lowest total cost)
        numeric_costs = [float(cost.replace('$', '').replace(',', '')) 
                        for cost in comparison_df['Total_Cost']]
        comparison_df['_numeric_cost'] = numeric_costs
        comparison_df = comparison_df.sort_values('_numeric_cost').drop('_numeric_cost', axis=1)
        
        # Save comparison report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"model_comparison_report_{timestamp}.csv"
        comparison_df.to_csv(report_path, index=False)
        
        logger.info(f"Comprehensive comparison saved to: {report_path}")
        
        # Log summary
        best_model = comparison_df.iloc[0]['Model']
        best_cost = comparison_df.iloc[0]['Total_Cost']
        best_roi = comparison_df.iloc[0]['ROI_Percentage']
        
        logger.info(f"BEST MODEL: {best_model}")
        logger.info(f"  Total Cost: {best_cost}")
        logger.info(f"  ROI: {best_roi}")
        logger.info(f"  F1-Score: {comparison_df.iloc[0]['F1-Score']}")
        
        return comparison_df
    
    def save_models_and_results(self, save_path=None):
        """Save all models and results."""
        if save_path is None:
            save_path = "enhanced_fraud_models.joblib"
        
        save_data = {
            'models': self.models,
            'scaler': self.scaler,
            'results': self.results,
            'cv_scores': self.cv_scores,
            'cost_config': {
                'fraud_cost': self.fraud_cost,
                'false_positive_cost': self.false_positive_cost,
                'investigation_cost': self.investigation_cost
            }
        }
        
        joblib.dump(save_data, save_path)
        logger.info(f"Models and results saved to: {save_path}")

def main():
    """Main training pipeline with enhanced cost-sensitive approach."""
    logger.info("Starting Enhanced Fraud Detection Training")
    
    # Load data
    try:
        df = pd.read_csv('creditcard.csv')
        logger.info(f"Dataset loaded: {len(df):,} transactions")
        logger.info(f"Fraud rate: {df['Class'].mean()*100:.3f}%")
    except FileNotFoundError:
        logger.error("creditcard.csv not found. Please ensure the dataset is available.")
        return
    
    # Prepare features
    feature_columns = [col for col in df.columns if col not in ['Class']]
    X = df[feature_columns]
    y = df['Class']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info(f"Training set: {len(X_train):,} samples")
    logger.info(f"Test set: {len(X_test):,} samples")
    
    # Initialize detector with business costs
    detector = CostSensitiveFraudDetector(
        fraud_cost=200,  # Cost of missing fraud
        false_positive_cost=10,  # Cost of false alarm
        investigation_cost=30  # Cost to investigate
    )
    
    detector.scaler = scaler
    
    # Train models
    detector.train_baseline_models(X_train_scaled, X_test_scaled, y_train, y_test)
    detector.train_advanced_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Create comprehensive comparison
    comparison_df = detector.create_comprehensive_comparison()
    
    if comparison_df is not None:
        print("\n" + "="*100)
        print("COMPREHENSIVE MODEL COMPARISON")
        print("="*100)
        print(comparison_df[['Model', 'F1-Score', 'Total_Cost', 'ROI_Percentage', 
                            'Alert_Volume', 'CV_F1_Mean']].to_string(index=False))
        print("="*100)
    
    # Save everything
    detector.save_models_and_results()
    
    logger.info("Enhanced fraud detection training completed!")

if __name__ == "__main__":
    main()