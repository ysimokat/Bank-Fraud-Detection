#!/usr/bin/env python3
"""
Hybrid Ensemble System with Meta-Learners and Stacked Models
===========================================================

Implementation of sophisticated ensemble techniques:
1. Stacked ensembles with meta-models
2. Learned weighting based on context (time, amount, features)
3. Dynamic ensemble selection
4. Multi-level ensemble architecture
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                             VotingClassifier, StackingClassifier)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                           f1_score, precision_score, recall_score, accuracy_score)
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import xgboost as xgb
import lightgbm as lgb
import joblib
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContextAwareEnsemble:
    """
    Context-aware ensemble that adjusts model weights based on transaction context.
    
    Features:
    1. Time-of-day specific weighting
    2. Amount-based model selection
    3. Feature pattern recognition
    4. Dynamic weight adjustment
    """
    
    def __init__(self, base_models):
        """
        Args:
            base_models: Dictionary of base models
        """
        self.base_models = base_models
        self.context_weights = {}
        self.context_clusters = {}
        self.context_scalers = {}
        self.is_fitted = False
        
    def _extract_context_features(self, X):
        """Extract context features from transaction data."""
        context_features = []
        
        # Assuming X has columns: [Time, Amount, V1, V2, ..., V28]
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        for i, row in enumerate(X_array):
            features = {
                'hour_of_day': (row[0] % (24 * 3600)) // 3600,  # Assuming Time is first column
                'amount_log': np.log1p(row[1]) if len(row) > 1 else 0,  # Assuming Amount is second
                'amount_bin': self._get_amount_bin(row[1]) if len(row) > 1 else 0,
                'v_features_mean': np.mean(row[2:]) if len(row) > 2 else 0,
                'v_features_std': np.std(row[2:]) if len(row) > 2 else 0,
                'v_features_max': np.max(np.abs(row[2:])) if len(row) > 2 else 0
            }
            context_features.append(features)
        
        return pd.DataFrame(context_features)
    
    def _get_amount_bin(self, amount):
        """Get amount bin for context."""
        if amount <= 10:
            return 0  # Very low
        elif amount <= 50:
            return 1  # Low
        elif amount <= 200:
            return 2  # Medium
        elif amount <= 1000:
            return 3  # High
        else:
            return 4  # Very high
    
    def fit(self, X, y):
        """
        Fit the context-aware ensemble.
        
        Args:
            X: Training features
            y: Training labels
        """
        logger.info("Fitting context-aware ensemble")
        
        # Extract context features
        context_df = self._extract_context_features(X)
        
        # Create context clusters
        scaler = StandardScaler()
        context_scaled = scaler.fit_transform(context_df)
        self.context_scalers['main'] = scaler
        
        # Cluster contexts
        n_clusters = min(10, len(X) // 100)  # At least 100 samples per cluster
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        context_clusters = kmeans.fit_predict(context_scaled)
        self.context_clusters['kmeans'] = kmeans
        
        # Train base models
        for name, model in self.base_models.items():
            logger.info(f"Training base model: {name}")
            model.fit(X, y)
        
        # Learn context-specific weights
        self._learn_context_weights(X, y, context_df, context_clusters)
        
        self.is_fitted = True
        logger.info("Context-aware ensemble fitted successfully")
    
    def _learn_context_weights(self, X, y, context_df, context_clusters):
        """Learn optimal weights for each context cluster."""
        logger.info("Learning context-specific weights")
        
        # Initialize weights
        n_models = len(self.base_models)
        n_clusters = len(np.unique(context_clusters))
        
        # For each cluster, find optimal weights
        for cluster_id in range(n_clusters):
            cluster_mask = context_clusters == cluster_id
            
            if np.sum(cluster_mask) < 10:  # Skip small clusters
                continue
            
            X_cluster = X[cluster_mask]
            y_cluster = y[cluster_mask]
            
            # Get predictions from each base model for this cluster
            cluster_predictions = []
            for name, model in self.base_models.items():
                if hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(X_cluster)[:, 1]
                else:
                    pred = model.predict(X_cluster).astype(float)
                cluster_predictions.append(pred)
            
            cluster_predictions = np.array(cluster_predictions).T
            
            # Use ridge regression to find optimal weights
            ridge = Ridge(alpha=1.0, fit_intercept=False)
            ridge.fit(cluster_predictions, y_cluster)
            
            # Normalize weights to sum to 1
            weights = ridge.coef_
            weights = np.maximum(weights, 0)  # Ensure non-negative
            
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
            else:
                weights = np.ones(n_models) / n_models  # Equal weights as fallback
            
            self.context_weights[cluster_id] = weights
            
            logger.info(f"Cluster {cluster_id} ({np.sum(cluster_mask)} samples): "
                       f"weights = {weights}")
    
    def predict_proba(self, X):
        """
        Predict probabilities using context-aware ensemble.
        
        Args:
            X: Features to predict
        
        Returns:
            Prediction probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Extract context features
        context_df = self._extract_context_features(X)
        context_scaled = self.context_scalers['main'].transform(context_df)
        
        # Assign contexts
        context_clusters = self.context_clusters['kmeans'].predict(context_scaled)
        
        # Get base model predictions
        base_predictions = []
        for name, model in self.base_models.items():
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)[:, 1]
            else:
                pred = model.predict(X).astype(float)
            base_predictions.append(pred)
        
        base_predictions = np.array(base_predictions).T
        
        # Combine predictions using context-specific weights
        ensemble_predictions = np.zeros(len(X))
        
        for i, cluster_id in enumerate(context_clusters):
            if cluster_id in self.context_weights:
                weights = self.context_weights[cluster_id]
            else:
                # Use average weights for unseen clusters
                weights = np.mean(list(self.context_weights.values()), axis=0)
            
            ensemble_predictions[i] = np.dot(base_predictions[i], weights)
        
        # Return as probabilities
        return np.column_stack([1 - ensemble_predictions, ensemble_predictions])
    
    def predict(self, X):
        """Predict class labels."""
        probabilities = self.predict_proba(X)
        return (probabilities[:, 1] > 0.5).astype(int)

class MetaLearnerEnsemble:
    """
    Meta-learner ensemble using stacking with multiple meta-models.
    
    Features:
    1. Multiple meta-learners
    2. Cross-validation for meta-features
    3. Feature engineering for meta-level
    4. Ensemble of meta-learners
    """
    
    def __init__(self, base_models, meta_models=None, cv_folds=5):
        """
        Args:
            base_models: Dictionary of base models
            meta_models: Dictionary of meta-learners
            cv_folds: Cross-validation folds for meta-features
        """
        self.base_models = base_models
        self.cv_folds = cv_folds
        
        if meta_models is None:
            self.meta_models = {
                'logistic': LogisticRegression(random_state=42, max_iter=1000),
                'rf_meta': RandomForestClassifier(n_estimators=50, random_state=42),
                'xgb_meta': xgb.XGBClassifier(n_estimators=50, random_state=42, objective='binary:logistic')
            }
        else:
            self.meta_models = meta_models
        
        self.meta_features_train = None
        self.is_fitted = False
        
    def _generate_meta_features(self, X, y, training=True):
        """
        Generate meta-features using cross-validation.
        
        Args:
            X: Input features
            y: Target labels (only needed for training)
            training: Whether this is training or prediction
        
        Returns:
            Meta-features array
        """
        n_samples = len(X)
        n_models = len(self.base_models)
        
        if training:
            # Use cross-validation to generate meta-features
            meta_features = np.zeros((n_samples, n_models * 2))  # prob + prediction
            
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
            
            for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
                X_fold_train, X_fold_val = X[train_idx], X[val_idx]
                y_fold_train = y[train_idx]
                
                for i, (name, model) in enumerate(self.base_models.items()):
                    # Clone and train model on fold
                    fold_model = model.__class__(**model.get_params())
                    fold_model.fit(X_fold_train, y_fold_train)
                    
                    # Generate predictions for validation set
                    if hasattr(fold_model, 'predict_proba'):
                        prob = fold_model.predict_proba(X_fold_val)[:, 1]
                    else:
                        prob = fold_model.predict(X_fold_val).astype(float)
                    
                    pred = fold_model.predict(X_fold_val).astype(float)
                    
                    # Store meta-features
                    meta_features[val_idx, i * 2] = prob
                    meta_features[val_idx, i * 2 + 1] = pred
        
        else:
            # Use fitted base models for prediction
            meta_features = np.zeros((n_samples, n_models * 2))
            
            for i, (name, model) in enumerate(self.base_models.items()):
                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(X)[:, 1]
                else:
                    prob = model.predict(X).astype(float)
                
                pred = model.predict(X).astype(float)
                
                meta_features[:, i * 2] = prob
                meta_features[:, i * 2 + 1] = pred
        
        # Add engineered meta-features
        meta_features_engineered = self._engineer_meta_features(meta_features, X)
        
        return meta_features_engineered
    
    def _engineer_meta_features(self, meta_features, X):
        """Engineer additional meta-features."""
        n_models = len(self.base_models)
        
        # Extract probability predictions
        prob_predictions = meta_features[:, ::2]  # Every other column starting from 0
        
        # Engineered features
        additional_features = []
        
        # Agreement measures
        additional_features.append(np.std(prob_predictions, axis=1))  # Disagreement
        additional_features.append(np.mean(prob_predictions, axis=1))  # Average confidence
        additional_features.append(np.max(prob_predictions, axis=1))  # Max confidence
        additional_features.append(np.min(prob_predictions, axis=1))  # Min confidence
        
        # Voting measures
        binary_predictions = meta_features[:, 1::2]  # Every other column starting from 1
        additional_features.append(np.mean(binary_predictions, axis=1))  # Vote fraction
        additional_features.append(np.std(binary_predictions, axis=1))  # Vote disagreement
        
        # Rank-based features
        for i in range(prob_predictions.shape[0]):
            ranks = np.argsort(prob_predictions[i])
            additional_features.append([np.mean(ranks)])  # Average rank
        
        # Reshape and combine
        additional_features_array = np.column_stack(additional_features)
        
        # Combine original meta-features with engineered ones
        meta_features_final = np.column_stack([meta_features, additional_features_array])
        
        return meta_features_final
    
    def fit(self, X, y):
        """
        Fit the meta-learner ensemble.
        
        Args:
            X: Training features
            y: Training labels
        """
        logger.info("Fitting meta-learner ensemble")
        
        # Train base models
        for name, model in self.base_models.items():
            logger.info(f"Training base model: {name}")
            model.fit(X, y)
        
        # Generate meta-features using cross-validation
        logger.info("Generating meta-features")
        self.meta_features_train = self._generate_meta_features(X, y, training=True)
        
        # Train meta-learners
        for name, meta_model in self.meta_models.items():
            logger.info(f"Training meta-learner: {name}")
            meta_model.fit(self.meta_features_train, y)
        
        self.is_fitted = True
        logger.info("Meta-learner ensemble fitted successfully")
    
    def predict_proba(self, X):
        """
        Predict probabilities using meta-learner ensemble.
        
        Args:
            X: Features to predict
        
        Returns:
            Prediction probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Generate meta-features
        meta_features = self._generate_meta_features(X, None, training=False)
        
        # Get predictions from each meta-learner
        meta_predictions = []
        for name, meta_model in self.meta_models.items():
            if hasattr(meta_model, 'predict_proba'):
                pred = meta_model.predict_proba(meta_features)[:, 1]
            else:
                pred = meta_model.predict(meta_features).astype(float)
            meta_predictions.append(pred)
        
        # Ensemble of meta-learners (simple average)
        ensemble_pred = np.mean(meta_predictions, axis=0)
        
        return np.column_stack([1 - ensemble_pred, ensemble_pred])
    
    def predict(self, X):
        """Predict class labels."""
        probabilities = self.predict_proba(X)
        return (probabilities[:, 1] > 0.5).astype(int)

class DynamicEnsembleSelector:
    """
    Dynamic ensemble that selects the best model combination for each prediction.
    
    Features:
    1. Instance-based model selection
    2. Confidence-weighted ensembling
    3. Performance-based model ranking
    4. Adaptive selection criteria
    """
    
    def __init__(self, base_models, selection_strategy='confidence'):
        """
        Args:
            base_models: Dictionary of base models
            selection_strategy: 'confidence', 'performance', 'diversity'
        """
        self.base_models = base_models
        self.selection_strategy = selection_strategy
        self.model_performance = {}
        self.model_confidence_calibration = {}
        self.is_fitted = False
    
    def fit(self, X, y):
        """
        Fit the dynamic ensemble selector.
        
        Args:
            X: Training features
            y: Training labels
        """
        logger.info("Fitting dynamic ensemble selector")
        
        # Train base models and evaluate performance
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        for name, model in self.base_models.items():
            logger.info(f"Training and evaluating: {name}")
            
            # Train on full dataset
            model.fit(X, y)
            
            # Cross-validation performance
            cv_scores = []
            cv_confidences = []
            
            for train_idx, val_idx in cv.split(X, y):
                X_train_cv, X_val_cv = X[train_idx], X[val_idx]
                y_train_cv, y_val_cv = y[train_idx], y[val_idx]
                
                # Train fold model
                fold_model = model.__class__(**model.get_params())
                fold_model.fit(X_train_cv, y_train_cv)
                
                # Evaluate
                if hasattr(fold_model, 'predict_proba'):
                    y_prob = fold_model.predict_proba(X_val_cv)[:, 1]
                    y_pred = (y_prob > 0.5).astype(int)
                else:
                    y_pred = fold_model.predict(X_val_cv)
                    y_prob = y_pred.astype(float)
                
                # Performance metrics
                f1 = f1_score(y_val_cv, y_pred)
                precision = precision_score(y_val_cv, y_pred)
                recall = recall_score(y_val_cv, y_pred)
                
                cv_scores.append({
                    'f1': f1,
                    'precision': precision,
                    'recall': recall
                })
                
                # Confidence calibration
                cv_confidences.extend(y_prob)
            
            # Store performance
            self.model_performance[name] = {
                'f1_mean': np.mean([s['f1'] for s in cv_scores]),
                'f1_std': np.std([s['f1'] for s in cv_scores]),
                'precision_mean': np.mean([s['precision'] for s in cv_scores]),
                'recall_mean': np.mean([s['recall'] for s in cv_scores])
            }
            
            # Store confidence distribution
            self.model_confidence_calibration[name] = np.array(cv_confidences)
        
        self.is_fitted = True
        logger.info("Dynamic ensemble selector fitted successfully")
        
        # Log performance summary
        for name, perf in self.model_performance.items():
            logger.info(f"{name}: F1={perf['f1_mean']:.4f}±{perf['f1_std']:.4f}, "
                       f"Precision={perf['precision_mean']:.4f}, "
                       f"Recall={perf['recall_mean']:.4f}")
    
    def _select_models_for_instance(self, x_instance):
        """
        Select best models for a specific instance.
        
        Args:
            x_instance: Single instance features
        
        Returns:
            Selected model names and their weights
        """
        if self.selection_strategy == 'performance':
            # Select top performing models
            model_scores = [(name, perf['f1_mean']) 
                          for name, perf in self.model_performance.items()]
            model_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Take top 3 models
            selected_models = [name for name, _ in model_scores[:3]]
            weights = [score for _, score in model_scores[:3]]
            weights = np.array(weights) / np.sum(weights)
            
        elif self.selection_strategy == 'confidence':
            # Select models based on prediction confidence
            selected_models = []
            weights = []
            
            for name, model in self.base_models.items():
                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba([x_instance])[0, 1]
                    # Use distance from 0.5 as confidence
                    confidence = abs(prob - 0.5) * 2
                else:
                    confidence = 0.5  # Default confidence for models without probabilities
                
                selected_models.append(name)
                weights.append(confidence)
            
            weights = np.array(weights)
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
            else:
                weights = np.ones(len(weights)) / len(weights)
        
        else:  # diversity
            # Select diverse models (simplified)
            selected_models = list(self.base_models.keys())
            weights = np.ones(len(selected_models)) / len(selected_models)
        
        return selected_models, weights
    
    def predict_proba(self, X):
        """
        Predict probabilities using dynamic model selection.
        
        Args:
            X: Features to predict
        
        Returns:
            Prediction probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        predictions = []
        
        for i, x_instance in enumerate(X):
            # Select models for this instance
            selected_models, weights = self._select_models_for_instance(x_instance)
            
            # Get predictions from selected models
            instance_predictions = []
            for model_name in selected_models:
                model = self.base_models[model_name]
                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba([x_instance])[0, 1]
                else:
                    prob = model.predict([x_instance])[0].astype(float)
                instance_predictions.append(prob)
            
            # Weighted ensemble
            ensemble_pred = np.dot(instance_predictions, weights)
            predictions.append(ensemble_pred)
        
        predictions = np.array(predictions)
        return np.column_stack([1 - predictions, predictions])
    
    def predict(self, X):
        """Predict class labels."""
        probabilities = self.predict_proba(X)
        return (probabilities[:, 1] > 0.5).astype(int)

class HybridEnsembleSystem:
    """
    Complete hybrid ensemble system combining multiple ensemble strategies.
    
    Features:
    1. Context-aware ensemble
    2. Meta-learner ensemble
    3. Dynamic ensemble selector
    4. Final meta-ensemble combining all approaches
    """
    
    def __init__(self, base_models):
        """
        Args:
            base_models: Dictionary of base models
        """
        self.base_models = base_models
        
        # Initialize ensemble components
        self.context_ensemble = ContextAwareEnsemble(base_models)
        self.meta_ensemble = MetaLearnerEnsemble(base_models)
        self.dynamic_ensemble = DynamicEnsembleSelector(base_models)
        
        # Final meta-ensemble
        self.final_meta_model = LogisticRegression(random_state=42, max_iter=1000)
        
        self.is_fitted = False
    
    def fit(self, X, y):
        """
        Fit the complete hybrid ensemble system.
        
        Args:
            X: Training features
            y: Training labels
        """
        logger.info("Fitting hybrid ensemble system")
        
        # Split data for final meta-model training
        X_base, X_meta, y_base, y_meta = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42
        )
        
        # Fit ensemble components
        logger.info("Fitting context-aware ensemble")
        self.context_ensemble.fit(X_base, y_base)
        
        logger.info("Fitting meta-learner ensemble")
        self.meta_ensemble.fit(X_base, y_base)
        
        logger.info("Fitting dynamic ensemble selector")
        self.dynamic_ensemble.fit(X_base, y_base)
        
        # Generate meta-features for final ensemble
        logger.info("Training final meta-ensemble")
        meta_features = self._generate_final_meta_features(X_meta)
        
        # Train final meta-model
        self.final_meta_model.fit(meta_features, y_meta)
        
        self.is_fitted = True
        logger.info("Hybrid ensemble system fitted successfully")
    
    def _generate_final_meta_features(self, X):
        """Generate meta-features from all ensemble components."""
        # Get predictions from each ensemble
        context_probs = self.context_ensemble.predict_proba(X)[:, 1]
        meta_probs = self.meta_ensemble.predict_proba(X)[:, 1]
        dynamic_probs = self.dynamic_ensemble.predict_proba(X)[:, 1]
        
        # Combine into meta-features
        meta_features = np.column_stack([
            context_probs,
            meta_probs,
            dynamic_probs,
            # Additional engineered features
            np.abs(context_probs - meta_probs),  # Disagreement 1
            np.abs(context_probs - dynamic_probs),  # Disagreement 2
            np.abs(meta_probs - dynamic_probs),  # Disagreement 3
            np.mean([context_probs, meta_probs, dynamic_probs], axis=0),  # Average
            np.std([context_probs, meta_probs, dynamic_probs], axis=0)  # Std
        ])
        
        return meta_features
    
    def predict_proba(self, X):
        """
        Predict probabilities using the hybrid ensemble.
        
        Args:
            X: Features to predict
        
        Returns:
            Prediction probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Generate meta-features
        meta_features = self._generate_final_meta_features(X)
        
        # Final prediction
        final_probs = self.final_meta_model.predict_proba(meta_features)
        
        return final_probs
    
    def predict(self, X):
        """Predict class labels."""
        probabilities = self.predict_proba(X)
        return (probabilities[:, 1] > 0.5).astype(int)
    
    def get_detailed_predictions(self, X):
        """
        Get detailed predictions from all ensemble components.
        
        Args:
            X: Features to predict
        
        Returns:
            Dictionary with predictions from all components
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Get predictions from all components
        context_probs = self.context_ensemble.predict_proba(X)[:, 1]
        meta_probs = self.meta_ensemble.predict_proba(X)[:, 1]
        dynamic_probs = self.dynamic_ensemble.predict_proba(X)[:, 1]
        final_probs = self.predict_proba(X)[:, 1]
        
        return {
            'context_aware': context_probs,
            'meta_learner': meta_probs,
            'dynamic_selector': dynamic_probs,
            'final_hybrid': final_probs
        }

def main():
    """Main function to demonstrate hybrid ensemble system."""
    logger.info("Starting Hybrid Ensemble System Demo")
    
    # Load data
    try:
        df = pd.read_csv('creditcard.csv')
        logger.info(f"Dataset loaded: {len(df):,} transactions")
    except FileNotFoundError:
        logger.error("creditcard.csv not found. Please ensure the dataset is available.")
        return
    
    # Prepare data
    feature_columns = [col for col in df.columns if col != 'Class']
    X = df[feature_columns].values
    y = df['Class'].values
    
    # Use subset for demo
    n_samples = min(10000, len(X))
    indices = np.random.choice(len(X), n_samples, replace=False)
    X_demo = X[indices]
    y_demo = y[indices]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_demo)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_demo, test_size=0.2, stratify=y_demo, random_state=42
    )
    
    logger.info(f"Training set: {len(X_train)}, Test set: {len(X_test)}")
    
    # Define base models
    base_models = {
        'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced'),
        'XGBoost': xgb.XGBClassifier(n_estimators=50, random_state=42, scale_pos_weight=10),
        'Logistic Regression': LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
        'SVM': SVC(probability=True, random_state=42, class_weight='balanced'),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=50, random_state=42)
    }
    
    # Create and fit hybrid ensemble
    hybrid_ensemble = HybridEnsembleSystem(base_models)
    hybrid_ensemble.fit(X_train, y_train)
    
    # Evaluate on test set
    logger.info("Evaluating hybrid ensemble system")
    
    # Get detailed predictions
    detailed_predictions = hybrid_ensemble.get_detailed_predictions(X_test)
    
    # Evaluate each component
    for component_name, predictions in detailed_predictions.items():
        y_pred = (predictions > 0.5).astype(int)
        
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, predictions)
        
        logger.info(f"{component_name}: F1={f1:.4f}, Precision={precision:.4f}, "
                   f"Recall={recall:.4f}, ROC-AUC={roc_auc:.4f}")
    
    # Save results
    results = {
        'hybrid_ensemble': hybrid_ensemble,
        'detailed_predictions': detailed_predictions,
        'test_labels': y_test,
        'scaler': scaler
    }
    
    joblib.dump(results, 'hybrid_ensemble_results.joblib')
    logger.info("Hybrid ensemble system demo completed!")
    
    return hybrid_ensemble, detailed_predictions

if __name__ == "__main__":
    main()