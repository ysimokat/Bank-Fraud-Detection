#!/usr/bin/env python3
"""
Online Learning and Streaming Fraud Detection System
===================================================

Implementation of advanced online/streaming capabilities:
1. Incremental learning with concept drift adaptation
2. Online model updates with forgetting factors
3. Real-time streaming processing
4. Adaptive learning rates
5. Model ensemble with online reweighting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from collections import deque
import logging
import time
import threading
import queue
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Try to import river for online learning
try:
    from river import (
        linear_model as river_linear,
        ensemble as river_ensemble,
        tree as river_tree,
        naive_bayes as river_nb,
        preprocessing as river_preprocessing,
        drift as river_drift,
        metrics as river_metrics
    )
    RIVER_AVAILABLE = True
except ImportError:
    RIVER_AVAILABLE = False
    logging.warning("River not available. Some online learning features will be limited.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdaptiveOnlineClassifier(BaseEstimator, ClassifierMixin):
    """
    Adaptive online classifier with concept drift detection and forgetting.
    
    Features:
    1. Incremental learning with forgetting factor
    2. Concept drift detection
    3. Adaptive learning rate
    4. Performance tracking
    """
    
    def __init__(self, base_model=None, forgetting_factor=0.95, 
                 drift_threshold=0.1, adaptation_window=1000):
        """
        Args:
            base_model: Base online model (SGD, PassiveAggressive, etc.)
            forgetting_factor: Factor for exponential forgetting (0-1)
            drift_threshold: Threshold for drift detection
            adaptation_window: Window size for adaptation tracking
        """
        self.base_model = base_model or SGDClassifier(
            loss='log', learning_rate='adaptive', eta0=0.01,
            random_state=42, warm_start=True
        )
        self.forgetting_factor = forgetting_factor
        self.drift_threshold = drift_threshold
        self.adaptation_window = adaptation_window
        
        # Tracking variables
        self.performance_history = deque(maxlen=adaptation_window)
        self.sample_weights = deque(maxlen=adaptation_window)
        self.is_fitted = False
        self.n_samples_seen = 0
        self.current_learning_rate = 0.01
        
        # Drift detection
        self.drift_detector = None
        self.last_drift_detection = 0
        
    def _detect_drift(self, X, y):
        """Simple drift detection based on performance degradation."""
        if len(self.performance_history) < 100:
            return False
        
        # Calculate recent vs historical performance
        recent_performance = np.mean(list(self.performance_history)[-50:])
        historical_performance = np.mean(list(self.performance_history)[-200:-50])
        
        # Detect significant performance drop
        performance_drop = historical_performance - recent_performance
        
        if performance_drop > self.drift_threshold:
            logger.info(f"Concept drift detected! Performance drop: {performance_drop:.4f}")
            return True
        
        return False
    
    def _adapt_learning_rate(self):
        """Adapt learning rate based on recent performance."""
        if len(self.performance_history) < 50:
            return
        
        # Calculate performance trend
        recent_scores = list(self.performance_history)[-50:]
        trend = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
        
        if trend < -0.001:  # Performance declining
            self.current_learning_rate *= 1.1  # Increase learning rate
        elif trend > 0.001:  # Performance improving
            self.current_learning_rate *= 0.99  # Slightly decrease learning rate
        
        # Clamp learning rate
        self.current_learning_rate = np.clip(self.current_learning_rate, 0.001, 0.1)
        
        # Update model learning rate if possible
        if hasattr(self.base_model, 'eta0'):
            self.base_model.eta0 = self.current_learning_rate
    
    def partial_fit(self, X, y, classes=None):
        """
        Incrementally fit the model.
        
        Args:
            X: Features
            y: Labels
            classes: Class labels (required for first call)
        """
        if not self.is_fitted:
            self.is_fitted = True
            if classes is None:
                classes = np.unique(y)
            self.classes_ = classes
        
        # Calculate sample weights with forgetting
        n_samples = len(X)
        weights = []
        
        for i in range(n_samples):
            # More recent samples get higher weight
            age_weight = self.forgetting_factor ** (n_samples - i - 1)
            weights.append(age_weight)
        
        weights = np.array(weights)
        
        # Fit the model
        self.base_model.partial_fit(X, y, classes=self.classes_, sample_weight=weights)
        
        # Update tracking
        self.n_samples_seen += n_samples
        
        # Track performance (simplified - in practice would use holdout validation)
        if hasattr(self.base_model, 'predict_proba'):
            y_pred_proba = self.base_model.predict_proba(X)
            if y_pred_proba.shape[1] > 1:
                y_score = y_pred_proba[:, 1]
                if len(np.unique(y)) > 1:
                    auc_score = roc_auc_score(y, y_score)
                    self.performance_history.append(auc_score)
        
        # Drift detection and adaptation
        if self.n_samples_seen % 100 == 0:  # Check every 100 samples
            drift_detected = self._detect_drift(X, y)
            
            if drift_detected:
                self._handle_drift()
            
            self._adapt_learning_rate()
    
    def _handle_drift(self):
        """Handle detected concept drift."""
        logger.info("Handling concept drift")
        
        # Reset some model parameters for faster adaptation
        if hasattr(self.base_model, 'eta0'):
            self.base_model.eta0 *= 2  # Increase learning rate temporarily
        
        # Clear old performance history to focus on recent performance
        if len(self.performance_history) > 100:
            self.performance_history = deque(
                list(self.performance_history)[-100:], 
                maxlen=self.adaptation_window
            )
        
        self.last_drift_detection = self.n_samples_seen
    
    def predict(self, X):
        """Predict class labels."""
        return self.base_model.predict(X)
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        if hasattr(self.base_model, 'predict_proba'):
            return self.base_model.predict_proba(X)
        else:
            # Fallback for models without probability prediction
            predictions = self.predict(X)
            proba = np.zeros((len(predictions), len(self.classes_)))
            for i, pred in enumerate(predictions):
                proba[i, pred] = 1.0
            return proba

class OnlineEnsemble:
    """
    Online ensemble with dynamic model weighting and selection.
    
    Features:
    1. Multiple online learners
    2. Performance-based weighting
    3. Dynamic model addition/removal
    4. Forgetting for older models
    """
    
    def __init__(self, base_models=None, weighting_strategy='performance'):
        """
        Args:
            base_models: List of base online models
            weighting_strategy: 'performance', 'diversity', 'recent'
        """
        if base_models is None:
            self.base_models = {
                'sgd_log': AdaptiveOnlineClassifier(
                    SGDClassifier(loss='log', learning_rate='adaptive', random_state=42, warm_start=True)
                ),
                'sgd_hinge': AdaptiveOnlineClassifier(
                    SGDClassifier(loss='hinge', learning_rate='adaptive', random_state=43, warm_start=True)
                ),
                'passive_aggressive': AdaptiveOnlineClassifier(
                    PassiveAggressiveClassifier(random_state=44, warm_start=True)
                )
            }
        else:
            self.base_models = base_models
        
        self.weighting_strategy = weighting_strategy
        self.model_weights = {name: 1.0 for name in self.base_models.keys()}
        self.model_performance = {name: deque(maxlen=1000) for name in self.base_models.keys()}
        self.classes_ = None
        self.is_fitted = False
        
    def partial_fit(self, X, y, classes=None):
        """
        Incrementally fit all models in the ensemble.
        
        Args:
            X: Features
            y: Labels  
            classes: Class labels (required for first call)
        """
        if not self.is_fitted:
            self.is_fitted = True
            if classes is None:
                classes = np.unique(y)
            self.classes_ = classes
        
        # Fit each model
        for name, model in self.base_models.items():
            try:
                model.partial_fit(X, y, classes=self.classes_)
                
                # Track individual model performance
                if len(X) > 0:
                    y_pred = model.predict(X)
                    accuracy = accuracy_score(y, y_pred)
                    self.model_performance[name].append(accuracy)
                
            except Exception as e:
                logger.warning(f"Model {name} failed to fit: {e}")
        
        # Update model weights
        self._update_weights()
    
    def _update_weights(self):
        """Update model weights based on recent performance."""
        if self.weighting_strategy == 'performance':
            for name in self.base_models.keys():
                if len(self.model_performance[name]) > 10:
                    # Use recent performance
                    recent_performance = np.mean(list(self.model_performance[name])[-50:])
                    self.model_weights[name] = max(0.1, recent_performance)  # Minimum weight of 0.1
        
        elif self.weighting_strategy == 'diversity':
            # Implement diversity-based weighting (simplified)
            total_weight = len(self.base_models)
            for name in self.base_models.keys():
                self.model_weights[name] = 1.0 / total_weight
        
        # Normalize weights
        total_weight = sum(self.model_weights.values())
        if total_weight > 0:
            for name in self.model_weights:
                self.model_weights[name] /= total_weight
    
    def predict(self, X):
        """Predict using weighted ensemble."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        # Get predictions from all models
        predictions = {}
        for name, model in self.base_models.items():
            try:
                predictions[name] = model.predict(X)
            except Exception as e:
                logger.warning(f"Model {name} failed to predict: {e}")
        
        if not predictions:
            raise ValueError("No models available for prediction")
        
        # Weighted voting
        ensemble_pred = np.zeros(len(X))
        
        for name, pred in predictions.items():
            weight = self.model_weights.get(name, 0)
            ensemble_pred += weight * pred
        
        return (ensemble_pred > 0.5).astype(int)
    
    def predict_proba(self, X):
        """Predict probabilities using weighted ensemble."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        # Get probabilities from all models
        probabilities = {}
        for name, model in self.base_models.items():
            try:
                probabilities[name] = model.predict_proba(X)
            except Exception as e:
                logger.warning(f"Model {name} failed to predict probabilities: {e}")
        
        if not probabilities:
            raise ValueError("No models available for prediction")
        
        # Weighted ensemble
        ensemble_proba = np.zeros((len(X), len(self.classes_)))
        
        for name, proba in probabilities.items():
            weight = self.model_weights.get(name, 0)
            ensemble_proba += weight * proba
        
        return ensemble_proba

class StreamingFraudDetector:
    """
    Complete streaming fraud detection system.
    
    Features:
    1. Real-time transaction processing
    2. Online model updates
    3. Performance monitoring
    4. Drift detection and adaptation
    """
    
    def __init__(self, model=None, buffer_size=1000, update_frequency=100):
        """
        Args:
            model: Online model (ensemble by default)
            buffer_size: Size of transaction buffer
            update_frequency: How often to update model (in transactions)
        """
        self.model = model or OnlineEnsemble()
        self.buffer_size = buffer_size
        self.update_frequency = update_frequency
        
        # Streaming infrastructure
        self.transaction_buffer = deque(maxlen=buffer_size)
        self.label_buffer = deque(maxlen=buffer_size)
        self.prediction_buffer = deque(maxlen=buffer_size)
        
        # Performance tracking
        self.performance_metrics = {
            'timestamp': [],
            'accuracy': [],
            'f1_score': [],
            'precision': [],
            'recall': [],
            'processing_time': []
        }
        
        # Threading for real-time processing
        self.processing_queue = queue.Queue(maxsize=10000)
        self.is_running = False
        self.processing_thread = None
        
        # Statistics
        self.n_transactions_processed = 0
        self.n_model_updates = 0
        self.total_processing_time = 0
        
    def start_streaming(self):
        """Start the streaming processing thread."""
        logger.info("Starting streaming fraud detector")
        
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._process_stream)
        self.processing_thread.start()
    
    def stop_streaming(self):
        """Stop the streaming processing."""
        logger.info("Stopping streaming fraud detector")
        
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join()
    
    def _process_stream(self):
        """Main streaming processing loop."""
        while self.is_running:
            try:
                # Get transaction from queue (with timeout)
                transaction_data = self.processing_queue.get(timeout=1.0)
                
                start_time = time.time()
                
                # Process transaction
                self._process_single_transaction(transaction_data)
                
                # Track processing time
                processing_time = time.time() - start_time
                self.total_processing_time += processing_time
                
                self.processing_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing transaction: {e}")
    
    def _process_single_transaction(self, transaction_data):
        """
        Process a single transaction.
        
        Args:
            transaction_data: Dictionary with 'features', 'label' (optional), 'timestamp'
        """
        features = transaction_data['features']
        label = transaction_data.get('label')
        timestamp = transaction_data.get('timestamp', datetime.now())
        
        # Make prediction
        if self.model.is_fitted:
            start_pred_time = time.time()
            prediction = self.model.predict([features])[0]
            prediction_proba = self.model.predict_proba([features])[0]
            pred_time = time.time() - start_pred_time
        else:
            prediction = 0
            prediction_proba = np.array([1.0, 0.0])
            pred_time = 0
        
        # Store in buffers
        self.transaction_buffer.append(features)
        self.prediction_buffer.append({
            'prediction': prediction,
            'probability': prediction_proba[1],
            'timestamp': timestamp,
            'processing_time': pred_time
        })
        
        if label is not None:
            self.label_buffer.append(label)
        
        self.n_transactions_processed += 1
        
        # Update model if we have labels and it's time to update
        if (label is not None and 
            len(self.label_buffer) >= 10 and 
            self.n_transactions_processed % self.update_frequency == 0):
            
            self._update_model()
        
        # Update performance metrics periodically
        if self.n_transactions_processed % (self.update_frequency * 2) == 0:
            self._update_performance_metrics()
    
    def _update_model(self):
        """Update the model with recent labeled data."""
        if len(self.transaction_buffer) == 0 or len(self.label_buffer) == 0:
            return
        
        logger.info(f"Updating model with {len(self.label_buffer)} samples")
        
        # Get recent transactions and labels
        n_samples = min(len(self.transaction_buffer), len(self.label_buffer))
        
        if n_samples < 5:  # Need minimum samples
            return
        
        X = np.array(list(self.transaction_buffer)[-n_samples:])
        y = np.array(list(self.label_buffer)[-n_samples:])
        
        # Update model
        start_time = time.time()
        
        try:
            self.model.partial_fit(X, y)
            self.n_model_updates += 1
            
            update_time = time.time() - start_time
            logger.info(f"Model updated in {update_time:.3f}s (update #{self.n_model_updates})")
            
        except Exception as e:
            logger.error(f"Model update failed: {e}")
    
    def _update_performance_metrics(self):
        """Update performance metrics based on recent predictions."""
        if len(self.label_buffer) < 50 or len(self.prediction_buffer) < 50:
            return
        
        # Get recent predictions and labels
        n_samples = min(len(self.label_buffer), len(self.prediction_buffer))
        recent_labels = list(self.label_buffer)[-n_samples:]
        recent_predictions = [p['prediction'] for p in list(self.prediction_buffer)[-n_samples:]]
        recent_probabilities = [p['probability'] for p in list(self.prediction_buffer)[-n_samples:]]
        
        # Calculate metrics
        try:
            accuracy = accuracy_score(recent_labels, recent_predictions)
            f1 = f1_score(recent_labels, recent_predictions)
            precision = precision_score(recent_labels, recent_predictions, zero_division=0)
            recall = recall_score(recent_labels, recent_predictions, zero_division=0)
            
            # Average processing time
            avg_processing_time = np.mean([
                p['processing_time'] for p in list(self.prediction_buffer)[-n_samples:]
            ])
            
            # Store metrics
            self.performance_metrics['timestamp'].append(datetime.now())
            self.performance_metrics['accuracy'].append(accuracy)
            self.performance_metrics['f1_score'].append(f1)
            self.performance_metrics['precision'].append(precision)
            self.performance_metrics['recall'].append(recall)
            self.performance_metrics['processing_time'].append(avg_processing_time)
            
            logger.info(f"Performance update - Acc: {accuracy:.4f}, F1: {f1:.4f}, "
                       f"Precision: {precision:.4f}, Recall: {recall:.4f}")
            
        except Exception as e:
            logger.warning(f"Failed to calculate performance metrics: {e}")
    
    def submit_transaction(self, features, label=None, timestamp=None):
        """
        Submit a transaction for processing.
        
        Args:
            features: Transaction features
            label: True label (if available)
            timestamp: Transaction timestamp
        
        Returns:
            Success status
        """
        transaction_data = {
            'features': features,
            'label': label,
            'timestamp': timestamp or datetime.now()
        }
        
        try:
            self.processing_queue.put(transaction_data, timeout=1.0)
            return True
        except queue.Full:
            logger.warning("Processing queue is full, transaction dropped")
            return False
    
    def get_performance_report(self):
        """Get comprehensive performance report."""
        if not self.performance_metrics['timestamp']:
            return "No performance data available"
        
        latest_metrics = {
            'accuracy': self.performance_metrics['accuracy'][-1] if self.performance_metrics['accuracy'] else 0,
            'f1_score': self.performance_metrics['f1_score'][-1] if self.performance_metrics['f1_score'] else 0,
            'precision': self.performance_metrics['precision'][-1] if self.performance_metrics['precision'] else 0,
            'recall': self.performance_metrics['recall'][-1] if self.performance_metrics['recall'] else 0,
            'avg_processing_time': np.mean(self.performance_metrics['processing_time']) if self.performance_metrics['processing_time'] else 0
        }
        
        report = f"""
        Streaming Fraud Detector Performance Report
        ==========================================
        
        Processing Statistics:
        - Total transactions processed: {self.n_transactions_processed:,}
        - Model updates: {self.n_model_updates}
        - Average processing time: {latest_metrics['avg_processing_time']:.4f}s
        
        Latest Performance Metrics:
        - Accuracy: {latest_metrics['accuracy']:.4f}
        - F1-Score: {latest_metrics['f1_score']:.4f}
        - Precision: {latest_metrics['precision']:.4f}
        - Recall: {latest_metrics['recall']:.4f}
        
        Queue Status:
        - Queue size: {self.processing_queue.qsize()}
        - Buffer usage: {len(self.transaction_buffer)}/{self.buffer_size}
        """
        
        return report
    
    def plot_performance_history(self, save_path=None):
        """Plot performance metrics over time."""
        if not self.performance_metrics['timestamp']:
            logger.warning("No performance data to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        timestamps = self.performance_metrics['timestamp']
        
        # Accuracy
        axes[0, 0].plot(timestamps, self.performance_metrics['accuracy'], 'b-', linewidth=2)
        axes[0, 0].set_title('Accuracy Over Time')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].grid(True)
        
        # F1-Score
        axes[0, 1].plot(timestamps, self.performance_metrics['f1_score'], 'g-', linewidth=2)
        axes[0, 1].set_title('F1-Score Over Time')
        axes[0, 1].set_ylabel('F1-Score')
        axes[0, 1].grid(True)
        
        # Precision and Recall
        axes[1, 0].plot(timestamps, self.performance_metrics['precision'], 'orange', label='Precision', linewidth=2)
        axes[1, 0].plot(timestamps, self.performance_metrics['recall'], 'purple', label='Recall', linewidth=2)
        axes[1, 0].set_title('Precision and Recall Over Time')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Processing Time
        axes[1, 1].plot(timestamps, self.performance_metrics['processing_time'], 'red', linewidth=2)
        axes[1, 1].set_title('Processing Time Over Time')
        axes[1, 1].set_ylabel('Time (seconds)')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Performance plot saved to {save_path}")
        
        plt.show()

def simulate_streaming_fraud_detection(df, duration_minutes=5, transactions_per_second=10):
    """
    Simulate streaming fraud detection.
    
    Args:
        df: Transaction DataFrame
        duration_minutes: Simulation duration
        transactions_per_second: Rate of transactions
    """
    logger.info(f"Starting streaming simulation: {duration_minutes} minutes, {transactions_per_second} TPS")
    
    # Initialize streaming detector
    detector = StreamingFraudDetector(
        model=OnlineEnsemble(),
        buffer_size=5000,
        update_frequency=50
    )
    
    # Start streaming
    detector.start_streaming()
    
    # Prepare data
    feature_columns = [col for col in df.columns if col != 'Class']
    X = df[feature_columns].values
    y = df['Class'].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Simulation parameters
    total_transactions = duration_minutes * 60 * transactions_per_second
    transaction_indices = np.random.choice(len(X_scaled), total_transactions, replace=True)
    
    start_time = time.time()
    
    try:
        for i, idx in enumerate(transaction_indices):
            # Simulate transaction arrival
            features = X_scaled[idx]
            label = y[idx]
            
            # Submit transaction (with some probability of having label)
            has_label = np.random.random() < 0.3  # 30% of transactions have immediate labels
            label_to_submit = label if has_label else None
            
            success = detector.submit_transaction(features, label_to_submit)
            
            if not success:
                logger.warning(f"Failed to submit transaction {i}")
            
            # Control transaction rate
            expected_time = start_time + (i + 1) / transactions_per_second
            current_time = time.time()
            
            if current_time < expected_time:
                time.sleep(expected_time - current_time)
            
            # Progress update
            if i % (transactions_per_second * 30) == 0:  # Every 30 seconds
                elapsed_minutes = (time.time() - start_time) / 60
                logger.info(f"Simulation progress: {elapsed_minutes:.1f}/{duration_minutes} minutes")
                print(detector.get_performance_report())
    
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
    
    finally:
        # Stop streaming and generate final report
        detector.stop_streaming()
        
        logger.info("Streaming simulation completed")
        print("\n" + "="*60)
        print("FINAL PERFORMANCE REPORT")
        print("="*60)
        print(detector.get_performance_report())
        
        # Plot performance
        detector.plot_performance_history(save_path="streaming_performance.png")
        
        return detector

def main():
    """Main function to demonstrate online learning and streaming."""
    logger.info("Starting Online Learning and Streaming Demo")
    
    # Load data
    try:
        df = pd.read_csv('creditcard.csv')
        logger.info(f"Dataset loaded: {len(df):,} transactions")
    except FileNotFoundError:
        logger.error("creditcard.csv not found. Please ensure the dataset is available.")
        return
    
    # Use subset for demo
    df_demo = df.sample(min(50000, len(df))).reset_index(drop=True)
    logger.info(f"Using {len(df_demo):,} transactions for demo")
    
    # Run streaming simulation
    detector = simulate_streaming_fraud_detection(
        df_demo, 
        duration_minutes=2,  # 2 minute demo
        transactions_per_second=20
    )
    
    logger.info("Online learning and streaming demo completed!")
    
    return detector

if __name__ == "__main__":
    main()