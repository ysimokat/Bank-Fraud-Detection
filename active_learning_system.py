#!/usr/bin/env python3
"""
Active Learning System for Continuous Model Improvement
======================================================

Implements intelligent sampling strategies to identify the most informative
transactions for human review, enabling continuous model improvement.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_recall_curve
import joblib
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ActiveLearningStrategy:
    """Base class for active learning strategies."""
    
    def select_samples(self, X_pool, model, n_samples=10):
        """Select most informative samples from pool."""
        raise NotImplementedError

class UncertaintySampling(ActiveLearningStrategy):
    """Select samples with highest prediction uncertainty."""
    
    def select_samples(self, X_pool, model, n_samples=10):
        """Select samples closest to decision boundary."""
        if hasattr(model, 'predict_proba'):
            # Get prediction probabilities
            probs = model.predict_proba(X_pool)
            
            # Calculate uncertainty (entropy for multi-class, margin for binary)
            if probs.shape[1] == 2:
                # Binary classification - use margin sampling
                uncertainty = 1 - np.abs(probs[:, 1] - 0.5) * 2
            else:
                # Multi-class - use entropy
                uncertainty = entropy(probs.T)
            
            # Select top uncertain samples
            uncertain_indices = np.argsort(uncertainty)[-n_samples:]
            
            return uncertain_indices, uncertainty[uncertain_indices]
        else:
            # Fallback to random sampling
            return np.random.choice(len(X_pool), n_samples, replace=False), None

class DiversitySampling(ActiveLearningStrategy):
    """Select diverse samples to cover feature space."""
    
    def select_samples(self, X_pool, model, n_samples=10):
        """Select diverse samples using clustering."""
        from sklearn.cluster import KMeans
        
        # Cluster the pool
        n_clusters = min(n_samples, len(X_pool))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X_pool)
        
        # Select one sample from each cluster (closest to centroid)
        selected_indices = []
        
        for i in range(n_clusters):
            cluster_mask = cluster_labels == i
            cluster_points = X_pool[cluster_mask]
            
            if len(cluster_points) > 0:
                # Find point closest to centroid
                centroid = kmeans.cluster_centers_[i]
                distances = np.linalg.norm(cluster_points - centroid, axis=1)
                closest_idx = np.where(cluster_mask)[0][np.argmin(distances)]
                selected_indices.append(closest_idx)
        
        return np.array(selected_indices), None

class QueryByCommittee(ActiveLearningStrategy):
    """Use ensemble disagreement for sample selection."""
    
    def __init__(self, committee_models):
        self.committee = committee_models
    
    def select_samples(self, X_pool, model, n_samples=10):
        """Select samples with highest disagreement among committee."""
        # Get predictions from all committee members
        predictions = np.array([m.predict(X_pool) for m in self.committee])
        
        # Calculate disagreement (variance in predictions)
        disagreement = np.var(predictions, axis=0)
        
        # Select most disagreed upon samples
        disagreement_indices = np.argsort(disagreement)[-n_samples:]
        
        return disagreement_indices, disagreement[disagreement_indices]

class ExpectedModelChange(ActiveLearningStrategy):
    """Select samples that would most change the model."""
    
    def select_samples(self, X_pool, model, n_samples=10):
        """Estimate expected gradient magnitude."""
        # This is a simplified version
        # In practice, would compute actual gradient estimates
        
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X_pool)
            
            # Estimate gradient magnitude (simplified)
            # Higher uncertainty → larger expected gradient
            if probs.shape[1] == 2:
                gradient_magnitude = -probs[:, 1] * np.log(probs[:, 1] + 1e-10) - \
                                   probs[:, 0] * np.log(probs[:, 0] + 1e-10)
            else:
                gradient_magnitude = entropy(probs.T)
            
            # Weight by feature magnitude (samples with extreme features)
            feature_magnitude = np.linalg.norm(X_pool, axis=1)
            gradient_magnitude *= (1 + feature_magnitude / np.max(feature_magnitude))
            
            selected_indices = np.argsort(gradient_magnitude)[-n_samples:]
            
            return selected_indices, gradient_magnitude[selected_indices]
        
        return np.random.choice(len(X_pool), n_samples, replace=False), None

class FraudActiveLearner:
    """Active learning system for fraud detection."""
    
    def __init__(self, initial_model=None):
        self.model = initial_model or RandomForestClassifier(n_estimators=100, random_state=42)
        self.labeled_indices = []
        self.performance_history = []
        self.query_history = []
        self.strategies = {
            'uncertainty': UncertaintySampling(),
            'diversity': DiversitySampling(),
            'expected_change': ExpectedModelChange()
        }
        self.active_strategy = 'uncertainty'
        
    def initialize(self, X_initial, y_initial):
        """Initialize with small labeled dataset."""
        logger.info(f"Initializing with {len(X_initial)} labeled samples")
        
        self.model.fit(X_initial, y_initial)
        self.labeled_indices = list(range(len(X_initial)))
        
        # Evaluate initial performance
        initial_score = self._evaluate_model(X_initial, y_initial)
        self.performance_history.append({
            'iteration': 0,
            'n_labeled': len(X_initial),
            'f1_score': initial_score,
            'strategy': 'initial'
        })
        
    def query_samples(self, X_pool, n_queries=10, strategy=None):
        """Query most informative samples from pool."""
        strategy = strategy or self.active_strategy
        
        if strategy not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Select samples using chosen strategy
        selected_indices, scores = self.strategies[strategy].select_samples(
            X_pool, self.model, n_queries
        )
        
        # Record query
        self.query_history.append({
            'iteration': len(self.query_history),
            'strategy': strategy,
            'indices': selected_indices,
            'scores': scores
        })
        
        logger.info(f"Selected {len(selected_indices)} samples using {strategy} strategy")
        
        return selected_indices
    
    def update_model(self, X_new, y_new, X_pool_remaining=None):
        """Update model with newly labeled data."""
        # Combine with existing labeled data
        if hasattr(self, 'X_labeled'):
            X_combined = np.vstack([self.X_labeled, X_new])
            y_combined = np.hstack([self.y_labeled, y_new])
        else:
            X_combined = X_new
            y_combined = y_new
            
        self.X_labeled = X_combined
        self.y_labeled = y_combined
        
        # Retrain model
        self.model.fit(X_combined, y_combined)
        
        # Evaluate performance
        score = self._evaluate_model(X_combined, y_combined)
        
        self.performance_history.append({
            'iteration': len(self.performance_history),
            'n_labeled': len(X_combined),
            'f1_score': score,
            'strategy': self.active_strategy
        })
        
        logger.info(f"Model updated with {len(X_new)} new samples. F1-score: {score:.4f}")
        
        # Adaptive strategy selection
        if X_pool_remaining is not None:
            self._adapt_strategy(X_pool_remaining)
        
        return score
    
    def _evaluate_model(self, X, y):
        """Evaluate model performance."""
        # Simple train/test split for evaluation
        if len(X) < 20:
            return 0.0
            
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        
        return f1_score(y_test, y_pred)
    
    def _adapt_strategy(self, X_pool):
        """Adaptively select best strategy based on current state."""
        # Simple adaptive strategy selection
        n_labeled = len(self.labeled_indices)
        pool_size = len(X_pool)
        
        if n_labeled < 100:
            # Early stage: focus on diversity
            self.active_strategy = 'diversity'
        elif n_labeled < 500:
            # Middle stage: uncertainty sampling
            self.active_strategy = 'uncertainty'
        else:
            # Later stage: expected model change
            self.active_strategy = 'expected_change'
        
        # Override based on performance plateau
        if len(self.performance_history) > 5:
            recent_scores = [h['f1_score'] for h in self.performance_history[-5:]]
            if np.std(recent_scores) < 0.01:  # Performance plateau
                # Switch strategy
                strategies = list(self.strategies.keys())
                current_idx = strategies.index(self.active_strategy)
                self.active_strategy = strategies[(current_idx + 1) % len(strategies)]
                logger.info(f"Performance plateau detected. Switching to {self.active_strategy}")
    
    def get_confidence_threshold(self):
        """Calculate optimal confidence threshold for querying."""
        if not hasattr(self.model, 'predict_proba'):
            return 0.5
        
        # Use validation set to find optimal threshold
        if hasattr(self, 'X_labeled') and len(self.X_labeled) > 50:
            # Get predictions on labeled data
            probs = self.model.predict_proba(self.X_labeled)[:, 1]
            
            # Find threshold that maximizes F1
            precisions, recalls, thresholds = precision_recall_curve(self.y_labeled, probs)
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
            
            optimal_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
            
            return optimal_threshold
        
        return 0.5
    
    def visualize_progress(self):
        """Visualize active learning progress."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Performance over iterations
        iterations = [h['iteration'] for h in self.performance_history]
        f1_scores = [h['f1_score'] for h in self.performance_history]
        n_labeled = [h['n_labeled'] for h in self.performance_history]
        
        axes[0, 0].plot(iterations, f1_scores, marker='o', linewidth=2)
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('F1-Score')
        axes[0, 0].set_title('Model Performance Over Time')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Labeled samples growth
        axes[0, 1].plot(iterations, n_labeled, marker='s', color='green', linewidth=2)
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Number of Labeled Samples')
        axes[0, 1].set_title('Labeled Dataset Growth')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Strategy usage
        strategy_counts = defaultdict(int)
        for h in self.performance_history:
            strategy_counts[h.get('strategy', 'unknown')] += 1
        
        strategies = list(strategy_counts.keys())
        counts = list(strategy_counts.values())
        
        axes[1, 0].bar(strategies, counts, color=['blue', 'green', 'red', 'orange'][:len(strategies)])
        axes[1, 0].set_xlabel('Strategy')
        axes[1, 0].set_ylabel('Times Used')
        axes[1, 0].set_title('Active Learning Strategy Usage')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Learning efficiency
        if len(f1_scores) > 1:
            efficiency = np.diff(f1_scores) / np.diff(n_labeled)
            axes[1, 1].plot(iterations[1:], efficiency, marker='d', color='purple', linewidth=2)
            axes[1, 1].set_xlabel('Iteration')
            axes[1, 1].set_ylabel('F1 Improvement per Sample')
            axes[1, 1].set_title('Learning Efficiency')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig('/mnt/c/Users/Helen/Desktop/CodeMonkey/AI_Practice/Bank_Fraud_Detection/active_learning_progress.png',
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary
        print("\n📊 Active Learning Summary:")
        print(f"   • Total iterations: {len(self.performance_history)}")
        print(f"   • Final F1-score: {f1_scores[-1]:.4f}")
        print(f"   • Total labeled samples: {n_labeled[-1]}")
        print(f"   • F1 improvement: {f1_scores[-1] - f1_scores[0]:.4f}")
        print(f"   • Current strategy: {self.active_strategy}")

def simulate_active_learning(df, initial_size=100, pool_size=5000, n_iterations=20):
    """Simulate active learning process."""
    print("🚀 Simulating Active Learning for Fraud Detection")
    print("=" * 50)
    
    # Prepare data
    X = df.drop(['Class', 'Time'], axis=1).values
    y = df['Class'].values
    
    # Create initial labeled set and pool
    initial_indices = np.random.choice(len(X), initial_size, replace=False)
    pool_indices = np.random.choice(
        [i for i in range(len(X)) if i not in initial_indices],
        pool_size,
        replace=False
    )
    
    X_initial = X[initial_indices]
    y_initial = y[initial_indices]
    X_pool = X[pool_indices]
    y_pool = y[pool_indices]
    
    # Initialize active learner
    learner = FraudActiveLearner()
    learner.initialize(X_initial, y_initial)
    
    print(f"✅ Initialized with {initial_size} samples")
    print(f"📊 Pool size: {len(X_pool)} samples")
    
    # Active learning loop
    queries_per_iteration = 50
    
    for iteration in range(n_iterations):
        print(f"\n🔄 Iteration {iteration + 1}/{n_iterations}")
        
        # Query samples
        if len(X_pool) < queries_per_iteration:
            break
            
        query_indices = learner.query_samples(X_pool, n_queries=queries_per_iteration)
        
        # Simulate labeling (in practice, would be human annotation)
        X_queried = X_pool[query_indices]
        y_queried = y_pool[query_indices]
        
        # Update model
        score = learner.update_model(X_queried, y_queried, X_pool)
        
        # Remove queried samples from pool
        mask = np.ones(len(X_pool), dtype=bool)
        mask[query_indices] = False
        X_pool = X_pool[mask]
        y_pool = y_pool[mask]
        
        print(f"   📈 F1-score: {score:.4f}")
        print(f"   📊 Remaining pool: {len(X_pool)} samples")
    
    # Visualize results
    learner.visualize_progress()
    
    return learner

def create_human_in_the_loop_interface():
    """Create interface for human annotation."""
    print("\n🤝 Human-in-the-Loop Interface")
    print("=" * 40)
    
    print("""
    Active Learning Benefits:
    1. 🎯 Reduces labeling effort by 60-80%
    2. 📈 Faster model improvement
    3. 🔍 Focuses on difficult cases
    4. 💡 Identifies edge cases and anomalies
    5. 🔄 Continuous model improvement
    
    Implementation Steps:
    1. Deploy model with uncertainty estimation
    2. Queue uncertain transactions for review
    3. Present to human experts with context
    4. Collect labels and feedback
    5. Retrain model with new data
    6. Monitor improvement metrics
    """)

if __name__ == "__main__":
    # Load data
    df = pd.read_csv('/mnt/c/Users/Helen/Desktop/CodeMonkey/AI_Practice/Bank_Fraud_Detection/creditcard.csv')
    
    # Run simulation
    learner = simulate_active_learning(df)
    
    # Show interface design
    create_human_in_the_loop_interface()