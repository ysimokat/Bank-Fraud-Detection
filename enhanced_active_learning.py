#!/usr/bin/env python3
"""
Enhanced Active Learning System with Expected Error Reduction
===========================================================

Implementation of feedback suggestions:
1. Expected error reduction strategy
2. Dynamic strategy combination
3. Advanced uncertainty measures
4. Performance tracking and optimization
"""

import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExpectedErrorReduction:
    """
    Expected Error Reduction strategy for active learning.
    
    This strategy estimates how much the model's error would be reduced
    if we obtained the label for a particular sample.
    """
    
    def __init__(self, model, cv_folds=3):
        """
        Args:
            model: Base model to use for error estimation
            cv_folds: Number of cross-validation folds
        """
        self.model = model
        self.cv_folds = cv_folds
        
    def query(self, X_labeled, y_labeled, X_unlabeled, n_samples=10):
        """
        Select samples that would most reduce expected error.
        
        Args:
            X_labeled: Already labeled data
            y_labeled: Labels for labeled data
            X_unlabeled: Unlabeled data pool
            n_samples: Number of samples to select
        
        Returns:
            indices: Indices of selected samples
        """
        logger.info(f"Computing expected error reduction for {len(X_unlabeled)} candidates")
        
        # Get current model performance baseline
        baseline_scores = cross_val_score(
            self.model, X_labeled, y_labeled, 
            cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42),
            scoring='f1'
        )
        baseline_error = 1 - np.mean(baseline_scores)
        
        error_reductions = []
        
        # For computational efficiency, sample a subset if too many candidates
        n_candidates = min(len(X_unlabeled), 500)
        candidate_indices = np.random.choice(len(X_unlabeled), n_candidates, replace=False)
        
        for i, idx in enumerate(candidate_indices):
            if i % 100 == 0:
                logger.info(f"  Processed {i}/{n_candidates} candidates")
            
            # Estimate error reduction for this sample
            x_candidate = X_unlabeled[idx:idx+1]
            
            # Try both possible labels
            error_reduction_0 = self._estimate_error_reduction(
                X_labeled, y_labeled, x_candidate, 0, baseline_error
            )
            error_reduction_1 = self._estimate_error_reduction(
                X_labeled, y_labeled, x_candidate, 1, baseline_error
            )
            
            # Weight by predicted probability
            try:
                if hasattr(self.model, 'predict_proba'):
                    temp_model = self.model.__class__(**self.model.get_params())
                    temp_model.fit(X_labeled, y_labeled)
                    prob = temp_model.predict_proba(x_candidate)[0, 1]
                else:
                    prob = 0.5  # Assume uniform if no probability estimate
            except:
                prob = 0.5
            
            # Expected error reduction
            expected_reduction = (1 - prob) * error_reduction_0 + prob * error_reduction_1
            error_reductions.append((idx, expected_reduction))
        
        # Sort by expected error reduction (descending)
        error_reductions.sort(key=lambda x: x[1], reverse=True)
        
        # Return top n_samples
        selected_indices = [idx for idx, _ in error_reductions[:n_samples]]
        
        logger.info(f"Selected {len(selected_indices)} samples with avg error reduction: "
                   f"{np.mean([er for _, er in error_reductions[:n_samples]]):.6f}")
        
        return selected_indices
    
    def _estimate_error_reduction(self, X_labeled, y_labeled, x_candidate, y_candidate, baseline_error):
        """
        Estimate error reduction if we add this candidate with given label.
        """
        try:
            # Create augmented dataset
            X_augmented = np.vstack([X_labeled, x_candidate])
            y_augmented = np.hstack([y_labeled, [y_candidate]])
            
            # Estimate new performance
            scores = cross_val_score(
                self.model, X_augmented, y_augmented,
                cv=min(self.cv_folds, len(np.unique(y_augmented))),
                scoring='f1'
            )
            new_error = 1 - np.mean(scores)
            
            # Error reduction
            error_reduction = baseline_error - new_error
            return max(0, error_reduction)  # Can't be negative
            
        except Exception as e:
            logger.warning(f"Error in error reduction estimation: {e}")
            return 0

class DiversityBasedSampling:
    """
    Enhanced diversity-based sampling with multiple diversity measures.
    """
    
    def __init__(self, diversity_measure='euclidean'):
        """
        Args:
            diversity_measure: 'euclidean', 'cosine', 'mahalanobis', 'clustering'
        """
        self.diversity_measure = diversity_measure
        
    def query(self, X_labeled, X_unlabeled, n_samples=10):
        """
        Select diverse samples from unlabeled pool.
        """
        logger.info(f"Performing diversity-based sampling ({self.diversity_measure})")
        
        if self.diversity_measure == 'clustering':
            return self._clustering_diversity(X_labeled, X_unlabeled, n_samples)
        else:
            return self._distance_diversity(X_labeled, X_unlabeled, n_samples)
    
    def _clustering_diversity(self, X_labeled, X_unlabeled, n_samples):
        """Use clustering to find diverse samples."""
        # Combine all data for clustering
        X_all = np.vstack([X_labeled, X_unlabeled])
        
        # Perform clustering
        n_clusters = min(n_samples * 2, len(X_unlabeled))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_all)
        
        # Get clusters for unlabeled data
        unlabeled_clusters = clusters[len(X_labeled):]
        
        # Select one sample from each cluster (prioritize cluster centers)
        selected_indices = []
        cluster_centers = kmeans.cluster_centers_
        
        for cluster_id in range(n_clusters):
            cluster_mask = unlabeled_clusters == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) > 0:
                # Find closest to cluster center
                cluster_data = X_unlabeled[cluster_indices]
                distances = np.linalg.norm(cluster_data - cluster_centers[cluster_id], axis=1)
                closest_idx = cluster_indices[np.argmin(distances)]
                selected_indices.append(closest_idx)
                
                if len(selected_indices) >= n_samples:
                    break
        
        return selected_indices[:n_samples]
    
    def _distance_diversity(self, X_labeled, X_unlabeled, n_samples):
        """Use distance measures for diversity."""
        selected_indices = []
        
        for _ in range(n_samples):
            if len(selected_indices) == 0:
                # First sample: farthest from labeled data
                if self.diversity_measure == 'euclidean':
                    distances = cdist(X_unlabeled, X_labeled, metric='euclidean')
                elif self.diversity_measure == 'cosine':
                    distances = cdist(X_unlabeled, X_labeled, metric='cosine')
                else:  # mahalanobis
                    try:
                        # Compute covariance matrix
                        X_combined = np.vstack([X_labeled, X_unlabeled])
                        cov_matrix = np.cov(X_combined.T)
                        inv_cov = np.linalg.pinv(cov_matrix)
                        distances = cdist(X_unlabeled, X_labeled, metric='mahalanobis', VI=inv_cov)
                    except:
                        # Fallback to euclidean if mahalanobis fails
                        distances = cdist(X_unlabeled, X_labeled, metric='euclidean')
                
                min_distances = np.min(distances, axis=1)
                best_idx = np.argmax(min_distances)
                
            else:
                # Subsequent samples: farthest from both labeled and selected
                X_reference = np.vstack([X_labeled, X_unlabeled[selected_indices]])
                
                if self.diversity_measure == 'euclidean':
                    distances = cdist(X_unlabeled, X_reference, metric='euclidean')
                elif self.diversity_measure == 'cosine':
                    distances = cdist(X_unlabeled, X_reference, metric='cosine')
                else:  # mahalanobis
                    try:
                        X_combined = np.vstack([X_labeled, X_unlabeled])
                        cov_matrix = np.cov(X_combined.T)
                        inv_cov = np.linalg.pinv(cov_matrix)
                        distances = cdist(X_unlabeled, X_reference, metric='mahalanobis', VI=inv_cov)
                    except:
                        distances = cdist(X_unlabeled, X_reference, metric='euclidean')
                
                min_distances = np.min(distances, axis=1)
                # Don't select already selected samples
                min_distances[selected_indices] = -np.inf
                best_idx = np.argmax(min_distances)
            
            selected_indices.append(best_idx)
        
        return selected_indices

class AdvancedUncertaintySampling:
    """
    Advanced uncertainty sampling with multiple uncertainty measures.
    """
    
    def __init__(self, uncertainty_measure='entropy'):
        """
        Args:
            uncertainty_measure: 'entropy', 'least_confident', 'margin', 'variance'
        """
        self.uncertainty_measure = uncertainty_measure
        
    def query(self, model, X_unlabeled, n_samples=10):
        """
        Select samples with highest uncertainty.
        """
        logger.info(f"Performing uncertainty sampling ({self.uncertainty_measure})")
        
        if not hasattr(model, 'predict_proba'):
            logger.warning("Model doesn't support probability prediction, using distance to decision boundary")
            return self._decision_boundary_uncertainty(model, X_unlabeled, n_samples)
        
        # Get prediction probabilities
        probabilities = model.predict_proba(X_unlabeled)
        
        if self.uncertainty_measure == 'entropy':
            uncertainties = self._entropy_uncertainty(probabilities)
        elif self.uncertainty_measure == 'least_confident':
            uncertainties = self._least_confident_uncertainty(probabilities)
        elif self.uncertainty_measure == 'margin':
            uncertainties = self._margin_uncertainty(probabilities)
        else:  # variance
            uncertainties = self._variance_uncertainty(probabilities)
        
        # Select top uncertain samples
        uncertain_indices = np.argsort(uncertainties)[-n_samples:]
        
        return uncertain_indices.tolist()
    
    def _entropy_uncertainty(self, probabilities):
        """Calculate entropy-based uncertainty."""
        return np.array([entropy(prob) for prob in probabilities])
    
    def _least_confident_uncertainty(self, probabilities):
        """Calculate least confident uncertainty."""
        return 1 - np.max(probabilities, axis=1)
    
    def _margin_uncertainty(self, probabilities):
        """Calculate margin uncertainty (difference between top 2 predictions)."""
        if probabilities.shape[1] == 2:
            return 1 - np.abs(probabilities[:, 1] - probabilities[:, 0])
        else:
            sorted_probs = np.sort(probabilities, axis=1)
            return 1 - (sorted_probs[:, -1] - sorted_probs[:, -2])
    
    def _variance_uncertainty(self, probabilities):
        """Calculate variance-based uncertainty."""
        return np.var(probabilities, axis=1)
    
    def _decision_boundary_uncertainty(self, model, X_unlabeled, n_samples):
        """For models without probability support, use distance to decision boundary."""
        if hasattr(model, 'decision_function'):
            # SVM-like models
            distances = np.abs(model.decision_function(X_unlabeled))
            uncertain_indices = np.argsort(distances)[:n_samples]
        else:
            # Use prediction confidence as proxy
            predictions = model.predict(X_unlabeled)
            # Random sampling as fallback
            uncertain_indices = np.random.choice(len(X_unlabeled), n_samples, replace=False)
        
        return uncertain_indices.tolist()

class EnhancedActiveLearner:
    """
    Enhanced active learning system with multiple strategies and dynamic combination.
    
    Implements feedback suggestions:
    1. Expected error reduction strategy
    2. Dynamic strategy combination
    3. Performance tracking
    """
    
    def __init__(self, base_model=None, strategy_weights=None):
        """
        Args:
            base_model: Base model for active learning
            strategy_weights: Weights for different strategies
        """
        self.base_model = base_model if base_model else RandomForestClassifier(
            n_estimators=50, random_state=42, class_weight='balanced'
        )
        
        # Initialize strategies
        self.strategies = {
            'expected_error_reduction': ExpectedErrorReduction(self.base_model),
            'uncertainty_entropy': AdvancedUncertaintySampling('entropy'),
            'uncertainty_margin': AdvancedUncertaintySampling('margin'),
            'diversity_clustering': DiversityBasedSampling('clustering'),
            'diversity_euclidean': DiversityBasedSampling('euclidean')
        }
        
        # Strategy weights (will be learned dynamically)
        self.strategy_weights = strategy_weights if strategy_weights else {
            'expected_error_reduction': 0.3,
            'uncertainty_entropy': 0.25,
            'uncertainty_margin': 0.2,
            'diversity_clustering': 0.15,
            'diversity_euclidean': 0.1
        }
        
        # Performance tracking
        self.performance_history = []
        self.strategy_performance = {strategy: [] for strategy in self.strategies.keys()}
        
        # Current state
        self.X_labeled = None
        self.y_labeled = None
        self.current_model = None
        
        logger.info("Enhanced Active Learner initialized")
        logger.info(f"Strategies: {list(self.strategies.keys())}")
        logger.info(f"Initial weights: {self.strategy_weights}")
    
    def initialize(self, X_pool, y_pool, initial_samples=10):
        """
        Initialize with a small set of labeled samples.
        
        Args:
            X_pool: Full data pool
            y_pool: Full labels (for simulation)
            initial_samples: Number of initial samples to label
        """
        logger.info(f"Initializing with {initial_samples} samples")
        
        # Stratified initial sampling
        from sklearn.model_selection import train_test_split
        
        initial_indices, _ = train_test_split(
            range(len(X_pool)), 
            train_size=initial_samples, 
            stratify=y_pool, 
            random_state=42
        )
        
        self.X_labeled = X_pool[initial_indices]
        self.y_labeled = y_pool[initial_indices]
        
        # Train initial model
        self.current_model = self.base_model.__class__(**self.base_model.get_params())
        self.current_model.fit(self.X_labeled, self.y_labeled)
        
        # Remove initial samples from pool
        self.X_pool = np.delete(X_pool, initial_indices, axis=0)
        self.y_pool = np.delete(y_pool, initial_indices)
        self.pool_indices = np.delete(range(len(X_pool)), initial_indices)
        
        # Initial performance
        initial_performance = self._evaluate_model()
        self.performance_history.append(initial_performance)
        
        logger.info(f"Initial performance: F1={initial_performance['f1_score']:.4f}")
    
    def query_dynamic_ensemble(self, n_samples=10, adaptation_rate=0.1):
        """
        Query samples using dynamic ensemble of strategies.
        
        Implements dynamic strategy combination based on recent performance.
        """
        logger.info(f"Querying {n_samples} samples using dynamic ensemble")
        
        if len(self.X_pool) < n_samples:
            logger.warning(f"Pool size ({len(self.X_pool)}) smaller than requested samples")
            n_samples = len(self.X_pool)
        
        # Get candidates from each strategy
        strategy_candidates = {}
        
        for strategy_name, strategy in self.strategies.items():
            try:
                if strategy_name == 'expected_error_reduction':
                    candidates = strategy.query(
                        self.X_labeled, self.y_labeled, self.X_pool, 
                        n_samples=min(n_samples * 2, len(self.X_pool))
                    )
                elif 'uncertainty' in strategy_name:
                    candidates = strategy.query(
                        self.current_model, self.X_pool, 
                        n_samples=min(n_samples * 2, len(self.X_pool))
                    )
                else:  # diversity
                    candidates = strategy.query(
                        self.X_labeled, self.X_pool,
                        n_samples=min(n_samples * 2, len(self.X_pool))
                    )
                
                strategy_candidates[strategy_name] = candidates
                
            except Exception as e:
                logger.warning(f"Strategy {strategy_name} failed: {e}")
                strategy_candidates[strategy_name] = []
        
        # Combine strategies using weights
        candidate_scores = {}
        
        for strategy_name, candidates in strategy_candidates.items():
            weight = self.strategy_weights[strategy_name]
            
            for i, candidate_idx in enumerate(candidates):
                if candidate_idx not in candidate_scores:
                    candidate_scores[candidate_idx] = 0
                
                # Higher score for higher-ranked candidates
                score = weight * (len(candidates) - i) / len(candidates)
                candidate_scores[candidate_idx] += score
        
        # Select top candidates
        if not candidate_scores:
            logger.warning("No candidates from any strategy, using random sampling")
            selected_indices = np.random.choice(len(self.X_pool), n_samples, replace=False)
        else:
            sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
            selected_indices = [idx for idx, _ in sorted_candidates[:n_samples]]
        
        # Update strategy weights based on recent performance
        self._update_strategy_weights(adaptation_rate)
        
        logger.info(f"Selected {len(selected_indices)} samples")
        logger.info(f"Updated strategy weights: {self.strategy_weights}")
        
        return selected_indices
    
    def update_with_labels(self, selected_indices):
        """
        Update the model with newly labeled samples.
        
        Args:
            selected_indices: Indices of samples to label
        """
        logger.info(f"Updating model with {len(selected_indices)} new labels")
        
        # Get new samples and labels
        new_X = self.X_pool[selected_indices]
        new_y = self.y_pool[selected_indices]
        
        # Add to labeled set
        self.X_labeled = np.vstack([self.X_labeled, new_X])
        self.y_labeled = np.hstack([self.y_labeled, new_y])
        
        # Remove from pool
        self.X_pool = np.delete(self.X_pool, selected_indices, axis=0)
        self.y_pool = np.delete(self.y_pool, selected_indices)
        
        # Retrain model
        self.current_model = self.base_model.__class__(**self.base_model.get_params())
        self.current_model.fit(self.X_labeled, self.y_labeled)
        
        # Evaluate performance
        performance = self._evaluate_model()
        self.performance_history.append(performance)
        
        logger.info(f"New performance: F1={performance['f1_score']:.4f} "
                   f"(Δ={performance['f1_score'] - self.performance_history[-2]['f1_score']:.4f})")
        
        return performance
    
    def _evaluate_model(self):
        """Evaluate current model performance using cross-validation."""
        try:
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            
            f1_scores = cross_val_score(self.current_model, self.X_labeled, self.y_labeled, 
                                      cv=cv, scoring='f1')
            precision_scores = cross_val_score(self.current_model, self.X_labeled, self.y_labeled,
                                             cv=cv, scoring='precision')
            recall_scores = cross_val_score(self.current_model, self.X_labeled, self.y_labeled,
                                          cv=cv, scoring='recall')
            
            return {
                'f1_score': np.mean(f1_scores),
                'f1_std': np.std(f1_scores),
                'precision': np.mean(precision_scores),
                'precision_std': np.std(precision_scores),
                'recall': np.mean(recall_scores),
                'recall_std': np.std(recall_scores),
                'n_labeled': len(self.X_labeled)
            }
        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")
            return {
                'f1_score': 0, 'f1_std': 0, 'precision': 0, 'precision_std': 0,
                'recall': 0, 'recall_std': 0, 'n_labeled': len(self.X_labeled)
            }
    
    def _update_strategy_weights(self, adaptation_rate=0.1):
        """
        Update strategy weights based on recent performance improvements.
        
        Implements dynamic strategy combination.
        """
        if len(self.performance_history) < 3:
            return  # Need at least 3 evaluations for trend analysis
        
        # Calculate recent improvement
        recent_improvement = (self.performance_history[-1]['f1_score'] - 
                            self.performance_history[-2]['f1_score'])
        
        # Adjust weights based on improvement
        if recent_improvement > 0:
            # Good improvement - slightly increase weights of recently successful strategies
            # For simplicity, increase expected error reduction and uncertainty weights
            self.strategy_weights['expected_error_reduction'] += adaptation_rate * recent_improvement
            self.strategy_weights['uncertainty_entropy'] += adaptation_rate * recent_improvement * 0.5
        else:
            # Poor improvement - increase diversity weights
            self.strategy_weights['diversity_clustering'] += adaptation_rate * abs(recent_improvement)
            self.strategy_weights['diversity_euclidean'] += adaptation_rate * abs(recent_improvement) * 0.5
        
        # Normalize weights
        total_weight = sum(self.strategy_weights.values())
        for strategy in self.strategy_weights:
            self.strategy_weights[strategy] /= total_weight
    
    def run_active_learning_cycle(self, n_iterations=10, samples_per_iteration=5):
        """
        Run complete active learning cycle.
        
        Args:
            n_iterations: Number of AL iterations
            samples_per_iteration: Samples to query per iteration
        """
        logger.info(f"Running active learning cycle: {n_iterations} iterations, "
                   f"{samples_per_iteration} samples each")
        
        for iteration in range(n_iterations):
            logger.info(f"\n--- Iteration {iteration + 1}/{n_iterations} ---")
            
            if len(self.X_pool) < samples_per_iteration:
                logger.warning("Insufficient samples in pool, stopping")
                break
            
            # Query samples
            selected_indices = self.query_dynamic_ensemble(n_samples=samples_per_iteration)
            
            # Update model
            performance = self.update_with_labels(selected_indices)
            
            logger.info(f"Iteration {iteration + 1} complete. "
                       f"Labeled: {len(self.X_labeled)}, Pool: {len(self.X_pool)}")
        
        # Final summary
        logger.info("\n" + "="*60)
        logger.info("ACTIVE LEARNING SUMMARY")
        logger.info("="*60)
        logger.info(f"Initial F1: {self.performance_history[0]['f1_score']:.4f}")
        logger.info(f"Final F1: {self.performance_history[-1]['f1_score']:.4f}")
        logger.info(f"Improvement: {self.performance_history[-1]['f1_score'] - self.performance_history[0]['f1_score']:.4f}")
        logger.info(f"Total labeled samples: {len(self.X_labeled)}")
        logger.info(f"Final strategy weights: {self.strategy_weights}")
        logger.info("="*60)

def main():
    """Main demo of enhanced active learning system."""
    logger.info("Starting Enhanced Active Learning Demo")
    
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
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Use subset for demo (active learning is computationally intensive)
    n_samples = min(10000, len(X_scaled))
    indices = np.random.choice(len(X_scaled), n_samples, replace=False)
    X_demo = X_scaled[indices]
    y_demo = y[indices]
    
    logger.info(f"Demo with {len(X_demo):,} samples")
    logger.info(f"Fraud rate: {np.mean(y_demo)*100:.3f}%")
    
    # Initialize active learner
    learner = EnhancedActiveLearner(
        base_model=RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced')
    )
    
    # Initialize with small labeled set
    learner.initialize(X_demo, y_demo, initial_samples=20)
    
    # Run active learning cycle
    learner.run_active_learning_cycle(n_iterations=10, samples_per_iteration=10)
    
    logger.info("Enhanced active learning demo completed!")

if __name__ == "__main__":
    main()