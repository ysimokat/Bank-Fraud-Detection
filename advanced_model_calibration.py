#!/usr/bin/env python3
"""
Advanced Model Calibration and Threshold Optimization
====================================================

Implementation of sophisticated improvements:
1. Model calibration with Platt scaling and isotonic regression
2. Post-model threshold tuning with business cost optimization
3. Dynamic thresholding by transaction context
4. Precision-Recall curve optimization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (precision_recall_curve, roc_curve, average_precision_score,
                           brier_score_loss, log_loss, precision_score, recall_score, 
                           f1_score, confusion_matrix)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import joblib
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelCalibratorAdvanced:
    """
    Advanced model calibration with multiple methods and validation.
    
    Implements:
    1. Platt scaling (sigmoid calibration)
    2. Isotonic regression (non-parametric calibration) 
    3. Calibration evaluation metrics
    4. Cross-validation for robust calibration
    """
    
    def __init__(self, base_models, calibration_methods=['sigmoid', 'isotonic']):
        """
        Args:
            base_models: Dictionary of trained models
            calibration_methods: List of calibration methods to apply
        """
        self.base_models = base_models
        self.calibration_methods = calibration_methods
        self.calibrated_models = {}
        self.calibration_scores = {}
        
    def calibrate_models(self, X_cal, y_cal, cv_folds=3):
        """
        Calibrate all models using multiple methods.
        
        Args:
            X_cal: Calibration features
            y_cal: Calibration labels
            cv_folds: Cross-validation folds for calibration
        """
        logger.info("Starting model calibration process")
        
        for model_name, model in self.base_models.items():
            logger.info(f"Calibrating {model_name}")
            
            model_calibrated = {}
            model_scores = {}
            
            for method in self.calibration_methods:
                try:
                    # Create calibrated classifier
                    calibrated_clf = CalibratedClassifierCV(
                        base_estimator=model,
                        method=method,
                        cv=cv_folds,
                        ensemble=True  # Use ensemble of calibrators
                    )
                    
                    # Fit calibrator
                    calibrated_clf.fit(X_cal, y_cal)
                    
                    # Evaluate calibration quality
                    y_prob_cal = calibrated_clf.predict_proba(X_cal)[:, 1]
                    
                    # Calculate calibration metrics
                    brier_score = brier_score_loss(y_cal, y_prob_cal)
                    log_loss_score = log_loss(y_cal, y_prob_cal)
                    
                    # Reliability diagram data
                    fraction_of_positives, mean_predicted_value = calibration_curve(
                        y_cal, y_prob_cal, n_bins=10, strategy='uniform'
                    )
                    
                    # Expected Calibration Error (ECE)
                    ece = self._calculate_ece(y_cal, y_prob_cal)
                    
                    # Maximum Calibration Error (MCE)
                    mce = self._calculate_mce(y_cal, y_prob_cal)
                    
                    model_calibrated[method] = calibrated_clf
                    model_scores[method] = {
                        'brier_score': brier_score,
                        'log_loss': log_loss_score,
                        'ece': ece,
                        'mce': mce,
                        'fraction_of_positives': fraction_of_positives,
                        'mean_predicted_value': mean_predicted_value
                    }
                    
                    logger.info(f"  {method}: Brier={brier_score:.4f}, "
                               f"ECE={ece:.4f}, MCE={mce:.4f}")
                    
                except Exception as e:
                    logger.warning(f"Calibration failed for {model_name} with {method}: {e}")
            
            self.calibrated_models[model_name] = model_calibrated
            self.calibration_scores[model_name] = model_scores
    
    def _calculate_ece(self, y_true, y_prob, n_bins=10):
        """
        Calculate Expected Calibration Error.
        
        ECE measures the expected difference between confidence and accuracy.
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def _calculate_mce(self, y_true, y_prob, n_bins=10):
        """Calculate Maximum Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        max_error = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            
            if in_bin.sum() > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                error = np.abs(avg_confidence_in_bin - accuracy_in_bin)
                max_error = max(max_error, error)
        
        return max_error
    
    def get_best_calibrated_model(self, model_name, metric='ece'):
        """
        Get the best calibrated version of a model.
        
        Args:
            model_name: Name of the base model
            metric: Metric to optimize ('ece', 'mce', 'brier_score', 'log_loss')
        
        Returns:
            Best calibrated model and its method
        """
        if model_name not in self.calibrated_models:
            return None, None
        
        model_scores = self.calibration_scores[model_name]
        
        # Find best method (lower is better for all these metrics)
        best_method = min(model_scores.keys(), 
                         key=lambda x: model_scores[x][metric])
        
        best_model = self.calibrated_models[model_name][best_method]
        
        logger.info(f"Best calibration for {model_name}: {best_method} "
                   f"({metric}={model_scores[best_method][metric]:.4f})")
        
        return best_model, best_method
    
    def plot_calibration_curves(self, X_test, y_test, save_path=None):
        """
        Plot calibration curves for all models and methods.
        
        Args:
            X_test: Test features
            y_test: Test labels
            save_path: Path to save the plot
        """
        n_models = len(self.calibrated_models)
        n_methods = len(self.calibration_methods)
        
        fig, axes = plt.subplots(n_models, n_methods + 1, 
                                figsize=(4 * (n_methods + 1), 4 * n_models))
        
        if n_models == 1:
            axes = axes.reshape(1, -1)
        
        for i, (model_name, calibrated_versions) in enumerate(self.calibrated_models.items()):
            # Plot original model
            base_model = self.base_models[model_name]
            if hasattr(base_model, 'predict_proba'):
                y_prob_base = base_model.predict_proba(X_test)[:, 1]
                
                fraction_pos, mean_pred = calibration_curve(y_test, y_prob_base, n_bins=10)
                axes[i, 0].plot(mean_pred, fraction_pos, 's-', label='Original')
                axes[i, 0].plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
                axes[i, 0].set_title(f'{model_name} - Original')
                axes[i, 0].set_xlabel('Mean Predicted Probability')
                axes[i, 0].set_ylabel('Fraction of Positives')
                axes[i, 0].legend()
                axes[i, 0].grid(True)
            
            # Plot calibrated versions
            for j, method in enumerate(self.calibration_methods):
                if method in calibrated_versions:
                    cal_model = calibrated_versions[method]
                    y_prob_cal = cal_model.predict_proba(X_test)[:, 1]
                    
                    fraction_pos, mean_pred = calibration_curve(y_test, y_prob_cal, n_bins=10)
                    axes[i, j + 1].plot(mean_pred, fraction_pos, 's-', label=f'{method.title()}')
                    axes[i, j + 1].plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
                    axes[i, j + 1].set_title(f'{model_name} - {method.title()}')
                    axes[i, j + 1].set_xlabel('Mean Predicted Probability')
                    axes[i, j + 1].set_ylabel('Fraction of Positives')
                    axes[i, j + 1].legend()
                    axes[i, j + 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Calibration curves saved to {save_path}")
        
        plt.show()

class BusinessOptimalThresholds:
    """
    Optimize classification thresholds based on business costs and context.
    
    Implements:
    1. Business cost-based threshold optimization
    2. Dynamic thresholds by transaction context
    3. Precision-Recall curve analysis
    4. Multi-objective threshold selection
    """
    
    def __init__(self, fraud_cost=200, false_positive_cost=10, investigation_cost=30):
        """
        Args:
            fraud_cost: Cost of missing a fraud (false negative)
            false_positive_cost: Cost of false alarm
            investigation_cost: Cost to investigate each alert
        """
        self.fraud_cost = fraud_cost
        self.false_positive_cost = false_positive_cost
        self.investigation_cost = investigation_cost
        self.optimal_thresholds = {}
        self.threshold_analysis = {}
        
    def find_optimal_threshold_business(self, y_true, y_prob, model_name):
        """
        Find optimal threshold based on business costs.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            model_name: Name of the model
        
        Returns:
            Optimal threshold and cost analysis
        """
        logger.info(f"Finding optimal business threshold for {model_name}")
        
        # Try different thresholds
        thresholds = np.arange(0.01, 1.0, 0.01)
        costs = []
        metrics = []
        
        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            
            # Calculate confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            # Calculate business costs
            fraud_loss = fn * self.fraud_cost
            false_alarm_cost = fp * self.false_positive_cost
            investigation_cost = (tp + fp) * self.investigation_cost
            total_cost = fraud_loss + false_alarm_cost + investigation_cost
            
            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            costs.append(total_cost)
            metrics.append({
                'threshold': threshold,
                'total_cost': total_cost,
                'fraud_loss': fraud_loss,
                'false_alarm_cost': false_alarm_cost,
                'investigation_cost': investigation_cost,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
            })
        
        # Find optimal threshold (minimum total cost)
        optimal_idx = np.argmin(costs)
        optimal_threshold = thresholds[optimal_idx]
        optimal_metrics = metrics[optimal_idx]
        
        self.optimal_thresholds[model_name] = optimal_threshold
        self.threshold_analysis[model_name] = {
            'thresholds': thresholds,
            'costs': costs,
            'metrics': metrics,
            'optimal_threshold': optimal_threshold,
            'optimal_metrics': optimal_metrics
        }
        
        logger.info(f"Optimal threshold for {model_name}: {optimal_threshold:.3f}")
        logger.info(f"  Total cost: ${optimal_metrics['total_cost']:,.0f}")
        logger.info(f"  F1-score: {optimal_metrics['f1_score']:.4f}")
        logger.info(f"  Precision: {optimal_metrics['precision']:.4f}")
        logger.info(f"  Recall: {optimal_metrics['recall']:.4f}")
        
        return optimal_threshold, optimal_metrics
    
    def find_pareto_optimal_thresholds(self, y_true, y_prob, model_name):
        """
        Find Pareto-optimal thresholds trading off multiple objectives.
        
        Objectives: Minimize cost, maximize F1, maximize precision, maximize recall
        """
        logger.info(f"Finding Pareto-optimal thresholds for {model_name}")
        
        if model_name not in self.threshold_analysis:
            self.find_optimal_threshold_business(y_true, y_prob, model_name)
        
        metrics = self.threshold_analysis[model_name]['metrics']
        
        # Define objectives (normalize to 0-1 scale)
        objectives = []
        for m in metrics:
            cost_normalized = 1 - (m['total_cost'] / max(met['total_cost'] for met in metrics))
            objectives.append({
                'threshold': m['threshold'],
                'cost_efficiency': cost_normalized,
                'f1_score': m['f1_score'],
                'precision': m['precision'],
                'recall': m['recall']
            })
        
        # Find Pareto-optimal solutions
        pareto_optimal = []
        
        for i, obj1 in enumerate(objectives):
            is_dominated = False
            
            for j, obj2 in enumerate(objectives):
                if i == j:
                    continue
                
                # Check if obj1 is dominated by obj2
                if (obj2['cost_efficiency'] >= obj1['cost_efficiency'] and
                    obj2['f1_score'] >= obj1['f1_score'] and
                    obj2['precision'] >= obj1['precision'] and
                    obj2['recall'] >= obj1['recall'] and
                    (obj2['cost_efficiency'] > obj1['cost_efficiency'] or
                     obj2['f1_score'] > obj1['f1_score'] or
                     obj2['precision'] > obj1['precision'] or
                     obj2['recall'] > obj1['recall'])):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_optimal.append(obj1)
        
        # Sort by threshold
        pareto_optimal.sort(key=lambda x: x['threshold'])
        
        self.threshold_analysis[model_name]['pareto_optimal'] = pareto_optimal
        
        logger.info(f"Found {len(pareto_optimal)} Pareto-optimal thresholds")
        
        return pareto_optimal
    
    def find_dynamic_thresholds(self, df, y_prob_col, context_cols=['Hour', 'Amount_Bin']):
        """
        Find optimal thresholds for different transaction contexts.
        
        Args:
            df: DataFrame with features and predictions
            y_prob_col: Column name for fraud probabilities
            context_cols: Columns defining contexts (e.g., hour, amount range)
        """
        logger.info("Finding dynamic thresholds by transaction context")
        
        # Create context bins
        df_analysis = df.copy()
        
        # Create hour bins if not exists
        if 'Hour' in context_cols and 'Hour' not in df_analysis.columns:
            df_analysis['Hour'] = (df_analysis.get('Time', 0) % (24 * 3600)) // 3600
        
        # Create amount bins if not exists
        if 'Amount_Bin' in context_cols and 'Amount_Bin' not in df_analysis.columns:
            df_analysis['Amount_Bin'] = pd.cut(
                df_analysis.get('Amount', 0), 
                bins=[0, 10, 50, 200, 1000, float('inf')],
                labels=['Very_Low', 'Low', 'Medium', 'High', 'Very_High']
            )
        
        dynamic_thresholds = {}
        
        # Find thresholds for each context
        for context_col in context_cols:
            if context_col not in df_analysis.columns:
                continue
                
            context_thresholds = {}
            
            for context_value in df_analysis[context_col].unique():
                if pd.isna(context_value):
                    continue
                
                # Filter data for this context
                context_mask = df_analysis[context_col] == context_value
                context_data = df_analysis[context_mask]
                
                if len(context_data) < 50:  # Skip contexts with too few samples
                    continue
                
                # Find optimal threshold for this context
                y_true_context = context_data['Class']
                y_prob_context = context_data[y_prob_col]
                
                threshold, metrics = self.find_optimal_threshold_business(
                    y_true_context, y_prob_context, f"{context_col}_{context_value}"
                )
                
                context_thresholds[context_value] = {
                    'threshold': threshold,
                    'metrics': metrics,
                    'sample_size': len(context_data)
                }
            
            dynamic_thresholds[context_col] = context_thresholds
        
        self.dynamic_thresholds = dynamic_thresholds
        
        # Log summary
        for context_col, thresholds in dynamic_thresholds.items():
            logger.info(f"Dynamic thresholds for {context_col}:")
            for value, data in thresholds.items():
                logger.info(f"  {value}: {data['threshold']:.3f} "
                           f"(n={data['sample_size']}, cost=${data['metrics']['total_cost']:,.0f})")
        
        return dynamic_thresholds
    
    def plot_threshold_analysis(self, model_name, save_path=None):
        """
        Plot comprehensive threshold analysis.
        
        Args:
            model_name: Name of the model to analyze
            save_path: Path to save the plot
        """
        if model_name not in self.threshold_analysis:
            logger.error(f"No threshold analysis found for {model_name}")
            return
        
        analysis = self.threshold_analysis[model_name]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        thresholds = analysis['thresholds']
        costs = analysis['costs']
        metrics = analysis['metrics']
        
        # Cost vs Threshold
        axes[0, 0].plot(thresholds, costs, 'b-', linewidth=2)
        axes[0, 0].axvline(analysis['optimal_threshold'], color='r', linestyle='--', 
                          label=f"Optimal: {analysis['optimal_threshold']:.3f}")
        axes[0, 0].set_xlabel('Threshold')
        axes[0, 0].set_ylabel('Total Business Cost ($)')
        axes[0, 0].set_title('Business Cost vs Threshold')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # F1-Score vs Threshold
        f1_scores = [m['f1_score'] for m in metrics]
        axes[0, 1].plot(thresholds, f1_scores, 'g-', linewidth=2)
        axes[0, 1].axvline(analysis['optimal_threshold'], color='r', linestyle='--')
        axes[0, 1].set_xlabel('Threshold')
        axes[0, 1].set_ylabel('F1-Score')
        axes[0, 1].set_title('F1-Score vs Threshold')
        axes[0, 1].grid(True)
        
        # Precision and Recall vs Threshold
        precisions = [m['precision'] for m in metrics]
        recalls = [m['recall'] for m in metrics]
        axes[1, 0].plot(thresholds, precisions, 'orange', label='Precision', linewidth=2)
        axes[1, 0].plot(thresholds, recalls, 'purple', label='Recall', linewidth=2)
        axes[1, 0].axvline(analysis['optimal_threshold'], color='r', linestyle='--')
        axes[1, 0].set_xlabel('Threshold')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Precision and Recall vs Threshold')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Cost Breakdown
        fraud_losses = [m['fraud_loss'] for m in metrics]
        false_alarm_costs = [m['false_alarm_cost'] for m in metrics]
        investigation_costs = [m['investigation_cost'] for m in metrics]
        
        axes[1, 1].plot(thresholds, fraud_losses, label='Fraud Loss', linewidth=2)
        axes[1, 1].plot(thresholds, false_alarm_costs, label='False Alarm Cost', linewidth=2)
        axes[1, 1].plot(thresholds, investigation_costs, label='Investigation Cost', linewidth=2)
        axes[1, 1].axvline(analysis['optimal_threshold'], color='r', linestyle='--')
        axes[1, 1].set_xlabel('Threshold')
        axes[1, 1].set_ylabel('Cost ($)')
        axes[1, 1].set_title('Cost Breakdown vs Threshold')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.suptitle(f'Threshold Analysis - {model_name}', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Threshold analysis plot saved to {save_path}")
        
        plt.show()
    
    def create_threshold_recommendation_report(self):
        """Create comprehensive threshold recommendation report."""
        logger.info("Creating threshold recommendation report")
        
        recommendations = {}
        
        for model_name, analysis in self.threshold_analysis.items():
            optimal_metrics = analysis['optimal_metrics']
            
            # Business efficiency score
            baseline_cost = len(analysis['metrics']) * self.investigation_cost  # If we investigated everything
            actual_cost = optimal_metrics['total_cost']
            efficiency_score = (baseline_cost - actual_cost) / baseline_cost * 100
            
            # Risk tolerance categories
            if optimal_metrics['recall'] >= 0.9:
                risk_category = "Conservative (High Recall)"
            elif optimal_metrics['precision'] >= 0.9:
                risk_category = "Aggressive (High Precision)"
            else:
                risk_category = "Balanced"
            
            recommendations[model_name] = {
                'optimal_threshold': analysis['optimal_threshold'],
                'business_efficiency': efficiency_score,
                'risk_category': risk_category,
                'expected_daily_alerts': optimal_metrics['tp'] + optimal_metrics['fp'],
                'fraud_detection_rate': optimal_metrics['recall'],
                'alert_precision': optimal_metrics['precision'],
                'daily_cost_saving': baseline_cost - actual_cost,
                'cost_per_transaction': actual_cost / (optimal_metrics['tp'] + optimal_metrics['tn'] + 
                                                     optimal_metrics['fp'] + optimal_metrics['fn'])
            }
        
        # Create summary DataFrame
        summary_data = []
        for model_name, rec in recommendations.items():
            summary_data.append({
                'Model': model_name,
                'Optimal_Threshold': f"{rec['optimal_threshold']:.3f}",
                'Business_Efficiency_%': f"{rec['business_efficiency']:.1f}%",
                'Risk_Category': rec['risk_category'],
                'Daily_Alerts': int(rec['expected_daily_alerts']),
                'Detection_Rate_%': f"{rec['fraud_detection_rate']*100:.1f}%",
                'Alert_Precision_%': f"{rec['alert_precision']*100:.1f}%",
                'Cost_Per_Transaction': f"${rec['cost_per_transaction']:.4f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"threshold_optimization_report_{timestamp}.csv"
        summary_df.to_csv(report_path, index=False)
        
        logger.info("THRESHOLD OPTIMIZATION REPORT")
        logger.info("=" * 60)
        print(summary_df.to_string(index=False))
        logger.info("=" * 60)
        logger.info(f"Report saved to: {report_path}")
        
        return summary_df, recommendations

def main():
    """Main function to demonstrate advanced calibration and threshold optimization."""
    logger.info("Starting Advanced Model Calibration and Threshold Optimization")
    
    # Load data
    try:
        df = pd.read_csv('creditcard.csv')
        logger.info(f"Dataset loaded: {len(df):,} transactions")
    except FileNotFoundError:
        logger.error("creditcard.csv not found. Please ensure the dataset is available.")
        return
    
    # Prepare data
    feature_columns = [col for col in df.columns if col != 'Class']
    X = df[feature_columns]
    y = df['Class']
    
    # Train-test-calibration split
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_cal, y_train, y_cal = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42)
    
    logger.info(f"Train: {len(X_train)}, Calibration: {len(X_cal)}, Test: {len(X_test)}")
    
    # Train some base models
    base_models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, scale_pos_weight=10),
        'Logistic Regression': LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
    }
    
    # Train models
    for name, model in base_models.items():
        logger.info(f"Training {name}")
        model.fit(X_train, y_train)
    
    # Model Calibration
    calibrator = ModelCalibratorAdvanced(base_models)
    calibrator.calibrate_models(X_cal, y_cal)
    
    # Plot calibration curves
    calibrator.plot_calibration_curves(X_test, y_test, save_path="calibration_curves.png")
    
    # Get best calibrated models
    best_models = {}
    for model_name in base_models.keys():
        best_model, best_method = calibrator.get_best_calibrated_model(model_name, metric='ece')
        if best_model:
            best_models[model_name] = best_model
    
    # Threshold Optimization
    threshold_optimizer = BusinessOptimalThresholds(
        fraud_cost=200, false_positive_cost=10, investigation_cost=30
    )
    
    # Find optimal thresholds for each model
    for model_name, model in best_models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Business-optimal threshold
        optimal_threshold, metrics = threshold_optimizer.find_optimal_threshold_business(
            y_test, y_prob, model_name
        )
        
        # Pareto-optimal thresholds
        pareto_thresholds = threshold_optimizer.find_pareto_optimal_thresholds(
            y_test, y_prob, model_name
        )
        
        # Plot threshold analysis
        threshold_optimizer.plot_threshold_analysis(
            model_name, save_path=f"threshold_analysis_{model_name.replace(' ', '_')}.png"
        )
    
    # Dynamic thresholds
    df_test = pd.concat([X_test, y_test], axis=1)
    
    # Add predictions for dynamic threshold analysis
    for model_name, model in best_models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        df_test[f'{model_name}_prob'] = y_prob
        
        # Find dynamic thresholds
        dynamic_thresholds = threshold_optimizer.find_dynamic_thresholds(
            df_test, f'{model_name}_prob', context_cols=['Hour', 'Amount_Bin']
        )
    
    # Generate comprehensive report
    summary_df, recommendations = threshold_optimizer.create_threshold_recommendation_report()
    
    logger.info("Advanced calibration and threshold optimization completed!")
    
    return calibrator, threshold_optimizer, summary_df

if __name__ == "__main__":
    main()