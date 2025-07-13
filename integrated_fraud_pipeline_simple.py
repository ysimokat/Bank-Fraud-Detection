#!/usr/bin/env python3
"""
Simplified Integrated Fraud Detection Pipeline
==============================================

A streamlined version that handles import errors gracefully.
Focuses on core functionality while being robust to missing dependencies.

Author: Yanhong Simokat
"""

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

# Core imports that should always work
from fraud_detection_models import FraudDetectionPipeline
from gpu_config import gpu_config

# Try importing advanced components
try:
    from enhanced_fraud_models import CostSensitiveFraudDetector as EnhancedFraudDetector
    ENHANCED_AVAILABLE = True
except ImportError:
    print("⚠️ Enhanced fraud models not available")
    ENHANCED_AVAILABLE = False

try:
    from enhanced_deep_learning import EnhancedFraudDetector as DeepLearningDetector
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    print("⚠️ Deep learning models not available")
    DEEP_LEARNING_AVAILABLE = False

try:
    from graph_neural_network import HybridGNNFraudSystem
    GNN_AVAILABLE = True
except ImportError:
    print("⚠️ Graph neural networks not available")
    GNN_AVAILABLE = False

class SimplifiedIntegratedPipeline:
    """
    Simplified fraud detection pipeline that gracefully handles missing components.
    """
    
    def __init__(self):
        self.basic_pipeline = None
        self.enhanced_detector = None
        self.deep_learning_detector = None
        self.gnn_system = None
        self.all_models = {}
        self.all_results = {}
        
        # Print GPU info
        print("🖥️ System Configuration:")
        gpu_config.print_config()
        
    def run_basic_models(self, df):
        """Run basic ML models - this should always work."""
        print("\n" + "="*60)
        print("🚀 PHASE 1: Basic Machine Learning Models")
        print("="*60)
        
        self.basic_pipeline = FraudDetectionPipeline()
        
        # Load and preprocess
        X_train, X_test, y_train, y_test = self.basic_pipeline.load_and_preprocess_data(
            'creditcard.csv'
        )
        
        # Train models
        self.basic_pipeline.train_baseline_models(X_train, y_train)
        self.basic_pipeline.train_anomaly_models(X_train, y_train)
        self.basic_pipeline.train_with_smote(X_train, y_train)
        
        # Evaluate
        self.basic_pipeline.evaluate_models(X_test, y_test)
        
        # Store models and results
        self.all_models.update(self.basic_pipeline.models)
        self.all_results.update(self.basic_pipeline.results)
        
        return X_train, X_test, y_train, y_test
    
    def run_enhanced_models_safe(self, df):
        """Run enhanced models if available."""
        if not ENHANCED_AVAILABLE:
            print("\n⏭️ Skipping enhanced models (not available)")
            return
            
        print("\n" + "="*60)
        print("🚀 PHASE 2: Enhanced Models")
        print("="*60)
        
        try:
            # Prepare data
            feature_columns = [col for col in df.columns if col not in ['Class']]
            X = df[feature_columns]
            y = df['Class']
            
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42
            )
            
            self.enhanced_detector = EnhancedFraudDetector()
            
            # Train models
            self.enhanced_detector.train_baseline_models(X_train, X_test, y_train, y_test)
            self.enhanced_detector.train_advanced_models(X_train, X_test, y_train, y_test)
            
            # Store results
            self.all_models.update(self.enhanced_detector.models)
            
            print("✅ Enhanced models completed")
            
        except Exception as e:
            print(f"⚠️ Enhanced models failed: {str(e)}")
    
    def run_deep_learning_safe(self, df):
        """Run deep learning models if available."""
        if not DEEP_LEARNING_AVAILABLE:
            print("\n⏭️ Skipping deep learning models (not available)")
            return
            
        print("\n" + "="*60)
        print("🚀 PHASE 3: Deep Learning Models")
        print("="*60)
        
        try:
            self.deep_learning_detector = DeepLearningDetector()
            
            # Prepare data
            data = self.deep_learning_detector.prepare_data(df)
            
            # Train models (reduced epochs)
            self.deep_learning_detector.train_with_focal_loss(data, epochs=20)
            self.deep_learning_detector.train_with_weighted_bce(data, epochs=20)
            self.deep_learning_detector.train_autoencoder(data, epochs=10)
            
            # Store results
            for model_name, results in self.deep_learning_detector.results.items():
                self.all_results[f'dl_{model_name}'] = {
                    'f1_score': results['f1_score'],
                    'roc_auc': results['roc_auc'],
                    'avg_precision': results.get('avg_precision', 0)
                }
            
            print("✅ Deep learning models completed")
            
        except Exception as e:
            print(f"⚠️ Deep learning failed: {str(e)}")
    
    def generate_report(self):
        """Generate final report."""
        print("\n" + "="*60)
        print("📊 FINAL REPORT")
        print("="*60)
        
        # Create comparison dataframe
        report_data = []
        for model_name, metrics in self.all_results.items():
            if isinstance(metrics, dict) and 'f1_score' in metrics:
                report_data.append({
                    'Model': model_name.replace('_', ' ').title(),
                    'F1-Score': f"{metrics['f1_score']:.4f}",
                    'ROC-AUC': f"{metrics.get('roc_auc', 0):.4f}"
                })
        
        if report_data:
            df_report = pd.DataFrame(report_data)
            df_report = df_report.sort_values('F1-Score', ascending=False)
            
            print("\n🏆 Model Performance:")
            print(df_report.to_string(index=False))
        
        # Save results
        joblib.dump(self.all_models, 'fraud_models.joblib')
        if hasattr(self.basic_pipeline, 'scaler'):
            joblib.dump(self.basic_pipeline.scaler, 'scaler.joblib')
        joblib.dump(self.all_results, 'model_results.joblib')
        
        print("\n✅ Models saved successfully!")
        print(f"📊 Total models trained: {len(self.all_models)}")
        
        return df_report if report_data else None
    
    def run_pipeline(self, include_enhanced=True, include_deep_learning=True):
        """Run the simplified pipeline."""
        print("\n" + "🚀"*20)
        print("SIMPLIFIED INTEGRATED FRAUD DETECTION PIPELINE")
        print("🚀"*20)
        
        # Load data
        print("\n📊 Loading dataset...")
        df = pd.read_csv('creditcard.csv')
        print(f"✅ Loaded {len(df):,} transactions")
        
        # Phase 1: Basic Models (always run)
        self.run_basic_models(df)
        
        # Phase 2: Enhanced Models (if available)
        if include_enhanced:
            self.run_enhanced_models_safe(df)
        
        # Phase 3: Deep Learning (if available)
        if include_deep_learning:
            self.run_deep_learning_safe(df)
        
        # Generate report
        report_df = self.generate_report()
        
        print("\n" + "🎉"*20)
        print("PIPELINE COMPLETED!")
        print("🎉"*20)
        
        return report_df

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Simplified Integrated Fraud Pipeline')
    parser.add_argument('--skip-enhanced', action='store_true',
                       help='Skip enhanced models')
    parser.add_argument('--skip-dl', action='store_true',
                       help='Skip deep learning models')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode - basic models only')
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = SimplifiedIntegratedPipeline()
    
    # Run with options
    if args.quick:
        print("⚡ Running in QUICK MODE (Basic models only)")
        report = pipeline.run_pipeline(include_enhanced=False, include_deep_learning=False)
    else:
        report = pipeline.run_pipeline(
            include_enhanced=not args.skip_enhanced,
            include_deep_learning=not args.skip_dl
        )
    
    return pipeline, report

if __name__ == "__main__":
    pipeline, report = main()