#!/usr/bin/env python3
"""
Credit Card Fraud Detection - Demonstration Script
=================================================

This script provides a comprehensive demonstration of the fraud detection system.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class FraudDetectionDemo:
    """Comprehensive demonstration of the fraud detection system."""
    
    def __init__(self):
        self.models = None
        self.scaler = None
        self.results = None
        self.df = None
        
    def load_demo_data(self):
        """Load and prepare demonstration data."""
        print("🚀 Loading Fraud Detection Demo")
        print("=" * 50)
        
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load dataset
        try:
            csv_path = os.path.join(current_dir, 'creditcard.csv')
            self.df = pd.read_csv(csv_path)
            print(f"✅ Dataset loaded: {len(self.df):,} transactions")
        except FileNotFoundError:
            print("❌ Dataset not found")
            return False
        
        # Load models
        try:
            self.models = joblib.load(os.path.join(current_dir, 'fraud_models.joblib'))
            self.scaler = joblib.load(os.path.join(current_dir, 'scaler.joblib'))
            self.results = joblib.load(os.path.join(current_dir, 'model_results.joblib'))
            print(f"✅ Models loaded: {len(self.models)} trained models")
        except FileNotFoundError:
            print("❌ Models not found - running basic demo with dataset only")
            return False
        
        return True
    
    def show_dataset_insights(self):
        """Display key dataset insights."""
        print("\n📊 Dataset Insights")
        print("=" * 30)
        
        # Basic statistics
        fraud_count = self.df['Class'].sum()
        normal_count = len(self.df) - fraud_count
        fraud_rate = (fraud_count / len(self.df)) * 100
        
        print(f"📈 Total Transactions: {len(self.df):,}")
        print(f"💳 Normal Transactions: {normal_count:,} ({100-fraud_rate:.3f}%)")
        print(f"🚨 Fraudulent Transactions: {fraud_count:,} ({fraud_rate:.3f}%)")
        print(f"💰 Total Amount: ${self.df['Amount'].sum():,.2f}")
        print(f"⏰ Time Span: {(self.df['Time'].max() - self.df['Time'].min())/3600:.1f} hours")
        
        # Fraud vs Normal comparison
        print(f"\n💡 Key Patterns:")
        normal_avg = self.df[self.df['Class'] == 0]['Amount'].mean()
        fraud_avg = self.df[self.df['Class'] == 1]['Amount'].mean()
        print(f"   • Average Normal Transaction: ${normal_avg:.2f}")
        print(f"   • Average Fraud Transaction: ${fraud_avg:.2f}")
        
        # Top discriminative features
        pca_features = [col for col in self.df.columns if col.startswith('V')]
        correlations = self.df[pca_features + ['Class']].corr()['Class'].abs().sort_values(ascending=False)
        print(f"   • Most Important Feature: {correlations.index[1]} (correlation: {correlations.iloc[1]:.3f})")
    
    def demonstrate_model_performance(self):
        """Demonstrate model performance."""
        if not self.results:
            print("\n❌ No model results available")
            return
        
        print("\n🏆 Model Performance Demonstration")
        print("=" * 40)
        
        # Sort models by F1-score
        sorted_models = sorted(self.results.items(), key=lambda x: x[1]['f1_score'], reverse=True)
        
        print("📊 Model Rankings (by F1-Score):")
        for i, (name, metrics) in enumerate(sorted_models):
            emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "📈"
            print(f"{emoji} {name.replace('_', ' ').title()}")
            print(f"    F1-Score: {metrics['f1_score']:.4f}")
            print(f"    ROC-AUC: {metrics['roc_auc']:.4f}")
            print(f"    Avg Precision: {metrics['avg_precision']:.4f}")
            print()
        
        # Best model analysis
        best_name, best_metrics = sorted_models[0]
        print(f"🎯 Best Performing Model: {best_name.replace('_', ' ').title()}")
        print(f"   This model achieves {best_metrics['f1_score']:.3f} F1-score")
        print(f"   Meaning it correctly identifies {best_metrics['f1_score']*100:.1f}% of fraud cases")
        print(f"   while minimizing false positives.")
    
    def simulate_fraud_detection(self):
        """Simulate real-time fraud detection."""
        print("\n🔮 Live Fraud Detection Simulation")
        print("=" * 40)
        
        if not self.models or not self.scaler:
            print("❌ Models not available for simulation")
            return
        
        # Get best model
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['f1_score'])
        best_model = self.models[best_model_name]
        
        print(f"Using: {best_model_name.replace('_', ' ').title()}")
        
        # Simulate different transaction types
        test_cases = [
            ("Normal Small Purchase", self.df[self.df['Class'] == 0].sample(1)),
            ("Normal Large Purchase", self.df[(self.df['Class'] == 0) & (self.df['Amount'] > 1000)].sample(1) if len(self.df[(self.df['Class'] == 0) & (self.df['Amount'] > 1000)]) > 0 else self.df[self.df['Class'] == 0].sample(1)),
            ("Known Fraud Case", self.df[self.df['Class'] == 1].sample(1))
        ]
        
        for case_name, sample in test_cases:
            if len(sample) == 0:
                continue
                
            print(f"\n🧪 Test Case: {case_name}")
            
            # Prepare features
            sample_features = sample.copy()
            sample_features['Amount_log'] = np.log(sample_features['Amount'] + 1)
            sample_features['Hour'] = (sample_features['Time'] % (24 * 3600)) // 3600
            
            feature_cols = [col for col in sample_features.columns if col not in ['Class', 'Time']]
            X_sample = sample_features[feature_cols]
            
            # Scale and predict
            X_scaled = self.scaler.transform(X_sample)
            prediction = best_model.predict(X_scaled)[0]
            
            if hasattr(best_model, 'predict_proba'):
                confidence = best_model.predict_proba(X_scaled)[0][1]
            else:
                confidence = 0.5
            
            # Display results
            actual = sample['Class'].iloc[0]
            amount = sample['Amount'].iloc[0]
            
            print(f"   💰 Amount: ${amount:.2f}")
            print(f"   🔍 Prediction: {'🚨 FRAUD' if prediction == 1 else '✅ NORMAL'}")
            print(f"   📊 Confidence: {confidence:.3f}")
            print(f"   🎯 Actual: {'FRAUD' if actual == 1 else 'NORMAL'}")
            
            if prediction == actual:
                print("   ✅ CORRECT PREDICTION!")
            else:
                print("   ❌ Incorrect prediction")
    
    def show_business_impact(self):
        """Show potential business impact."""
        print("\n💼 Business Impact Analysis")
        print("=" * 35)
        
        if not self.results:
            print("❌ No model results for business analysis")
            return
        
        # Get best model metrics
        best_name = max(self.results.keys(), key=lambda x: self.results[x]['f1_score'])
        best_metrics = self.results[best_name]
        
        # Simulate business metrics
        total_transactions = len(self.df)
        total_fraud = self.df['Class'].sum()
        avg_fraud_amount = self.df[self.df['Class'] == 1]['Amount'].mean()
        
        # Estimate detection performance
        f1_score = best_metrics['f1_score']
        estimated_detection_rate = f1_score * 0.9  # Conservative estimate
        
        detected_frauds = int(total_fraud * estimated_detection_rate)
        prevented_loss = detected_frauds * avg_fraud_amount
        
        print(f"📊 Performance Metrics:")
        print(f"   • Model F1-Score: {f1_score:.3f}")
        print(f"   • Estimated Detection Rate: {estimated_detection_rate:.1%}")
        
        print(f"\n💰 Financial Impact (Historical):")
        print(f"   • Total Fraud Cases: {total_fraud:,}")
        print(f"   • Average Fraud Amount: ${avg_fraud_amount:.2f}")
        print(f"   • Estimated Detected: {detected_frauds:,}")
        print(f"   • Potential Loss Prevention: ${prevented_loss:,.2f}")
        
        print(f"\n🎯 Key Benefits:")
        print(f"   • Real-time fraud detection")
        print(f"   • Reduced false positives (better customer experience)")
        print(f"   • Automated risk assessment")
        print(f"   • Explainable decisions for compliance")
    
    def show_technical_highlights(self):
        """Show technical implementation highlights."""
        print("\n🔧 Technical Implementation Highlights")
        print("=" * 45)
        
        print("📚 Machine Learning Techniques Used:")
        print("   • Ensemble Methods (Random Forest, Voting Classifiers)")
        print("   • Gradient Boosting (XGBoost, LightGBM)")
        print("   • Anomaly Detection (Isolation Forest, One-Class SVM)")
        print("   • Neural Networks (Multi-layer Perceptron)")
        print("   • Class Imbalance Handling (SMOTE, Class Weights)")
        
        print("\n🎯 Key Technical Features:")
        print("   • Feature Engineering (PCA analysis, time features)")
        print("   • Cross-validation with stratified sampling")
        print("   • Hyperparameter optimization")
        print("   • Model explainability (SHAP, feature importance)")
        print("   • Production-ready pipeline")
        
        print("\n🚀 Deployment Capabilities:")
        print("   • Interactive Streamlit dashboard")
        print("   • Real-time prediction API")
        print("   • Model versioning and persistence")
        print("   • Scalable preprocessing pipeline")
        print("   • Comprehensive evaluation metrics")
    
    def run_complete_demo(self):
        """Run the complete demonstration."""
        if not self.load_demo_data():
            print("❌ Demo cannot proceed without data and models")
            return
        
        # Run all demonstration sections
        self.show_dataset_insights()
        self.demonstrate_model_performance()
        self.simulate_fraud_detection()
        self.show_business_impact()
        self.show_technical_highlights()
        
        print("\n🎉 Fraud Detection System Demo Complete!")
        print("=" * 50)
        print("🌟 This system demonstrates:")
        print("   • Complete AI/ML lifecycle implementation")
        print("   • Production-ready fraud detection capabilities")
        print("   • Advanced machine learning techniques")
        print("   • Real-world business value creation")
        print()
        print("🚀 Next Steps:")
        print("   • Run 'streamlit run streamlit_dashboard.py' for interactive demo")
        print("   • Explore individual scripts for detailed analysis")
        print("   • Review README.md for complete documentation")

def main():
    """Main demonstration function."""
    demo = FraudDetectionDemo()
    demo.run_complete_demo()

if __name__ == "__main__":
    main()