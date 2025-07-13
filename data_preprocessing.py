#!/usr/bin/env python3
"""
Credit Card Fraud Detection - Data Preprocessing
===============================================

This script handles data preprocessing, feature engineering, and train/test splitting.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
import joblib
import warnings
warnings.filterwarnings('ignore')

class FraudDataPreprocessor:
    """
    A comprehensive data preprocessor for credit card fraud detection.
    """
    
    def __init__(self):
        self.scaler = None
        self.feature_columns = None
        self.target_column = 'Class'
        
    def load_data(self, file_path):
        """Load the credit card fraud dataset."""
        print("📊 Loading dataset...")
        df = pd.read_csv(file_path)
        print(f"✅ Loaded {len(df):,} transactions")
        return df
    
    def engineer_features(self, df):
        """Create additional features from existing data."""
        print("🔧 Engineering features...")
        
        df_enhanced = df.copy()
        
        # Time-based features
        df_enhanced['Hour'] = (df_enhanced['Time'] % (24 * 3600)) // 3600
        df_enhanced['Day'] = df_enhanced['Time'] // (24 * 3600)
        
        # Amount-based features
        df_enhanced['Amount_log'] = np.log(df_enhanced['Amount'] + 1)
        df_enhanced['Amount_sqrt'] = np.sqrt(df_enhanced['Amount'])
        
        # Interaction features with top discriminative features
        top_features = ['V14', 'V4', 'V11', 'V12', 'V10']
        for i, feat1 in enumerate(top_features):
            for feat2 in top_features[i+1:]:
                df_enhanced[f'{feat1}_{feat2}_interaction'] = df_enhanced[feat1] * df_enhanced[feat2]
        
        # Statistical features
        pca_features = [col for col in df.columns if col.startswith('V')]
        df_enhanced['V_mean'] = df_enhanced[pca_features].mean(axis=1)
        df_enhanced['V_std'] = df_enhanced[pca_features].std(axis=1)
        df_enhanced['V_max'] = df_enhanced[pca_features].max(axis=1)
        df_enhanced['V_min'] = df_enhanced[pca_features].min(axis=1)
        
        print(f"✅ Created {len(df_enhanced.columns) - len(df.columns)} new features")
        return df_enhanced
    
    def detect_outliers(self, df, method='iqr'):
        """Detect outliers using IQR method."""
        print("🔍 Detecting outliers...")
        
        outlier_indices = set()
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns if col != self.target_column]
        
        for column in numeric_columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)].index
            outlier_indices.update(outliers)
        
        print(f"🔍 Found {len(outlier_indices)} outlier transactions ({len(outlier_indices)/len(df)*100:.2f}%)")
        return list(outlier_indices)
    
    def prepare_data_splits(self, df, test_size=0.2, val_size=0.2, random_state=42):
        """Create train/validation/test splits with stratification."""
        print("📊 Creating data splits...")
        
        # Separate features and target
        X = df.drop([self.target_column], axis=1)
        y = df[self.target_column]
        
        # Store feature columns
        self.feature_columns = X.columns.tolist()
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Second split: train vs val
        adjusted_val_size = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=adjusted_val_size, random_state=random_state, stratify=y_temp
        )
        
        print(f"✅ Train: {len(X_train):,} ({y_train.sum()} frauds)")
        print(f"✅ Validation: {len(X_val):,} ({y_val.sum()} frauds)")
        print(f"✅ Test: {len(X_test):,} ({y_test.sum()} frauds)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def scale_features(self, X_train, X_val, X_test, scaler_type='standard'):
        """Scale features using specified scaler."""
        print(f"⚖️ Scaling features using {scaler_type} scaler...")
        
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError("scaler_type must be 'standard' or 'robust'")
        
        # Fit on training data only
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrames
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=self.feature_columns, index=X_train.index)
        X_val_scaled = pd.DataFrame(X_val_scaled, columns=self.feature_columns, index=X_val.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=self.feature_columns, index=X_test.index)
        
        print("✅ Feature scaling completed")
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def handle_imbalance(self, X_train, y_train, method='smote'):
        """Handle class imbalance using various techniques."""
        print(f"⚖️ Handling class imbalance using {method}...")
        
        original_fraud_count = y_train.sum()
        original_normal_count = len(y_train) - original_fraud_count
        
        if method == 'smote':
            sampler = SMOTE(random_state=42)
        elif method == 'adasyn':
            sampler = ADASYN(random_state=42)
        elif method == 'smotetomek':
            sampler = SMOTETomek(random_state=42)
        elif method == 'undersample':
            sampler = RandomUnderSampler(random_state=42)
        elif method == 'none':
            print("⚠️ No resampling applied")
            return X_train, y_train
        else:
            raise ValueError("method must be one of: 'smote', 'adasyn', 'smotetomek', 'undersample', 'none'")
        
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
        
        # Convert back to DataFrame
        X_resampled = pd.DataFrame(X_resampled, columns=self.feature_columns)
        y_resampled = pd.Series(y_resampled, name=self.target_column)
        
        new_fraud_count = y_resampled.sum()
        new_normal_count = len(y_resampled) - new_fraud_count
        
        print(f"📊 Original: {original_normal_count:,} normal, {original_fraud_count:,} fraud")
        print(f"📊 Resampled: {new_normal_count:,} normal, {new_fraud_count:,} fraud")
        print(f"✅ New fraud ratio: {new_fraud_count/len(y_resampled)*100:.2f}%")
        
        return X_resampled, y_resampled
    
    def save_preprocessor(self, filepath='preprocessor.joblib'):
        """Save the preprocessor for later use."""
        joblib.dump({
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }, filepath)
        print(f"💾 Preprocessor saved to {filepath}")
    
    def load_preprocessor(self, filepath='preprocessor.joblib'):
        """Load a saved preprocessor."""
        data = joblib.load(filepath)
        self.scaler = data['scaler']
        self.feature_columns = data['feature_columns']
        print(f"📂 Preprocessor loaded from {filepath}")

def create_multiple_datasets(df, preprocessor):
    """Create multiple datasets with different preprocessing strategies."""
    print("🔄 Creating multiple preprocessing strategies...")
    
    datasets = {}
    
    # Strategy 1: Minimal preprocessing
    print("\n📊 Strategy 1: Minimal Preprocessing")
    df_minimal = preprocessor.engineer_features(df)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.prepare_data_splits(df_minimal)
    X_train_scaled, X_val_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_val, X_test, 'robust')
    
    datasets['minimal'] = {
        'X_train': X_train_scaled, 'X_val': X_val_scaled, 'X_test': X_test_scaled,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test
    }
    
    # Strategy 2: SMOTE balancing
    print("\n📊 Strategy 2: SMOTE Balancing")
    X_train_smote, y_train_smote = preprocessor.handle_imbalance(X_train_scaled, y_train, 'smote')
    datasets['smote'] = {
        'X_train': X_train_smote, 'X_val': X_val_scaled, 'X_test': X_test_scaled,
        'y_train': y_train_smote, 'y_val': y_val, 'y_test': y_test
    }
    
    # Strategy 3: Undersampling
    print("\n📊 Strategy 3: Undersampling")
    X_train_under, y_train_under = preprocessor.handle_imbalance(X_train_scaled, y_train, 'undersample')
    datasets['undersample'] = {
        'X_train': X_train_under, 'X_val': X_val_scaled, 'X_test': X_test_scaled,
        'y_train': y_train_under, 'y_val': y_val, 'y_test': y_test
    }
    
    # Strategy 4: SMOTETomek (hybrid)
    print("\n📊 Strategy 4: SMOTETomek")
    X_train_hybrid, y_train_hybrid = preprocessor.handle_imbalance(X_train_scaled, y_train, 'smotetomek')
    datasets['smotetomek'] = {
        'X_train': X_train_hybrid, 'X_val': X_val_scaled, 'X_test': X_test_scaled,
        'y_train': y_train_hybrid, 'y_val': y_val, 'y_test': y_test
    }
    
    return datasets

def main():
    """Main preprocessing pipeline."""
    print("🚀 Starting Credit Card Fraud Detection Preprocessing Pipeline")
    print("=" * 70)
    
    # Initialize preprocessor
    preprocessor = FraudDataPreprocessor()
    
    # Load data
    df = preprocessor.load_data('/mnt/c/Users/Helen/Desktop/CodeMonkey/AI_Practice/Bank_Fraud_Detection/creditcard.csv')
    
    # Create multiple datasets
    datasets = create_multiple_datasets(df, preprocessor)
    
    # Save datasets
    print("\n💾 Saving preprocessed datasets...")
    for strategy, data in datasets.items():
        joblib.dump(data, f'/mnt/c/Users/Helen/Desktop/CodeMonkey/AI_Practice/Bank_Fraud_Detection/dataset_{strategy}.joblib')
        print(f"✅ Saved dataset_{strategy}.joblib")
    
    # Save preprocessor
    preprocessor.save_preprocessor('/mnt/c/Users/Helen/Desktop/CodeMonkey/AI_Practice/Bank_Fraud_Detection/preprocessor.joblib')
    
    print("\n🎉 Preprocessing pipeline completed successfully!")
    print(f"📊 Created {len(datasets)} different preprocessing strategies")
    
    return datasets, preprocessor

if __name__ == "__main__":
    datasets, preprocessor = main()