#!/usr/bin/env python3
"""
Quick test to verify XGBoost early stopping fix
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
import xgboost as xgb

print("Testing XGBoost early stopping fix...")

# Create sample data
np.random.seed(42)
n_samples = 1000
n_features = 10

X = np.random.randn(n_samples, n_features)
# Create imbalanced target
y = np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05])

print(f"Sample data: {n_samples} samples, {n_features} features")
print(f"Class distribution: {np.bincount(y)}")

# Test 1: XGBoost with early stopping (should work for normal fit)
print("\n1. Testing XGBoost with early stopping...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

xgb_early = xgb.XGBClassifier(
    n_estimators=100,
    early_stopping_rounds=10,
    eval_metric='logloss',
    random_state=42
)

try:
    xgb_early.fit(
        X_train_split, y_train_split,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    print("✓ XGBoost with early stopping: SUCCESS")
    print(f"  Stopped at iteration: {xgb_early.best_iteration}")
except Exception as e:
    print(f"✗ XGBoost with early stopping: FAILED - {e}")

# Test 2: XGBoost without early stopping for CV (should work)
print("\n2. Testing XGBoost without early stopping for CV...")
xgb_no_early = xgb.XGBClassifier(
    n_estimators=50,  # Fixed number
    random_state=42
)

try:
    cv_scores = cross_val_score(xgb_no_early, X_train, y_train, cv=3, scoring='f1')
    print("✓ XGBoost cross-validation: SUCCESS")
    print(f"  CV F1 scores: {cv_scores}")
    print(f"  Mean F1: {cv_scores.mean():.4f}")
except Exception as e:
    print(f"✗ XGBoost cross-validation: FAILED - {e}")

# Test 3: The problematic case (early stopping + CV) - should fail
print("\n3. Testing problematic case (early stopping + CV)...")
try:
    cv_scores = cross_val_score(xgb_early, X_train, y_train, cv=3, scoring='f1')
    print("✓ XGBoost early stopping + CV: UNEXPECTED SUCCESS")
except Exception as e:
    print("✓ XGBoost early stopping + CV: EXPECTED FAILURE")
    print(f"  Error: {str(e)[:100]}...")

print("\nConclusion:")
print("- Use early stopping for final model training")
print("- Use fixed n_estimators for cross-validation")
print("- This is the fix applied to enhanced_fraud_models.py")