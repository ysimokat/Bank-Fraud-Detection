#!/usr/bin/env python3
"""
Credit Card Fraud Detection - Data Exploration
===============================================

This script performs initial data exploration of the credit card fraud dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

def load_and_explore_data(file_path):
    """Load and perform initial exploration of the credit card fraud dataset."""
    print("🔍 Loading Credit Card Fraud Dataset...")
    print("=" * 50)
    
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Basic information
    print(f"📊 Dataset Shape: {df.shape}")
    print(f"📈 Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print()
    
    print("📋 Dataset Info:")
    print("-" * 30)
    print(df.info())
    print()
    
    print("📈 Statistical Summary:")
    print("-" * 30)
    print(df.describe())
    print()
    
    # Check for missing values
    print("❓ Missing Values:")
    print("-" * 30)
    missing_values = df.isnull().sum()
    if missing_values.sum() == 0:
        print("✅ No missing values found!")
    else:
        print(missing_values[missing_values > 0])
    print()
    
    # Check target distribution
    print("🎯 Target Distribution (Class):")
    print("-" * 30)
    fraud_counts = df['Class'].value_counts()
    fraud_percentage = df['Class'].value_counts(normalize=True) * 100
    
    print(f"Normal transactions (0): {fraud_counts[0]:,} ({fraud_percentage[0]:.3f}%)")
    print(f"Fraudulent transactions (1): {fraud_counts[1]:,} ({fraud_percentage[1]:.3f}%)")
    print(f"Fraud rate: {fraud_percentage[1]:.4f}%")
    print()
    
    # Feature columns analysis
    print("🔢 Feature Analysis:")
    print("-" * 30)
    feature_cols = [col for col in df.columns if col not in ['Time', 'Amount', 'Class']]
    print(f"PCA Features (V1-V28): {len(feature_cols)} features")
    print(f"Time feature: {'Time' in df.columns}")
    print(f"Amount feature: {'Amount' in df.columns}")
    print()
    
    # Time analysis
    print("⏰ Time Analysis:")
    print("-" * 30)
    print(f"Time range: {df['Time'].min():.0f} to {df['Time'].max():.0f} seconds")
    print(f"Duration: {(df['Time'].max() - df['Time'].min()) / 3600:.1f} hours")
    print()
    
    # Amount analysis
    print("💰 Amount Analysis:")
    print("-" * 30)
    print(f"Amount range: ${df['Amount'].min():.2f} to ${df['Amount'].max():,.2f}")
    print(f"Mean amount: ${df['Amount'].mean():.2f}")
    print(f"Median amount: ${df['Amount'].median():.2f}")
    print()
    
    print("💰 Amount by Class:")
    print("-" * 30)
    amount_by_class = df.groupby('Class')['Amount'].agg(['mean', 'median', 'std', 'min', 'max'])
    print(amount_by_class)
    print()
    
    return df

def visualize_data_distribution(df):
    """Create visualizations for data distribution analysis."""
    print("📊 Creating Data Distribution Visualizations...")
    print("=" * 50)
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Class distribution
    plt.subplot(3, 3, 1)
    df['Class'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
    plt.title('Class Distribution\n(0: Normal, 1: Fraud)', fontsize=12, fontweight='bold')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    
    # 2. Time distribution
    plt.subplot(3, 3, 2)
    plt.hist(df['Time'], bins=50, alpha=0.7, color='lightblue', edgecolor='black')
    plt.title('Time Distribution', fontsize=12, fontweight='bold')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Frequency')
    
    # 3. Amount distribution (log scale)
    plt.subplot(3, 3, 3)
    amount_no_zero = df[df['Amount'] > 0]['Amount']
    plt.hist(np.log10(amount_no_zero), bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.title('Amount Distribution (Log Scale)', fontsize=12, fontweight='bold')
    plt.xlabel('Log10(Amount)')
    plt.ylabel('Frequency')
    
    # 4. Amount by Class (box plot)
    plt.subplot(3, 3, 4)
    sns.boxplot(data=df, x='Class', y='Amount')
    plt.title('Amount Distribution by Class', fontsize=12, fontweight='bold')
    plt.yscale('log')
    plt.ylabel('Amount (Log Scale)')
    
    # 5. Time vs Fraud (scatter plot sample)
    plt.subplot(3, 3, 5)
    sample_normal = df[df['Class'] == 0].sample(1000, random_state=42)
    sample_fraud = df[df['Class'] == 1]
    plt.scatter(sample_normal['Time'], sample_normal['Amount'], alpha=0.5, s=1, label='Normal', color='blue')
    plt.scatter(sample_fraud['Time'], sample_fraud['Amount'], alpha=0.8, s=10, label='Fraud', color='red')
    plt.title('Time vs Amount (Sample)', fontsize=12, fontweight='bold')
    plt.xlabel('Time')
    plt.ylabel('Amount')
    plt.legend()
    plt.yscale('log')
    
    # 6. Correlation heatmap (subset of features)
    plt.subplot(3, 3, 6)
    features_subset = ['V1', 'V2', 'V3', 'V4', 'V5', 'Amount', 'Class']
    correlation_matrix = df[features_subset].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
    plt.title('Feature Correlation (Subset)', fontsize=12, fontweight='bold')
    
    # 7. Feature distribution comparison (V1-V5)
    plt.subplot(3, 3, 7)
    features_to_plot = ['V1', 'V2', 'V3', 'V4', 'V5']
    for i, feature in enumerate(features_to_plot):
        plt.hist(df[df['Class'] == 0][feature], bins=30, alpha=0.5, 
                label=f'{feature} Normal' if i == 0 else None, color=f'C{i}', density=True)
        plt.hist(df[df['Class'] == 1][feature], bins=30, alpha=0.7, 
                label=f'{feature} Fraud' if i == 0 else None, color=f'C{i}', 
                linestyle='--', histtype='step', density=True)
    plt.title('Feature Distributions (V1-V5)', fontsize=12, fontweight='bold')
    plt.xlabel('Feature Value')
    plt.ylabel('Density')
    plt.legend()
    
    # 8. Fraud distribution over time
    plt.subplot(3, 3, 8)
    time_bins = pd.cut(df['Time'], bins=20)
    fraud_rate_by_time = df.groupby(time_bins)['Class'].mean()
    fraud_rate_by_time.plot(kind='line', marker='o', color='red')
    plt.title('Fraud Rate Over Time', fontsize=12, fontweight='bold')
    plt.xlabel('Time Bins')
    plt.ylabel('Fraud Rate')
    plt.xticks(rotation=45)
    
    # 9. Amount distribution by class (violin plot)
    plt.subplot(3, 3, 9)
    df_log_amount = df.copy()
    df_log_amount['Amount_log'] = np.log10(df_log_amount['Amount'] + 1)
    sns.violinplot(data=df_log_amount, x='Class', y='Amount_log')
    plt.title('Amount Distribution by Class\n(Log Scale)', fontsize=12, fontweight='bold')
    plt.ylabel('Log10(Amount + 1)')
    
    plt.tight_layout()
    plt.savefig('/mnt/c/Users/Helen/Desktop/CodeMonkey/AI_Practice/Bank_Fraud_Detection/data_exploration.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Visualizations saved as 'data_exploration.png'")

def analyze_feature_importance_pca(df):
    """Analyze the PCA features and their distributions."""
    print("🔬 Analyzing PCA Features...")
    print("=" * 50)
    
    # Get PCA features
    pca_features = [col for col in df.columns if col.startswith('V')]
    
    # Calculate statistics for each class
    normal_stats = df[df['Class'] == 0][pca_features].describe()
    fraud_stats = df[df['Class'] == 1][pca_features].describe()
    
    # Calculate effect sizes (difference in means / pooled standard deviation)
    effect_sizes = []
    for feature in pca_features:
        mean_normal = normal_stats.loc['mean', feature]
        mean_fraud = fraud_stats.loc['mean', feature]
        std_normal = normal_stats.loc['std', feature]
        std_fraud = fraud_stats.loc['std', feature]
        
        pooled_std = np.sqrt((std_normal**2 + std_fraud**2) / 2)
        effect_size = abs(mean_normal - mean_fraud) / pooled_std
        effect_sizes.append(effect_size)
    
    # Create a DataFrame with effect sizes
    effect_df = pd.DataFrame({
        'Feature': pca_features,
        'Effect_Size': effect_sizes
    }).sort_values('Effect_Size', ascending=False)
    
    print("🎯 Top 10 Most Discriminative PCA Features:")
    print("-" * 40)
    print(effect_df.head(10).to_string(index=False))
    print()
    
    # Visualize top discriminative features
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.ravel()
    
    top_features = effect_df.head(10)['Feature'].tolist()
    
    for i, feature in enumerate(top_features):
        normal_data = df[df['Class'] == 0][feature]
        fraud_data = df[df['Class'] == 1][feature]
        
        axes[i].hist(normal_data, bins=50, alpha=0.7, label='Normal', color='blue', density=True)
        axes[i].hist(fraud_data, bins=50, alpha=0.7, label='Fraud', color='red', density=True)
        axes[i].set_title(f'{feature}\n(Effect Size: {effect_df[effect_df.Feature == feature].Effect_Size.iloc[0]:.3f})')
        axes[i].legend()
        axes[i].set_ylabel('Density')
    
    plt.tight_layout()
    plt.savefig('/mnt/c/Users/Helen/Desktop/CodeMonkey/AI_Practice/Bank_Fraud_Detection/feature_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Feature analysis saved as 'feature_analysis.png'")
    
    return effect_df

def main():
    """Main function to run the data exploration."""
    file_path = '/mnt/c/Users/Helen/Desktop/CodeMonkey/AI_Practice/Bank_Fraud_Detection/creditcard.csv'
    
    # Load and explore data
    df = load_and_explore_data(file_path)
    
    # Create visualizations
    visualize_data_distribution(df)
    
    # Analyze PCA features
    effect_df = analyze_feature_importance_pca(df)
    
    print("🎉 Data exploration completed successfully!")
    print("📊 Key findings:")
    print(f"   • Dataset contains {len(df):,} transactions")
    print(f"   • Fraud rate: {(df['Class'].sum() / len(df) * 100):.4f}%")
    print(f"   • Highly imbalanced dataset")
    print(f"   • Top discriminative feature: {effect_df.iloc[0]['Feature']}")
    
    return df, effect_df

if __name__ == "__main__":
    df, effect_df = main()