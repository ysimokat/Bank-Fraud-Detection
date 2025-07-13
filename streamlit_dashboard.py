#!/usr/bin/env python3
"""
Credit Card Fraud Detection - Interactive Streamlit Dashboard
============================================================

This script creates an interactive web dashboard for demonstrating fraud detection models.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .fraud-alert {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f44336;
    }
    .normal-alert {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load the credit card fraud dataset."""
    try:
        df = pd.read_csv('/mnt/c/Users/Helen/Desktop/CodeMonkey/AI_Practice/Bank_Fraud_Detection/creditcard.csv')
        return df
    except FileNotFoundError:
        st.error("Dataset not found. Please ensure creditcard.csv is in the correct directory.")
        return None

@st.cache_data
def load_models():
    """Load the trained models."""
    try:
        models = joblib.load('/mnt/c/Users/Helen/Desktop/CodeMonkey/AI_Practice/Bank_Fraud_Detection/fraud_models.joblib')
        scaler = joblib.load('/mnt/c/Users/Helen/Desktop/CodeMonkey/AI_Practice/Bank_Fraud_Detection/scaler.joblib')
        results = joblib.load('/mnt/c/Users/Helen/Desktop/CodeMonkey/AI_Practice/Bank_Fraud_Detection/model_results.joblib')
        return models, scaler, results
    except FileNotFoundError:
        st.error("Models not found. Please run the training script first.")
        return None, None, None

def create_dataset_overview(df):
    """Create dataset overview section."""
    st.header("📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Transactions", f"{len(df):,}")
    
    with col2:
        fraud_count = df['Class'].sum()
        st.metric("Fraudulent Transactions", f"{fraud_count:,}")
    
    with col3:
        fraud_rate = (fraud_count / len(df)) * 100
        st.metric("Fraud Rate", f"{fraud_rate:.3f}%")
    
    with col4:
        duration = (df['Time'].max() - df['Time'].min()) / 3600
        st.metric("Duration (hours)", f"{duration:.1f}")
    
    # Class distribution chart
    fig = px.pie(
        values=df['Class'].value_counts().values,
        names=['Normal', 'Fraud'],
        title="Transaction Class Distribution",
        color_discrete_map={'Normal': '#2E86AB', 'Fraud': '#A23B72'}
    )
    st.plotly_chart(fig, use_container_width=True)

def create_eda_section(df):
    """Create exploratory data analysis section."""
    st.header("🔍 Exploratory Data Analysis")
    
    tab1, tab2, tab3 = st.tabs(["💰 Amount Analysis", "⏰ Time Analysis", "🎯 Feature Analysis"])
    
    with tab1:
        # Amount distribution
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                df[df['Amount'] > 0], 
                x='Amount', 
                nbins=50,
                title="Amount Distribution (Non-zero)",
                log_x=True
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(
                df, 
                x='Class', 
                y='Amount',
                title="Amount by Transaction Class"
            )
            fig.update_yaxis(type="log")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Time analysis
        df_time = df.copy()
        df_time['Hour'] = (df_time['Time'] % (24 * 3600)) // 3600
        
        hourly_fraud = df_time.groupby(['Hour', 'Class']).size().unstack(fill_value=0)
        hourly_fraud_rate = hourly_fraud[1] / (hourly_fraud[0] + hourly_fraud[1]) * 100
        
        fig = px.line(
            x=hourly_fraud_rate.index,
            y=hourly_fraud_rate.values,
            title="Fraud Rate by Hour of Day",
            labels={'x': 'Hour', 'y': 'Fraud Rate (%)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Feature correlation
        st.subheader("Feature Correlation with Fraud")
        
        pca_features = [col for col in df.columns if col.startswith('V')]
        correlation_with_fraud = df[pca_features + ['Class']].corr()['Class'].drop('Class').abs().sort_values(ascending=False)
        
        fig = px.bar(
            x=correlation_with_fraud.values[:15],
            y=correlation_with_fraud.index[:15],
            orientation='h',
            title="Top 15 Features Correlated with Fraud",
            labels={'x': 'Absolute Correlation', 'y': 'Features'}
        )
        st.plotly_chart(fig, use_container_width=True)

def create_model_performance(results):
    """Create model performance comparison."""
    st.header("🏆 Model Performance Comparison")
    
    # Prepare data
    model_data = []
    for name, metrics in results.items():
        model_data.append({
            'Model': name.replace('_', ' ').title(),
            'F1-Score': metrics['f1_score'],
            'ROC-AUC': metrics['roc_auc'],
            'Average Precision': metrics['avg_precision']
        })
    
    df_models = pd.DataFrame(model_data)
    
    # Performance metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig = px.bar(
            df_models.sort_values('F1-Score', ascending=True),
            x='F1-Score',
            y='Model',
            orientation='h',
            title="F1-Score Comparison",
            color='F1-Score',
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            df_models.sort_values('ROC-AUC', ascending=True),
            x='ROC-AUC',
            y='Model',
            orientation='h',
            title="ROC-AUC Comparison",
            color='ROC-AUC',
            color_continuous_scale='plasma'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        fig = px.bar(
            df_models.sort_values('Average Precision', ascending=True),
            x='Average Precision',
            y='Model',
            orientation='h',
            title="Average Precision Comparison",
            color='Average Precision',
            color_continuous_scale='cividis'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance table
    st.subheader("📊 Detailed Performance Metrics")
    st.dataframe(df_models.sort_values('F1-Score', ascending=False), use_container_width=True)

def create_prediction_interface(models, scaler, df):
    """Create interactive prediction interface."""
    st.header("🔮 Interactive Fraud Prediction")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Input Transaction Details")
        
        # Model selection
        model_names = list(models.keys())
        selected_model = st.selectbox("Select Model", model_names, index=0)
        
        # Transaction type
        transaction_type = st.radio("Transaction Type", ["Random Sample", "Custom Input"])
        
        if transaction_type == "Random Sample":
            sample_type = st.radio("Sample Type", ["Random", "Fraud Sample", "Normal Sample"])
            
            if st.button("Generate Sample"):
                if sample_type == "Random":
                    sample = df.sample(1)
                elif sample_type == "Fraud Sample":
                    fraud_samples = df[df['Class'] == 1]
                    if len(fraud_samples) > 0:
                        sample = fraud_samples.sample(1)
                    else:
                        st.error("No fraud samples found")
                        sample = df.sample(1)
                else:  # Normal Sample
                    normal_samples = df[df['Class'] == 0]
                    sample = normal_samples.sample(1)
                
                st.session_state['sample_data'] = sample
        
        else:  # Custom Input
            st.write("Enter transaction details:")
            amount = st.number_input("Amount ($)", min_value=0.0, value=100.0, step=0.01)
            hour = st.slider("Hour of Day", 0, 23, 12)
            
            # Simplified V features input
            st.write("PCA Features (V1-V5):")
            v1 = st.number_input("V1", value=0.0, step=0.1)
            v2 = st.number_input("V2", value=0.0, step=0.1)
            v3 = st.number_input("V3", value=0.0, step=0.1)
            v4 = st.number_input("V4", value=0.0, step=0.1)
            v5 = st.number_input("V5", value=0.0, step=0.1)
            
            # Create custom sample
            custom_data = {
                'Amount': amount,
                'Amount_log': np.log(amount + 1),
                'Hour': hour,
                'V1': v1, 'V2': v2, 'V3': v3, 'V4': v4, 'V5': v5
            }
            
            # Fill remaining V features with zeros for simplicity
            for i in range(6, 29):
                custom_data[f'V{i}'] = 0.0
            
            if st.button("Predict Custom Transaction"):
                custom_df = pd.DataFrame([custom_data])
                st.session_state['sample_data'] = custom_df
    
    with col2:
        st.subheader("Prediction Results")
        
        if 'sample_data' in st.session_state:
            sample = st.session_state['sample_data']
            
            # Prepare features
            feature_cols = [col for col in sample.columns if col not in ['Class', 'Time']]
            X_sample = sample[feature_cols]
            
            # Scale features
            X_sample_scaled = scaler.transform(X_sample)
            X_sample_scaled = pd.DataFrame(X_sample_scaled, columns=feature_cols)
            
            # Make prediction
            model = models[selected_model]
            
            try:
                if 'anomaly' in selected_model or 'isolation' in selected_model or 'svm' in selected_model:
                    prediction = model.predict(X_sample_scaled)[0]
                    prediction_binary = 1 if prediction == -1 else 0
                    if hasattr(model, 'decision_function'):
                        confidence = -model.decision_function(X_sample_scaled)[0]
                    else:
                        confidence = 0.5
                else:
                    prediction_binary = model.predict(X_sample_scaled)[0]
                    if hasattr(model, 'predict_proba'):
                        confidence = model.predict_proba(X_sample_scaled)[0][1]
                    else:
                        confidence = 0.5
                
                # Display prediction
                if prediction_binary == 1:
                    st.markdown("""
                    <div class="fraud-alert">
                    <h3>🚨 FRAUD DETECTED</h3>
                    <p>This transaction is predicted to be fraudulent!</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="normal-alert">
                    <h3>✅ NORMAL TRANSACTION</h3>
                    <p>This transaction appears to be legitimate.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Confidence score
                st.metric("Confidence Score", f"{confidence:.3f}")
                
                # Show transaction details
                st.subheader("Transaction Details")
                st.dataframe(sample, use_container_width=True)
                
                # Show actual label if available
                if 'Class' in sample.columns:
                    actual = sample['Class'].iloc[0]
                    st.write(f"**Actual Label:** {'Fraud' if actual == 1 else 'Normal'}")
                    
                    if actual == prediction_binary:
                        st.success("✅ Correct Prediction!")
                    else:
                        st.error("❌ Incorrect Prediction!")
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

def main():
    """Main dashboard function."""
    # Title
    st.markdown('<h1 class="main-header">💳 Credit Card Fraud Detection Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data and models
    df = load_data()
    models, scaler, results = load_models()
    
    if df is None or models is None:
        st.stop()
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["📊 Dataset Overview", "🔍 Exploratory Analysis", "🏆 Model Performance", "🔮 Live Prediction"]
    )
    
    # Page routing
    if page == "📊 Dataset Overview":
        create_dataset_overview(df)
        
        # Key insights
        st.header("💡 Key Insights")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info("**Highly Imbalanced Dataset**\n\nFraud transactions represent only 0.17% of all transactions, making this a challenging detection problem.")
        
        with col2:
            st.info("**PCA Features**\n\nV1-V28 are PCA-transformed features to protect user privacy while maintaining predictive power.")
        
        with col3:
            st.info("**Time & Amount**\n\nTransaction timing and amounts provide additional context for fraud detection patterns.")
    
    elif page == "🔍 Exploratory Analysis":
        create_eda_section(df)
    
    elif page == "🏆 Model Performance":
        create_model_performance(results)
        
        # Model descriptions
        st.header("🤖 Model Descriptions")
        
        with st.expander("🌳 Random Forest"):
            st.write("""
            **Best Performing Model (F1-Score: 0.841)**
            - Ensemble of decision trees
            - Handles class imbalance well with balanced class weights
            - Provides feature importance rankings
            - Robust to outliers and missing values
            """)
        
        with st.expander("🧠 Neural Network"):
            st.write("""
            **Strong Deep Learning Performance (F1-Score: 0.779)**
            - Multi-layer perceptron with hidden layers
            - Learns complex non-linear patterns
            - Good performance on imbalanced data
            - Early stopping prevents overfitting
            """)
        
        with st.expander("🔍 Anomaly Detection"):
            st.write("""
            **Unsupervised Approach**
            - Isolation Forest and One-Class SVM
            - Detects anomalies without labeled fraud examples
            - Useful for detecting new fraud patterns
            - Lower F1-scores but high precision for rare frauds
            """)
    
    elif page == "🔮 Live Prediction":
        create_prediction_interface(models, scaler, df)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
    <p>Built with ❤️ using Streamlit | Credit Card Fraud Detection System</p>
    <p>Dataset: <a href='https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud'>Kaggle Credit Card Fraud Detection</a></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()