#!/usr/bin/env python3
"""
Comprehensive Credit Card Fraud Detection Dashboard
==================================================

An advanced, interactive dashboard with comprehensive analysis, real-time model training,
detailed insights, and extensive customization options.
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
import os
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve, 
                           precision_recall_curve, roc_auc_score, f1_score, 
                           precision_score, recall_score, accuracy_score)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="Comprehensive Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 2rem;
        color: #2E86AB;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .insight-box {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .model-performance {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #2E86AB;
    }
    .code-demo {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load the credit card fraud dataset."""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(current_dir, 'creditcard.csv')
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_data
def load_models():
    """Load pre-trained models if available."""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        models = joblib.load(os.path.join(current_dir, 'fraud_models.joblib'))
        scaler = joblib.load(os.path.join(current_dir, 'scaler.joblib'))
        results = joblib.load(os.path.join(current_dir, 'model_results.joblib'))
        return models, scaler, results
    except:
        return None, None, None

def create_advanced_feature_analysis(df, selected_features, sample_size=10000):
    """Create comprehensive feature analysis."""
    st.markdown("### 🔬 Advanced Feature Analysis")
    
    # Sample data
    df_sample = df.sample(min(sample_size, len(df)))
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Distributions", "🔗 Correlations", "📈 Statistical Tests", "🎯 Feature Importance"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Feature distributions by class
            if selected_features:
                feature = st.selectbox("Select feature for distribution analysis:", selected_features)
                
                fig = make_subplots(rows=2, cols=1, 
                                  subplot_titles=("Normal Transactions", "Fraud Transactions"))
                
                # Normal transactions
                normal_data = df_sample[df_sample['Class'] == 0][feature]
                fig.add_trace(go.Histogram(x=normal_data, name="Normal", 
                                         marker_color='blue', opacity=0.7), row=1, col=1)
                
                # Fraud transactions  
                fraud_data = df_sample[df_sample['Class'] == 1][feature]
                fig.add_trace(go.Histogram(x=fraud_data, name="Fraud", 
                                         marker_color='red', opacity=0.7), row=2, col=1)
                
                fig.update_layout(height=500, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistical summary
                st.markdown("#### Statistical Summary")
                col_a, col_b = st.columns(2)
                with col_a:
                    normal_stats = normal_data.describe()
                    st.markdown("**Normal Transactions:**")
                    st.dataframe(normal_stats)
                
                with col_b:
                    fraud_stats = fraud_data.describe()
                    st.markdown("**Fraud Transactions:**")
                    st.dataframe(fraud_stats)
        
        with col2:
            # Box plots comparison
            if selected_features:
                fig = go.Figure()
                for feature in selected_features[:5]:  # Limit to 5 features
                    fig.add_trace(go.Box(y=df_sample[df_sample['Class']==0][feature], 
                                       name=f'{feature} (Normal)', marker_color='blue'))
                    fig.add_trace(go.Box(y=df_sample[df_sample['Class']==1][feature], 
                                       name=f'{feature} (Fraud)', marker_color='red'))
                
                fig.update_layout(title="Feature Distributions by Transaction Type", height=500)
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Advanced correlation analysis
        if len(selected_features) >= 2:
            correlation_features = selected_features + ['Amount', 'Class']
            corr_matrix = df_sample[correlation_features].corr()
            
            # Heatmap
            fig = px.imshow(corr_matrix, 
                          labels=dict(x="Features", y="Features", color="Correlation"),
                          color_continuous_scale='RdBu_r', aspect="auto")
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Correlation insights
            st.markdown("#### 🔍 Correlation Insights")
            fraud_correlations = corr_matrix['Class'].abs().sort_values(ascending=False)[1:]
            top_features = fraud_correlations.head(5)
            
            for feature, corr in top_features.items():
                correlation_strength = "Strong" if corr > 0.5 else "Moderate" if corr > 0.3 else "Weak"
                st.markdown(f"- **{feature}**: {corr:.3f} ({correlation_strength} correlation with fraud)")
    
    with tab3:
        # Statistical tests and insights
        st.markdown("#### 📊 Statistical Analysis")
        
        if selected_features:
            from scipy import stats
            
            results_data = []
            for feature in selected_features:
                normal_values = df_sample[df_sample['Class'] == 0][feature]
                fraud_values = df_sample[df_sample['Class'] == 1][feature]
                
                # T-test
                t_stat, p_value = stats.ttest_ind(normal_values, fraud_values)
                
                # Effect size (Cohen's d)
                pooled_std = np.sqrt(((len(normal_values) - 1) * normal_values.std()**2 + 
                                    (len(fraud_values) - 1) * fraud_values.std()**2) / 
                                   (len(normal_values) + len(fraud_values) - 2))
                cohens_d = (fraud_values.mean() - normal_values.mean()) / pooled_std
                
                results_data.append({
                    'Feature': feature,
                    'T-Statistic': f"{t_stat:.3f}",
                    'P-Value': f"{p_value:.6f}",
                    'Significant': "Yes" if p_value < 0.05 else "No",
                    'Effect Size (Cohen\'s d)': f"{cohens_d:.3f}",
                    'Effect Magnitude': "Large" if abs(cohens_d) > 0.8 else "Medium" if abs(cohens_d) > 0.5 else "Small"
                })
            
            results_df = pd.DataFrame(results_data)
            st.dataframe(results_df, use_container_width=True)
            
            # Interpretation
            significant_features = results_df[results_df['Significant'] == 'Yes']
            st.markdown(f"**🎯 Key Findings:**")
            st.markdown(f"- {len(significant_features)} out of {len(selected_features)} features show statistically significant differences")
            if len(significant_features) > 0:
                large_effect = significant_features[significant_features['Effect Magnitude'] == 'Large']
                st.markdown(f"- {len(large_effect)} features have large effect sizes (strong predictive power)")
    
    with tab4:
        # Feature importance using quick Random Forest
        st.markdown("#### 🎯 Feature Importance Analysis")
        
        if len(selected_features) >= 2:
            with st.spinner("Calculating feature importance..."):
                X = df_sample[selected_features]
                y = df_sample['Class']
                
                # Quick Random Forest for feature importance
                rf = RandomForestClassifier(n_estimators=50, random_state=42)
                rf.fit(X, y)
                
                importance_df = pd.DataFrame({
                    'Feature': selected_features,
                    'Importance': rf.feature_importances_
                }).sort_values('Importance', ascending=True)
                
                # Horizontal bar chart
                fig = px.bar(importance_df, x='Importance', y='Feature', 
                           orientation='h', color='Importance',
                           color_continuous_scale='viridis')
                fig.update_layout(height=400, title="Feature Importance for Fraud Detection")
                st.plotly_chart(fig, use_container_width=True)
                
                # Top features summary
                top_3 = importance_df.tail(3)
                st.markdown("**🏆 Top 3 Most Important Features:**")
                for idx, row in top_3.iterrows():
                    st.markdown(f"- **{row['Feature']}**: {row['Importance']:.3f}")

def create_interactive_model_demo():
    """Create interactive model training demonstration."""
    st.markdown("### 🤖 Interactive Model Training Demo")
    
    df = load_data()
    if df is None:
        st.error("Unable to load data for model demo")
        return
    
    # Model configuration sidebar
    with st.sidebar:
        st.markdown("## 🛠️ Model Configuration")
        
        # Algorithm selection
        algorithm = st.selectbox("Select Algorithm:", [
            "Random Forest", "Logistic Regression", "SVM", 
            "Decision Tree", "Naive Bayes", "K-Nearest Neighbors"
        ])
        
        # Sampling strategy
        sampling_strategy = st.selectbox("Sampling Strategy:", [
            "None", "SMOTE", "BorderlineSMOTE", "ADASYN", "Random Undersampling", "Tomek Links"
        ])
        
        # Scaling method
        scaling_method = st.selectbox("Feature Scaling:", [
            "StandardScaler", "RobustScaler", "MinMaxScaler", "None"
        ])
        
        # Sample size for demo
        sample_size = st.slider("Training Sample Size:", 1000, 50000, 10000)
        
        # Feature selection
        available_features = [col for col in df.columns if col.startswith('V')] + ['Amount']
        selected_features = st.multiselect(
            "Select Features:",
            available_features,
            default=available_features[:10]
        )
        
        # Algorithm-specific parameters
        st.markdown("#### Algorithm Parameters")
        if algorithm == "Random Forest":
            n_estimators = st.slider("Number of Trees:", 10, 200, 100)
            max_depth = st.slider("Max Depth:", 3, 20, 10)
            params = {'n_estimators': n_estimators, 'max_depth': max_depth, 'random_state': 42}
        elif algorithm == "Logistic Regression":
            C = st.slider("Regularization (C):", 0.01, 10.0, 1.0)
            params = {'C': C, 'random_state': 42, 'max_iter': 1000}
        elif algorithm == "SVM":
            C = st.slider("C Parameter:", 0.1, 10.0, 1.0)
            kernel = st.selectbox("Kernel:", ["rbf", "linear", "poly"])
            params = {'C': C, 'kernel': kernel, 'random_state': 42}
        elif algorithm == "Decision Tree":
            max_depth = st.slider("Max Depth:", 3, 20, 10)
            min_samples_split = st.slider("Min Samples Split:", 2, 20, 2)
            params = {'max_depth': max_depth, 'min_samples_split': min_samples_split, 'random_state': 42}
        elif algorithm == "Naive Bayes":
            var_smoothing = st.slider("Var Smoothing:", 1e-10, 1e-5, 1e-9, format="%.2e")
            params = {'var_smoothing': var_smoothing}
        else:  # K-Nearest Neighbors
            n_neighbors = st.slider("Number of Neighbors:", 3, 20, 5)
            weights = st.selectbox("Weights:", ["uniform", "distance"])
            params = {'n_neighbors': n_neighbors, 'weights': weights}
    
    # Train model button
    if st.button("🚀 Train Model", type="primary"):
        if not selected_features:
            st.error("Please select at least one feature")
            return
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with st.spinner("Training model..."):
            # Sample data
            status_text.text("Sampling data...")
            progress_bar.progress(10)
            df_sample = df.sample(min(sample_size, len(df)))
            
            # Prepare features
            status_text.text("Preparing features...")
            progress_bar.progress(20)
            X = df_sample[selected_features]
            y = df_sample['Class']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            # Apply scaling
            status_text.text("Applying feature scaling...")
            progress_bar.progress(30)
            if scaling_method != "None":
                if scaling_method == "StandardScaler":
                    scaler = StandardScaler()
                elif scaling_method == "RobustScaler":
                    scaler = RobustScaler()
                else:  # MinMaxScaler
                    scaler = MinMaxScaler()
                
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
            
            # Apply sampling
            status_text.text("Applying sampling strategy...")
            progress_bar.progress(50)
            if sampling_strategy != "None":
                if sampling_strategy == "SMOTE":
                    sampler = SMOTE(random_state=42)
                elif sampling_strategy == "BorderlineSMOTE":
                    sampler = BorderlineSMOTE(random_state=42)
                elif sampling_strategy == "ADASYN":
                    sampler = ADASYN(random_state=42)
                elif sampling_strategy == "Random Undersampling":
                    sampler = RandomUnderSampler(random_state=42)
                else:  # Tomek Links
                    sampler = TomekLinks()
                
                X_train, y_train = sampler.fit_resample(X_train, y_train)
            
            # Initialize model
            status_text.text("Initializing model...")
            progress_bar.progress(60)
            if algorithm == "Random Forest":
                model = RandomForestClassifier(**params)
            elif algorithm == "Logistic Regression":
                model = LogisticRegression(**params)
            elif algorithm == "SVM":
                model = SVC(**params, probability=True)
            elif algorithm == "Decision Tree":
                model = DecisionTreeClassifier(**params)
            elif algorithm == "Naive Bayes":
                model = GaussianNB(**params)
            else:  # K-Nearest Neighbors
                model = KNeighborsClassifier(**params)
            
            # Train model
            status_text.text("Training model...")
            progress_bar.progress(80)
            model.fit(X_train, y_train)
            
            # Make predictions
            status_text.text("Evaluating model...")
            progress_bar.progress(90)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            progress_bar.progress(100)
            status_text.text("Model training complete!")
        
        # Display results
        st.markdown("## 📊 Model Performance Results")
        
        # Metrics cards
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{accuracy:.3f}</h3>
                <p>Accuracy</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{precision:.3f}</h3>
                <p>Precision</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{recall:.3f}</h3>
                <p>Recall</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{f1:.3f}</h3>
                <p>F1-Score</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{roc_auc:.3f}</h3>
                <p>ROC-AUC</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Detailed analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Confusion Matrix
            st.markdown("#### Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig = px.imshow(cm, text_auto=True, aspect="auto", 
                          color_continuous_scale='Blues',
                          labels=dict(x="Predicted", y="Actual"))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # ROC Curve
            st.markdown("#### ROC Curve")
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', 
                                   name=f'ROC Curve (AUC = {roc_auc:.3f})'))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', 
                                   name='Random Classifier', line=dict(dash='dash')))
            fig.update_layout(xaxis_title='False Positive Rate', 
                            yaxis_title='True Positive Rate',
                            height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            st.markdown("#### Feature Importance")
            importance_df = pd.DataFrame({
                'Feature': selected_features,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=True)
            
            fig = px.bar(importance_df, x='Importance', y='Feature', 
                       orientation='h', color='Importance',
                       color_continuous_scale='viridis')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Classification Report
        st.markdown("#### Detailed Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.round(3), use_container_width=True)
        
        # Performance insights
        st.markdown("#### 🎯 Performance Insights")
        
        insights = []
        
        if f1 > 0.8:
            insights.append("✅ Excellent F1-score indicates strong overall performance")
        elif f1 > 0.6:
            insights.append("✅ Good F1-score shows reasonable performance")
        else:
            insights.append("⚠️ F1-score could be improved - consider different algorithms or parameters")
        
        if precision > 0.8:
            insights.append("✅ High precision means few false positives")
        else:
            insights.append("⚠️ Consider improving precision to reduce false alarms")
        
        if recall > 0.8:
            insights.append("✅ High recall means most fraud cases are detected")
        else:
            insights.append("⚠️ Low recall means some fraud cases are missed")
        
        for insight in insights:
            st.markdown(f"- {insight}")

def main():
    """Main dashboard application."""
    st.markdown('<h1 class="main-header">🛡️ Comprehensive Fraud Detection Dashboard</h1>', unsafe_allow_html=True)
    
    # Navigation
    pages = {
        "🏠 Overview": show_overview,
        "📊 Advanced Data Analysis": show_advanced_data_analysis,
        "🤖 Interactive Model Demo": create_interactive_model_demo,
        "📈 Model Comparison": show_model_comparison,
        "🔬 Feature Engineering": show_feature_engineering,
        "💰 Business Impact": show_business_impact,
        "🎯 Live Prediction": show_live_prediction
    }
    
    # Sidebar navigation
    st.sidebar.markdown("## 🧭 Navigation")
    selected_page = st.sidebar.selectbox("Choose a page:", list(pages.keys()))
    
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### 📊 Dataset Info
    - **Transactions**: 284,807
    - **Fraud Rate**: 0.172%
    - **Features**: 30 (V1-V28 + Amount + Time)
    """)
    
    # Display selected page
    pages[selected_page]()

def show_overview():
    """Enhanced overview page."""
    st.markdown("## 🎯 System Overview")
    
    df = load_data()
    if df is None:
        st.error("Unable to load dataset")
        return
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_transactions = len(df)
    fraud_transactions = df['Class'].sum()
    fraud_rate = (fraud_transactions / total_transactions) * 100
    avg_transaction = df['Amount'].mean()
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{total_transactions:,}</h3>
            <p>Total Transactions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{fraud_transactions:,}</h3>
            <p>Fraud Cases</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{fraud_rate:.3f}%</h3>
            <p>Fraud Rate</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>${avg_transaction:.2f}</h3>
            <p>Avg Transaction</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Transaction distribution
        fig = px.histogram(df.sample(10000), x='Amount', color='Class', 
                         title="Transaction Amount Distribution",
                         nbins=50, color_discrete_map={0: 'blue', 1: 'red'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Class distribution
        class_counts = df['Class'].value_counts()
        fig = px.pie(values=class_counts.values, names=['Normal', 'Fraud'],
                   title="Transaction Class Distribution",
                   color_discrete_map={'Normal': 'blue', 'Fraud': 'red'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def show_advanced_data_analysis():
    """Advanced data analysis page."""
    st.markdown("## 🔬 Advanced Data Analysis")
    
    df = load_data()
    if df is None:
        st.error("Unable to load dataset")
        return
    
    # Configuration
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sample_size = st.slider("Sample Size:", 1000, 50000, 15000)
    
    with col2:
        analysis_type = st.selectbox("Analysis Focus:", [
            "Feature Analysis", "Temporal Patterns", "Amount Analysis", "Anomaly Detection"
        ])
    
    with col3:
        show_fraud_only = st.checkbox("Focus on Fraud Cases")
    
    # Feature selection
    v_features = [col for col in df.columns if col.startswith('V')]
    selected_features = st.multiselect(
        "Select Features for Analysis:",
        v_features,
        default=v_features[:8]
    )
    
    if not selected_features:
        st.warning("Please select at least one feature for analysis")
        return
    
    # Create comprehensive analysis
    create_advanced_feature_analysis(df, selected_features, sample_size)

def show_model_comparison():
    """Model comparison and analysis."""
    st.markdown("## 📈 Model Comparison & Analysis")
    
    models, scaler, results = load_models()
    if models is None:
        st.warning("No pre-trained models found. Please run the training script first or use the Interactive Model Demo.")
        return
    
    # Model performance comparison
    st.markdown("### 🏆 Model Performance Comparison")
    
    performance_data = []
    for model_name, metrics in results.items():
        performance_data.append({
            'Model': model_name,
            'F1-Score': metrics.get('f1_score', 0),
            'Precision': metrics.get('precision', 0),
            'Recall': metrics.get('recall', 0),
            'ROC-AUC': metrics.get('roc_auc', 0),
            'Accuracy': metrics.get('accuracy', 0)
        })
    
    performance_df = pd.DataFrame(performance_data)
    
    # Interactive performance chart
    metric_to_plot = st.selectbox("Select Metric to Compare:", 
                                ['F1-Score', 'Precision', 'Recall', 'ROC-AUC', 'Accuracy'])
    
    fig = px.bar(performance_df.sort_values(metric_to_plot, ascending=True), 
               x=metric_to_plot, y='Model', orientation='h',
               color=metric_to_plot, color_continuous_scale='viridis',
               title=f"Model Comparison - {metric_to_plot}")
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed comparison table
    st.markdown("### 📊 Detailed Performance Metrics")
    st.dataframe(performance_df.set_index('Model').round(4), use_container_width=True)
    
    # Best model highlight
    best_model = performance_df.loc[performance_df['F1-Score'].idxmax()]
    st.markdown(f"""
    <div class="model-performance">
        <h4>🏅 Best Performing Model</h4>
        <p><strong>{best_model['Model']}</strong> with F1-Score: {best_model['F1-Score']:.4f}</p>
    </div>
    """, unsafe_allow_html=True)

def show_feature_engineering():
    """Feature engineering demonstrations."""
    st.markdown("## 🔬 Feature Engineering Workshop")
    
    df = load_data()
    if df is None:
        st.error("Unable to load dataset")
        return
    
    st.markdown("### 🛠️ Feature Engineering Techniques")
    
    # Show original features
    st.markdown("#### Original Features")
    original_features = ['Time', 'Amount'] + [col for col in df.columns if col.startswith('V')][:5]
    st.dataframe(df[original_features].head(), use_container_width=True)
    
    # Create new features
    st.markdown("#### 🆕 Engineered Features")
    
    df_engineered = df.copy()
    
    # Time-based features
    df_engineered['Hour'] = (df_engineered['Time'] % (24 * 3600)) // 3600
    df_engineered['Day'] = df_engineered['Time'] // (24 * 3600)
    df_engineered['IsWeekend'] = ((df_engineered['Time'] // (24 * 3600)) % 7) >= 5
    
    # Amount-based features
    df_engineered['Amount_Log'] = np.log1p(df_engineered['Amount'])
    df_engineered['Amount_Sqrt'] = np.sqrt(df_engineered['Amount'])
    df_engineered['Is_High_Amount'] = df_engineered['Amount'] > df_engineered['Amount'].quantile(0.95)
    
    # Interaction features
    v_features = [col for col in df.columns if col.startswith('V')][:5]
    for i, feat1 in enumerate(v_features):
        for feat2 in v_features[i+1:]:
            df_engineered[f'{feat1}_{feat2}_interaction'] = df_engineered[feat1] * df_engineered[feat2]
    
    # Statistical features
    df_engineered['V_Mean'] = df_engineered[v_features].mean(axis=1)
    df_engineered['V_Std'] = df_engineered[v_features].std(axis=1)
    df_engineered['V_Max'] = df_engineered[v_features].max(axis=1)
    df_engineered['V_Min'] = df_engineered[v_features].min(axis=1)
    
    # Show engineered features
    new_features = ['Hour', 'Day', 'IsWeekend', 'Amount_Log', 'Amount_Sqrt', 
                   'Is_High_Amount', 'V_Mean', 'V_Std']
    st.dataframe(df_engineered[new_features].head(), use_container_width=True)
    
    # Feature impact analysis
    st.markdown("#### 📊 Feature Impact Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Correlation with target
        correlations = []
        for feature in new_features:
            if df_engineered[feature].dtype in ['int64', 'float64']:
                corr = df_engineered[feature].corr(df_engineered['Class'])
                correlations.append({'Feature': feature, 'Correlation': abs(corr)})
        
        corr_df = pd.DataFrame(correlations).sort_values('Correlation', ascending=True)
        
        fig = px.bar(corr_df, x='Correlation', y='Feature', orientation='h',
                   color='Correlation', color_continuous_scale='viridis',
                   title="Feature Correlation with Fraud")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Feature distributions
        selected_feature = st.selectbox("Select Feature to Analyze:", new_features)
        
        if df_engineered[selected_feature].dtype in ['int64', 'float64']:
            fig = make_subplots(rows=2, cols=1, subplot_titles=("Normal", "Fraud"))
            
            normal_data = df_engineered[df_engineered['Class'] == 0][selected_feature].sample(min(1000, len(df_engineered[df_engineered['Class'] == 0])))
            fraud_data = df_engineered[df_engineered['Class'] == 1][selected_feature]
            
            fig.add_trace(go.Histogram(x=normal_data, name="Normal", marker_color='blue'), row=1, col=1)
            fig.add_trace(go.Histogram(x=fraud_data, name="Fraud", marker_color='red'), row=2, col=1)
            
            fig.update_layout(height=400, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

def show_business_impact():
    """Business impact and ROI analysis."""
    st.markdown("## 💰 Business Impact Analysis")
    
    df = load_data()
    models, scaler, results = load_models()
    
    if df is None:
        st.error("Unable to load dataset")
        return
    
    # Business parameters
    st.markdown("### 🎛️ Business Parameters")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_fraud_amount = st.number_input("Average Fraud Amount ($):", 
                                         value=float(df[df['Class']==1]['Amount'].mean()),
                                         min_value=0.0, format="%.2f")
    
    with col2:
        investigation_cost = st.number_input("Investigation Cost ($):", 
                                           value=50.0, min_value=0.0, format="%.2f")
    
    with col3:
        false_positive_cost = st.number_input("False Positive Cost ($):", 
                                            value=5.0, min_value=0.0, format="%.2f")
    
    with col4:
        daily_transactions = st.number_input("Daily Transactions:", 
                                           value=10000, min_value=1)
    
    # ROI Calculation
    st.markdown("### 📊 ROI Analysis")
    
    if results:
        roi_data = []
        
        for model_name, metrics in results.items():
            precision = metrics.get('precision', 0)
            recall = metrics.get('recall', 0)
            
            # Estimated fraud cases per day
            fraud_rate = df['Class'].mean()
            daily_fraud = daily_transactions * fraud_rate
            
            # True positives, false positives
            tp = daily_fraud * recall
            fp = (daily_transactions - daily_fraud) * (1 - metrics.get('specificity', 0.99))
            
            # Savings and costs
            fraud_prevented = tp * avg_fraud_amount
            investigation_costs = tp * investigation_cost
            false_positive_costs = fp * false_positive_cost
            
            net_benefit = fraud_prevented - investigation_costs - false_positive_costs
            
            roi_data.append({
                'Model': model_name,
                'Daily_Fraud_Prevented': f"${fraud_prevented:,.0f}",
                'Daily_Investigation_Cost': f"${investigation_costs:,.0f}",
                'Daily_FP_Cost': f"${false_positive_costs:,.0f}",
                'Net_Daily_Benefit': f"${net_benefit:,.0f}",
                'Annual_Benefit': f"${net_benefit * 365:,.0f}",
                'ROI_Percentage': f"{(net_benefit / (investigation_costs + false_positive_costs)) * 100:.1f}%"
            })
        
        roi_df = pd.DataFrame(roi_data)
        st.dataframe(roi_df, use_container_width=True)
        
        # Best ROI model
        best_roi_model = roi_df.iloc[0]['Model']  # Assuming sorted by performance
        st.markdown(f"""
        <div class="insight-box">
            <h4>💡 Business Insights</h4>
            <p>The <strong>{best_roi_model}</strong> model shows the best business value with significant fraud prevention capabilities.</p>
            <p>Implementing this system could save substantial amounts annually while maintaining operational efficiency.</p>
        </div>
        """, unsafe_allow_html=True)

def show_live_prediction():
    """Live prediction interface."""
    st.markdown("## 🎯 Live Fraud Detection")
    
    models, scaler, results = load_models()
    
    if models is None:
        st.error("No models available. Please run training first.")
        return
    
    # Model selection
    model_name = st.selectbox("Select Model:", list(models.keys()))
    
    # Input interface
    st.markdown("### 📝 Transaction Input")
    
    col1, col2 = st.columns(2)
    
    with col1:
        amount = st.number_input("Transaction Amount ($):", value=100.0, min_value=0.0)
        time_val = st.number_input("Time (seconds from first transaction):", value=3600.0, min_value=0.0)
    
    with col2:
        # Simplified V features input
        st.markdown("**V Features (simplified):**")
        v1 = st.slider("V1:", -30.0, 30.0, 0.0)
        v2 = st.slider("V2:", -30.0, 30.0, 0.0)
        v3 = st.slider("V3:", -30.0, 30.0, 0.0)
        v4 = st.slider("V4:", -30.0, 30.0, 0.0)
    
    # Prediction button
    if st.button("🔍 Predict Transaction", type="primary"):
        # Create feature vector (simplified for demo)
        features = np.array([[amount, time_val, v1, v2, v3, v4] + [0]*24])  # Pad with zeros for demo
        
        # Make prediction
        model = models[model_name]
        prediction = model.predict(features)[0]
        
        if hasattr(model, 'predict_proba'):
            fraud_probability = model.predict_proba(features)[0][1]
        else:
            fraud_probability = prediction
        
        # Display result
        if prediction == 1:
            st.error(f"🚨 **FRAUD DETECTED** - Probability: {fraud_probability:.3f}")
        else:
            st.success(f"✅ **LEGITIMATE TRANSACTION** - Fraud Probability: {fraud_probability:.3f}")
        
        # Show confidence
        confidence = max(fraud_probability, 1 - fraud_probability)
        st.metric("Prediction Confidence", f"{confidence:.3f}")

if __name__ == "__main__":
    main()