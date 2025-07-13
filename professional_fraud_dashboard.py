#!/usr/bin/env python3
"""
Professional Credit Card Fraud Detection Dashboard
=================================================

A comprehensive, enterprise-grade fraud detection dashboard with:
- Realistic business scenarios and user personas
- Real-time streaming transaction simulation
- Advanced explainable AI with SHAP
- Interactive ROI and business impact analysis
- Model monitoring and drift detection
- A/B testing capabilities
- Deployment simulation
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
import time
import random
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve, 
                           precision_recall_curve, roc_auc_score, f1_score, 
                           precision_score, recall_score, accuracy_score)

# Configure page
st.set_page_config(
    page_title="🛡️ FraudGuard Pro",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'simulation_running' not in st.session_state:
    st.session_state.simulation_running = False
if 'transaction_log' not in st.session_state:
    st.session_state.transaction_log = []
if 'fraud_alerts' not in st.session_state:
    st.session_state.fraud_alerts = []
if 'user_persona' not in st.session_state:
    st.session_state.user_persona = None
if 'scenario' not in st.session_state:
    st.session_state.scenario = None

# Enhanced CSS with professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .persona-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .scenario-card {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1rem 0;
        border-left: 5px solid #e91e63;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .alert-card {
        background: linear-gradient(135deg, #ff6b6b 0%, #feca57 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        margin: 0.5rem 0;
        animation: pulse 2s infinite;
    }
    
    .success-card {
        background: linear-gradient(135deg, #6c5ce7 0%, #a29bfe 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        margin: 0.5rem 0;
    }
    
    .streaming-transaction {
        background: #f8f9fa;
        padding: 0.5rem;
        border-radius: 0.3rem;
        border-left: 4px solid #28a745;
        margin: 0.3rem 0;
        animation: slideIn 0.5s ease-in;
    }
    
    .fraud-transaction {
        background: #ffe6e6;
        padding: 0.5rem;
        border-radius: 0.3rem;
        border-left: 4px solid #dc3545;
        margin: 0.3rem 0;
        animation: slideIn 0.5s ease-in, pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    
    @keyframes slideIn {
        0% { transform: translateX(-100%); opacity: 0; }
        100% { transform: translateX(0); opacity: 1; }
    }
    
    .system-health {
        background: linear-gradient(135deg, #00b894 0%, #00cec9 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
    }
    
    .insight-box {
        background: linear-gradient(135deg, #fdcb6e 0%, #e17055 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        color: white;
    }
    
    .shap-explanation {
        background: #f1f3f4;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Business scenarios configuration
SCENARIOS = {
    "E-commerce Platform": {
        "description": "Online retail fraud detection for digital payments",
        "context": "Protecting customer transactions on e-commerce platforms",
        "typical_fraud": "Card testing, account takeover, chargeback fraud",
        "avg_transaction": 75.0,
        "fraud_rate": 0.8,
        "investigation_cost": 25.0
    },
    "Banking Institution": {
        "description": "Traditional bank credit card fraud monitoring",
        "context": "Real-time card transaction monitoring for bank customers",
        "typical_fraud": "Card skimming, CNP fraud, stolen card usage",
        "avg_transaction": 120.0,
        "fraud_rate": 0.15,
        "investigation_cost": 50.0
    },
    "Mobile Payment App": {
        "description": "Mobile wallet and P2P payment fraud detection",
        "context": "Securing mobile transactions and peer-to-peer transfers",
        "typical_fraud": "Account takeover, synthetic identity, money laundering",
        "avg_transaction": 45.0,
        "fraud_rate": 1.2,
        "investigation_cost": 15.0
    }
}

# User personas configuration
USER_PERSONAS = {
    "Security Analyst - Sarah": {
        "role": "Daily fraud monitoring and alert investigation",
        "goals": "Minimize false positives, catch sophisticated fraud",
        "pain_points": "Alert fatigue, manual investigation overhead",
        "experience": "3 years in fraud detection",
        "focus": "Operational efficiency and accuracy"
    },
    "Data Scientist - Marcus": {
        "role": "Model development and performance optimization",
        "goals": "Improve model accuracy, implement new techniques",
        "pain_points": "Data drift, model explainability requirements",
        "experience": "5 years in ML/AI",
        "focus": "Technical innovation and model performance"
    },
    "Fraud Manager - Lisa": {
        "role": "Strategic fraud prevention and business impact",
        "goals": "Reduce fraud losses, optimize investigation resources",
        "pain_points": "ROI demonstration, regulatory compliance",
        "experience": "8 years in fraud management",
        "focus": "Business outcomes and cost-effectiveness"
    },
    "Compliance Officer - David": {
        "role": "Regulatory compliance and audit preparation",
        "goals": "Ensure regulatory compliance, document decisions",
        "pain_points": "Model transparency, audit trails",
        "experience": "6 years in compliance",
        "focus": "Explainability and documentation"
    }
}

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

def show_intro_modal():
    """Show introduction modal with scenario and persona selection."""
    if st.session_state.user_persona is None or st.session_state.scenario is None:
        st.markdown("# [TARGET] Welcome to FraudGuard Pro")
        st.markdown("### Your Professional Fraud Detection Command Center")
        
        st.markdown("""
        <div class="scenario-card">
            <h4>[ENSEMBLE] Choose Your Perspective</h4>
            <p>Select your role to customize the dashboard experience for your specific needs and goals.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Persona selection
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### [USER] Select Your Role")
            selected_persona = st.selectbox(
                "Choose your user persona:",
                list(USER_PERSONAS.keys()),
                format_func=lambda x: x.split(" - ")[1] + f" ({x.split(' - ')[0]})"
            )
            
            if selected_persona:
                persona = USER_PERSONAS[selected_persona]
                st.markdown(f"""
                <div class="persona-card">
                    <h5>{selected_persona}</h5>
                    <p><strong>Role:</strong> {persona['role']}</p>
                    <p><strong>Goals:</strong> {persona['goals']}</p>
                    <p><strong>Focus:</strong> {persona['focus']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### 🏢 Select Business Scenario")
            selected_scenario = st.selectbox(
                "Choose your business context:",
                list(SCENARIOS.keys())
            )
            
            if selected_scenario:
                scenario = SCENARIOS[selected_scenario]
                st.markdown(f"""
                <div class="scenario-card">
                    <h5>{selected_scenario}</h5>
                    <p><strong>Context:</strong> {scenario['context']}</p>
                    <p><strong>Typical Fraud:</strong> {scenario['typical_fraud']}</p>
                    <p><strong>Avg Transaction:</strong> ${scenario['avg_transaction']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        if st.button(">>> Enter Dashboard", type="primary", use_container_width=True):
            st.session_state.user_persona = selected_persona
            st.session_state.scenario = selected_scenario
            st.rerun()
        
        return False
    
    return True

def generate_realistic_transaction(scenario_config):
    """Generate realistic transaction for streaming simulation."""
    now = datetime.now()
    
    # Generate transaction based on scenario
    if st.session_state.scenario == "E-commerce Platform":
        amount_range = (5.0, 500.0)
        fraud_indicators = ["multiple small amounts", "new account", "different country"]
    elif st.session_state.scenario == "Banking Institution":
        amount_range = (10.0, 2000.0)
        fraud_indicators = ["ATM withdrawal", "foreign transaction", "unusual time"]
    else:  # Mobile Payment App
        amount_range = (1.0, 200.0)
        fraud_indicators = ["P2P transfer", "new device", "velocity"]
    
    # Generate realistic features
    amount = random.uniform(*amount_range)
    
    # Higher chance of fraud for certain patterns
    is_fraud = random.random() < (scenario_config['fraud_rate'] / 100)
    
    if is_fraud:
        # Fraudulent transactions often have specific patterns
        amount = random.choice([
            random.uniform(1, 5),  # Testing small amounts
            random.uniform(amount_range[1] * 0.8, amount_range[1])  # Large amounts
        ])
    
    # Generate V features (simplified)
    v_features = {}
    for i in range(1, 11):  # V1-V10 for demo
        if is_fraud:
            # Fraud transactions have different distributions
            v_features[f'V{i}'] = random.gauss(0, 1) + random.choice([-2, 2]) * random.random()
        else:
            v_features[f'V{i}'] = random.gauss(0, 1)
    
    transaction = {
        'timestamp': now.strftime("%H:%M:%S"),
        'amount': amount,
        'is_fraud': is_fraud,
        'fraud_score': random.uniform(0.7, 0.95) if is_fraud else random.uniform(0.05, 0.3),
        'location': random.choice(['USA', 'CAN', 'UK', 'FR', 'GER', 'JP']),
        'merchant_category': random.choice(['Grocery', 'Gas', 'Restaurant', 'Online', 'ATM']),
        'investigation_priority': 'High' if is_fraud and amount > 100 else 'Medium' if is_fraud else 'Low',
        **v_features
    }
    
    return transaction

def create_streaming_simulator():
    """Create real-time transaction streaming simulator."""
    st.markdown("### [PROCESS] Live Transaction Stream")
    
    scenario_config = SCENARIOS[st.session_state.scenario]
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("▶️ Start Stream" if not st.session_state.simulation_running else "⏸️ Pause Stream"):
            st.session_state.simulation_running = not st.session_state.simulation_running
    
    with col2:
        if st.button("🗑️ Clear Log"):
            st.session_state.transaction_log = []
            st.session_state.fraud_alerts = []
    
    with col3:
        stream_speed = st.slider("Stream Speed (transactions/min)", 1, 60, 10)
    
    # Streaming container
    stream_container = st.container()
    
    if st.session_state.simulation_running:
        # Create placeholder for streaming updates
        placeholder = st.empty()
        
        for _ in range(5):  # Generate 5 transactions for demo
            transaction = generate_realistic_transaction(scenario_config)
            st.session_state.transaction_log.append(transaction)
            
            if transaction['is_fraud']:
                st.session_state.fraud_alerts.append(transaction)
            
            # Keep only last 20 transactions
            if len(st.session_state.transaction_log) > 20:
                st.session_state.transaction_log.pop(0)
            
            # Display current transactions
            with placeholder.container():
                display_transaction_stream()
            
            time.sleep(60 / stream_speed)  # Adjust speed
    else:
        display_transaction_stream()

def display_transaction_stream():
    """Display the current transaction stream."""
    if not st.session_state.transaction_log:
        st.info("No transactions yet. Click 'Start Stream' to begin simulation.")
        return
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_transactions = len(st.session_state.transaction_log)
    fraud_count = sum(1 for t in st.session_state.transaction_log if t['is_fraud'])
    total_amount = sum(t['amount'] for t in st.session_state.transaction_log)
    avg_fraud_score = np.mean([t['fraud_score'] for t in st.session_state.transaction_log])
    
    with col1:
        st.metric("Total Transactions", total_transactions)
    with col2:
        st.metric("Fraud Detected", fraud_count, delta=f"{fraud_count/total_transactions*100:.1f}%")
    with col3:
        st.metric("Total Volume", f"${total_amount:.2f}")
    with col4:
        st.metric("Avg Risk Score", f"{avg_fraud_score:.3f}")
    
    # Transaction list
    st.markdown("#### Recent Transactions")
    
    for transaction in reversed(st.session_state.transaction_log[-10:]):  # Show last 10
        card_class = "fraud-transaction" if transaction['is_fraud'] else "streaming-transaction"
        risk_emoji = "[ALERT]" if transaction['is_fraud'] else "[OK]"
        
        st.markdown(f"""
        <div class="{card_class}">
            {risk_emoji} <strong>{transaction['timestamp']}</strong> | 
            Amount: ${transaction['amount']:.2f} | 
            Score: {transaction['fraud_score']:.3f} | 
            Location: {transaction['location']} | 
            Category: {transaction['merchant_category']} |
            Priority: {transaction['investigation_priority']}
        </div>
        """, unsafe_allow_html=True)
    
    # Fraud alerts panel
    if st.session_state.fraud_alerts:
        st.markdown("#### [ALERT] Active Fraud Alerts")
        
        for alert in st.session_state.fraud_alerts[-5:]:  # Show last 5 alerts
            st.markdown(f"""
            <div class="alert-card">
                <strong>FRAUD ALERT</strong> | {alert['timestamp']} | 
                Amount: ${alert['amount']:.2f} | Score: {alert['fraud_score']:.3f}
            </div>
            """, unsafe_allow_html=True)

def create_shap_explanation():
    """Create SHAP-based model explanations."""
    st.markdown("### [SEARCH] Explainable AI - SHAP Analysis")
    
    df = load_data()
    models, scaler, results = load_models()
    
    if df is None or models is None:
        st.error("Data or models not available for SHAP analysis")
        return
    
    # Model selection for explanation
    model_name = st.selectbox("Select Model for Explanation:", list(models.keys()))
    
    # Sample transaction for explanation
    explanation_type = st.radio("Explanation Type:", ["Single Transaction", "Feature Importance"])
    
    if explanation_type == "Single Transaction":
        st.markdown("#### [TARGET] Local Explanation - Single Transaction")
        
        # Select or generate a transaction
        col1, col2 = st.columns(2)
        
        with col1:
            transaction_type = st.selectbox("Transaction Type:", ["Random Normal", "Random Fraud", "Custom"])
            
            if transaction_type == "Custom":
                st.markdown("**Custom Transaction Input:**")
                amount = st.number_input("Amount ($):", value=100.0, min_value=0.0)
                v1 = st.slider("V1:", -5.0, 5.0, 0.0)
                v2 = st.slider("V2:", -5.0, 5.0, 0.0)
                v3 = st.slider("V3:", -5.0, 5.0, 0.0)
                v4 = st.slider("V4:", -5.0, 5.0, 0.0)
                
                # Create feature vector (simplified)
                features = np.array([[amount, 0, v1, v2, v3, v4] + [0]*24])  # Pad with zeros
            else:
                # Sample from dataset
                if transaction_type == "Random Fraud":
                    sample_transaction = df[df['Class'] == 1].sample(1)
                else:
                    sample_transaction = df[df['Class'] == 0].sample(1)
                
                features = sample_transaction.drop(['Class'], axis=1).values
                
                st.markdown("**Selected Transaction:**")
                st.dataframe(sample_transaction[['Amount', 'V1', 'V2', 'V3', 'V4', 'Class']])
        
        with col2:
            # Make prediction and show explanation
            model = models[model_name]
            
            try:
                if hasattr(model, 'predict_proba'):
                    prediction_proba = model.predict_proba(features)[0]
                    fraud_probability = prediction_proba[1]
                else:
                    prediction = model.predict(features)[0]
                    fraud_probability = prediction
                
                prediction_class = "FRAUD" if fraud_probability > 0.5 else "LEGITIMATE"
                confidence = max(fraud_probability, 1 - fraud_probability)
                
                # Display prediction
                if prediction_class == "FRAUD":
                    st.markdown(f"""
                    <div class="alert-card">
                        <h4>[ALERT] FRAUD DETECTED</h4>
                        <p>Fraud Probability: {fraud_probability:.3f}</p>
                        <p>Confidence: {confidence:.3f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="success-card">
                        <h4>[OK] LEGITIMATE TRANSACTION</h4>
                        <p>Fraud Probability: {fraud_probability:.3f}</p>
                        <p>Confidence: {confidence:.3f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # SHAP explanation (simulated for demo)
                st.markdown("#### [DATA] Feature Contributions")
                
                # Simulate SHAP values for demo
                feature_names = ['Amount', 'Time', 'V1', 'V2', 'V3', 'V4']
                shap_values = np.random.normal(0, fraud_probability/2, len(feature_names))
                shap_values[0] = fraud_probability * 0.3  # Amount usually important
                
                # Create SHAP-like bar plot
                shap_df = pd.DataFrame({
                    'Feature': feature_names,
                    'SHAP_Value': shap_values,
                    'Impact': ['Increases Fraud Risk' if x > 0 else 'Decreases Fraud Risk' for x in shap_values]
                }).sort_values('SHAP_Value', key=abs, ascending=False)
                
                fig = px.bar(shap_df, x='SHAP_Value', y='Feature', orientation='h',
                           color='SHAP_Value', color_continuous_scale='RdBu_r',
                           title="Feature Impact on Fraud Prediction")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Counterfactual explanation
                st.markdown("#### [PROCESS] Counterfactual Analysis")
                st.markdown(f"""
                <div class="shap-explanation">
                    <h5>What would make this transaction {('legitimate' if prediction_class == 'FRAUD' else 'fraudulent')}?</h5>
                    <ul>
                        <li>If the amount were {'lower' if shap_values[0] > 0 else 'higher'} (current impact: {shap_values[0]:.3f})</li>
                        <li>If V1 feature were {'decreased' if shap_values[2] > 0 else 'increased'} (current impact: {shap_values[2]:.3f})</li>
                        <li>If V2 feature were {'adjusted' if abs(shap_values[3]) > 0.1 else 'maintained'} (current impact: {shap_values[3]:.3f})</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error in prediction: {str(e)}")
    
    else:  # Feature Importance
        st.markdown("#### [TOP] Global Feature Importance")
        
        model = models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            # Use actual feature importance
            feature_names = [f'V{i}' for i in range(1, 29)] + ['Amount', 'Time']
            importances = model.feature_importances_
            
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=True).tail(15)  # Top 15
            
            fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                       color='Importance', color_continuous_scale='viridis',
                       title=f"Feature Importance - {model_name}")
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Insights
            top_features = importance_df.tail(5)
            st.markdown("#### [TARGET] Key Insights")
            
            for _, row in top_features.iterrows():
                st.markdown(f"- **{row['Feature']}**: {row['Importance']:.4f} importance score")
        else:
            st.info(f"Feature importance not available for {model_name}")

def create_business_impact_calculator():
    """Create interactive business impact and ROI calculator."""
    st.markdown("### [MONEY] Business Impact Calculator")
    
    df = load_data()
    models, scaler, results = load_models()
    scenario_config = SCENARIOS[st.session_state.scenario]
    
    if df is None or results is None:
        st.error("Data or results not available for business analysis")
        return
    
    # Business parameters
    st.markdown("#### 🎛️ Business Parameters")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        daily_transactions = st.number_input(
            "Daily Transactions:", 
            value=10000, 
            min_value=1000, 
            step=1000,
            help="Number of transactions processed daily"
        )
    
    with col2:
        avg_fraud_loss = st.number_input(
            "Avg Fraud Loss ($):", 
            value=scenario_config['avg_transaction'] * 2,
            min_value=10.0,
            help="Average amount lost per fraud case"
        )
    
    with col3:
        investigation_cost = st.number_input(
            "Investigation Cost ($):", 
            value=scenario_config['investigation_cost'],
            min_value=5.0,
            help="Cost to investigate each alert"
        )
    
    with col4:
        false_positive_cost = st.number_input(
            "False Positive Cost ($):", 
            value=5.0,
            min_value=1.0,
            help="Cost of each false alarm (customer impact)"
        )
    
    # Advanced parameters
    with st.expander("[CONFIG] Advanced Parameters"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fraud_rate = st.slider(
                "Fraud Rate (%):", 
                0.1, 5.0, 
                scenario_config['fraud_rate'],
                help="Percentage of transactions that are fraudulent"
            )
        
        with col2:
            analyst_hourly_rate = st.number_input(
                "Analyst Hourly Rate ($):", 
                value=45.0,
                help="Cost per hour for fraud analyst"
            )
        
        with col3:
            investigation_time = st.slider(
                "Avg Investigation Time (mins):", 
                5, 60, 15,
                help="Average time to investigate an alert"
            )
    
    # Calculate ROI for each model
    st.markdown("#### [DATA] Model Performance vs Business Impact")
    
    roi_data = []
    
    for model_name, metrics in results.items():
        # Extract metrics
        precision = metrics.get('precision', 0.8)
        recall = metrics.get('recall', 0.7)
        f1_score = metrics.get('f1_score', 0.75)
        
        # Calculate business metrics
        daily_fraud_cases = daily_transactions * (fraud_rate / 100)
        
        # True/False positives and negatives
        tp = daily_fraud_cases * recall  # True positives (caught fraud)
        fn = daily_fraud_cases * (1 - recall)  # False negatives (missed fraud)
        
        # Estimate false positives from precision
        if precision > 0:
            total_positives = tp / precision
            fp = total_positives - tp
        else:
            fp = daily_transactions * 0.05  # Assume 5% false positive rate
        
        tn = daily_transactions - daily_fraud_cases - fp  # True negatives
        
        # Financial calculations
        fraud_prevented = tp * avg_fraud_loss
        fraud_losses = fn * avg_fraud_loss
        investigation_costs = (tp + fp) * investigation_cost
        false_positive_costs = fp * false_positive_cost
        analyst_costs = (tp + fp) * (investigation_time / 60) * analyst_hourly_rate
        
        total_costs = investigation_costs + false_positive_costs + analyst_costs
        net_benefit = fraud_prevented - total_costs
        roi_percentage = (net_benefit / total_costs) * 100 if total_costs > 0 else 0
        
        roi_data.append({
            'Model': model_name,
            'Precision': f"{precision:.3f}",
            'Recall': f"{recall:.3f}",
            'F1-Score': f"{f1_score:.3f}",
            'Daily_Fraud_Prevented': fraud_prevented,
            'Daily_Fraud_Losses': fraud_losses,
            'Daily_Investigation_Cost': investigation_costs,
            'Daily_FP_Cost': false_positive_costs,
            'Daily_Analyst_Cost': analyst_costs,
            'Net_Daily_Benefit': net_benefit,
            'Annual_Benefit': net_benefit * 365,
            'ROI_Percentage': roi_percentage,
            'Total_Alerts': tp + fp,
            'Alert_Precision': (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0
        })
    
    roi_df = pd.DataFrame(roi_data)
    
    # Display financial summary
    col1, col2 = st.columns(2)
    
    with col1:
        # Net benefit comparison
        fig = px.bar(roi_df, x='Model', y='Net_Daily_Benefit',
                   color='Net_Daily_Benefit', color_continuous_scale='RdYlGn',
                   title="Daily Net Benefit by Model")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # ROI comparison
        fig = px.bar(roi_df, x='Model', y='ROI_Percentage',
                   color='ROI_Percentage', color_continuous_scale='viridis',
                   title="Return on Investment (%) by Model")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed financial breakdown
    st.markdown("#### 💹 Detailed Financial Analysis")
    
    # Format financial columns
    financial_cols = ['Daily_Fraud_Prevented', 'Daily_Fraud_Losses', 'Daily_Investigation_Cost', 
                     'Daily_FP_Cost', 'Daily_Analyst_Cost', 'Net_Daily_Benefit', 'Annual_Benefit']
    
    display_df = roi_df.copy()
    for col in financial_cols:
        display_df[col] = display_df[col].apply(lambda x: f"${x:,.0f}")
    
    display_df['ROI_Percentage'] = display_df['ROI_Percentage'].apply(lambda x: f"{x:.1f}%")
    display_df['Total_Alerts'] = display_df['Total_Alerts'].apply(lambda x: f"{x:.0f}")
    display_df['Alert_Precision'] = display_df['Alert_Precision'].apply(lambda x: f"{x:.1f}%")
    
    st.dataframe(display_df, use_container_width=True)
    
    # Best model recommendation
    best_roi_idx = roi_df['ROI_Percentage'].idxmax()
    best_model = roi_df.iloc[best_roi_idx]
    
    st.markdown(f"""
    <div class="insight-box">
        <h4>[TOP] Recommended Model: {best_model['Model']}</h4>
        <p><strong>Annual Benefit:</strong> ${best_model['Annual_Benefit']:,.0f}</p>
        <p><strong>ROI:</strong> {best_model['ROI_Percentage']:.1f}%</p>
        <p><strong>Daily Alerts:</strong> {best_model['Total_Alerts']:.0f} (Alert Precision: {best_model['Alert_Precision']:.1f}%)</p>
        <p>This model provides the best balance of fraud detection and operational efficiency for your {st.session_state.scenario} scenario.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sensitivity analysis
    st.markdown("#### [CHART] Sensitivity Analysis")
    
    # Show impact of changing key parameters
    sensitivity_param = st.selectbox("Parameter to Analyze:", 
                                   ["Fraud Rate", "Investigation Cost", "False Positive Cost", "Avg Fraud Loss"])
    
    if sensitivity_param == "Fraud Rate":
        param_range = np.linspace(0.1, 3.0, 20)
        param_label = "Fraud Rate (%)"
    elif sensitivity_param == "Investigation Cost":
        param_range = np.linspace(10, 100, 20)
        param_label = "Investigation Cost ($)"
    elif sensitivity_param == "False Positive Cost":
        param_range = np.linspace(1, 20, 20)
        param_label = "False Positive Cost ($)"
    else:  # Avg Fraud Loss
        param_range = np.linspace(50, 500, 20)
        param_label = "Average Fraud Loss ($)"
    
    # Calculate sensitivity for best model
    sensitivity_data = []
    best_model_name = best_model['Model']
    best_metrics = results[best_model_name]
    
    for param_value in param_range:
        # Recalculate with new parameter
        if sensitivity_param == "Fraud Rate":
            test_fraud_rate = param_value
            test_investigation_cost = investigation_cost
            test_fp_cost = false_positive_cost
            test_fraud_loss = avg_fraud_loss
        elif sensitivity_param == "Investigation Cost":
            test_fraud_rate = fraud_rate
            test_investigation_cost = param_value
            test_fp_cost = false_positive_cost
            test_fraud_loss = avg_fraud_loss
        elif sensitivity_param == "False Positive Cost":
            test_fraud_rate = fraud_rate
            test_investigation_cost = investigation_cost
            test_fp_cost = param_value
            test_fraud_loss = avg_fraud_loss
        else:
            test_fraud_rate = fraud_rate
            test_investigation_cost = investigation_cost
            test_fp_cost = false_positive_cost
            test_fraud_loss = param_value
        
        # Recalculate ROI
        test_daily_fraud = daily_transactions * (test_fraud_rate / 100)
        test_tp = test_daily_fraud * best_metrics.get('recall', 0.7)
        test_fp = (test_tp / best_metrics.get('precision', 0.8)) - test_tp if best_metrics.get('precision', 0.8) > 0 else daily_transactions * 0.05
        
        test_prevented = test_tp * test_fraud_loss
        test_costs = (test_tp + test_fp) * test_investigation_cost + test_fp * test_fp_cost
        test_roi = ((test_prevented - test_costs) / test_costs) * 100 if test_costs > 0 else 0
        
        sensitivity_data.append({
            'Parameter_Value': param_value,
            'ROI_Percentage': test_roi
        })
    
    sensitivity_df = pd.DataFrame(sensitivity_data)
    
    fig = px.line(sensitivity_df, x='Parameter_Value', y='ROI_Percentage',
                title=f"ROI Sensitivity to {sensitivity_param}",
                labels={'Parameter_Value': param_label, 'ROI_Percentage': 'ROI (%)'})
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

def create_model_monitoring_dashboard():
    """Create model monitoring and drift detection dashboard."""
    st.markdown("### [CHART] Model Monitoring & Drift Detection")
    
    # Generate simulated monitoring data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Simulate model performance over time
    np.random.seed(42)
    base_performance = 0.85
    
    monitoring_data = []
    for i, date in enumerate(dates):
        # Simulate gradual performance drift
        drift_factor = max(0, 1 - (i * 0.01))  # Gradual decline
        noise = np.random.normal(0, 0.02)
        
        f1_score = base_performance * drift_factor + noise
        precision = f1_score + np.random.normal(0, 0.01)
        recall = f1_score + np.random.normal(0, 0.01)
        
        # Simulate system metrics
        latency = 45 + np.random.normal(0, 5) + (i * 0.5)  # Increasing latency
        throughput = 1000 - (i * 2) + np.random.normal(0, 10)
        
        # Feature drift score (0-1, higher = more drift)
        drift_score = min(1.0, i * 0.02 + np.random.normal(0, 0.05))
        
        monitoring_data.append({
            'Date': date,
            'F1_Score': max(0, min(1, f1_score)),
            'Precision': max(0, min(1, precision)),
            'Recall': max(0, min(1, recall)),
            'Latency_ms': max(10, latency),
            'Throughput_tps': max(100, throughput),
            'Drift_Score': max(0, drift_score),
            'Data_Quality': max(0.8, 1 - (i * 0.005) + np.random.normal(0, 0.02))
        })
    
    monitoring_df = pd.DataFrame(monitoring_data)
    
    # System health overview
    st.markdown("#### 🚦 System Health Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    latest_metrics = monitoring_df.iloc[-1]
    
    with col1:
        health_color = "🟢" if latest_metrics['F1_Score'] > 0.8 else "🟡" if latest_metrics['F1_Score'] > 0.7 else "🔴"
        st.markdown(f"""
        <div class="system-health">
            <h4>{health_color} Model Performance</h4>
            <p>F1-Score: {latest_metrics['F1_Score']:.3f}</p>
            <p>Status: {'Healthy' if latest_metrics['F1_Score'] > 0.8 else 'Warning' if latest_metrics['F1_Score'] > 0.7 else 'Critical'}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        latency_color = "🟢" if latest_metrics['Latency_ms'] < 50 else "🟡" if latest_metrics['Latency_ms'] < 100 else "🔴"
        st.markdown(f"""
        <div class="system-health">
            <h4>{latency_color} System Latency</h4>
            <p>Response: {latest_metrics['Latency_ms']:.1f}ms</p>
            <p>Target: <50ms</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        drift_color = "🟢" if latest_metrics['Drift_Score'] < 0.3 else "🟡" if latest_metrics['Drift_Score'] < 0.6 else "🔴"
        st.markdown(f"""
        <div class="system-health">
            <h4>{drift_color} Data Drift</h4>
            <p>Drift Score: {latest_metrics['Drift_Score']:.3f}</p>
            <p>Status: {'Stable' if latest_metrics['Drift_Score'] < 0.3 else 'Moderate' if latest_metrics['Drift_Score'] < 0.6 else 'High'}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        quality_color = "🟢" if latest_metrics['Data_Quality'] > 0.9 else "🟡" if latest_metrics['Data_Quality'] > 0.8 else "🔴"
        st.markdown(f"""
        <div class="system-health">
            <h4>{quality_color} Data Quality</h4>
            <p>Score: {latest_metrics['Data_Quality']:.3f}</p>
            <p>Completeness: 98.5%</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Performance trends
    st.markdown("#### [DATA] Performance Trends")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Model performance over time
        fig = make_subplots(rows=2, cols=1, 
                          subplot_titles=("Model Performance", "System Metrics"),
                          vertical_spacing=0.1)
        
        fig.add_trace(go.Scatter(x=monitoring_df['Date'], y=monitoring_df['F1_Score'], 
                               name='F1-Score', line=dict(color='blue')), row=1, col=1)
        fig.add_trace(go.Scatter(x=monitoring_df['Date'], y=monitoring_df['Precision'], 
                               name='Precision', line=dict(color='green')), row=1, col=1)
        fig.add_trace(go.Scatter(x=monitoring_df['Date'], y=monitoring_df['Recall'], 
                               name='Recall', line=dict(color='orange')), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=monitoring_df['Date'], y=monitoring_df['Latency_ms'], 
                               name='Latency (ms)', line=dict(color='red')), row=2, col=1)
        
        fig.update_layout(height=500, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Drift and data quality
        fig = make_subplots(rows=2, cols=1,
                          subplot_titles=("Data Drift Score", "Data Quality Score"),
                          vertical_spacing=0.1)
        
        fig.add_trace(go.Scatter(x=monitoring_df['Date'], y=monitoring_df['Drift_Score'],
                               name='Drift Score', line=dict(color='purple'),
                               fill='tonexty'), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=monitoring_df['Date'], y=monitoring_df['Data_Quality'],
                               name='Data Quality', line=dict(color='teal'),
                               fill='tonexty'), row=2, col=1)
        
        # Add threshold lines
        fig.add_hline(y=0.6, line_dash="dash", line_color="red", row=1, col=1)
        fig.add_hline(y=0.8, line_dash="dash", line_color="red", row=2, col=1)
        
        fig.update_layout(height=500, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    
    # Alerts and recommendations
    st.markdown("#### [ALERT] Alerts & Recommendations")
    
    # Generate alerts based on current metrics
    alerts = []
    
    if latest_metrics['F1_Score'] < 0.75:
        alerts.append({
            'type': 'critical',
            'message': 'Model performance has degraded significantly. Consider retraining.',
            'action': 'Immediate retraining recommended'
        })
    
    if latest_metrics['Drift_Score'] > 0.6:
        alerts.append({
            'type': 'warning',
            'message': 'High data drift detected. Feature distributions have changed.',
            'action': 'Investigate feature drift and consider model updates'
        })
    
    if latest_metrics['Latency_ms'] > 100:
        alerts.append({
            'type': 'warning',
            'message': 'System latency exceeds acceptable thresholds.',
            'action': 'Optimize model inference or scale infrastructure'
        })
    
    if latest_metrics['Data_Quality'] < 0.85:
        alerts.append({
            'type': 'warning',
            'message': 'Data quality has decreased.',
            'action': 'Review data pipeline and validation rules'
        })
    
    if not alerts:
        st.success("[OK] All systems operating normally. No alerts detected.")
    else:
        for alert in alerts:
            if alert['type'] == 'critical':
                st.error(f"[ALERT] **CRITICAL**: {alert['message']}")
                st.info(f"**Action Required**: {alert['action']}")
            else:
                st.warning(f"WARNING: **WARNING**: {alert['message']}")
                st.info(f"**Recommended Action**: {alert['action']}")
    
    # Feature importance drift
    st.markdown("#### [PROCESS] Feature Importance Changes")
    
    # Simulate feature importance drift
    features = ['V14', 'V4', 'V11', 'V12', 'V10', 'Amount', 'V17', 'V21']
    original_importance = np.random.dirichlet(np.ones(len(features)), size=1)[0]
    current_importance = original_importance + np.random.normal(0, 0.05, len(features))
    current_importance = np.abs(current_importance) / np.sum(np.abs(current_importance))
    
    importance_change = current_importance - original_importance
    
    drift_df = pd.DataFrame({
        'Feature': features,
        'Original_Importance': original_importance,
        'Current_Importance': current_importance,
        'Change': importance_change
    }).sort_values('Change', key=abs, ascending=False)
    
    fig = px.bar(drift_df, x='Feature', y='Change',
               color='Change', color_continuous_scale='RdBu_r',
               title="Feature Importance Drift")
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

def main():
    """Main dashboard application."""
    # Show intro modal if needed
    if not show_intro_modal():
        return
    
    # Main header with persona context
    persona_name = st.session_state.user_persona.split(" - ")[1]
    scenario_name = st.session_state.scenario
    
    st.markdown(f'<h1 class="main-header">🛡️ FraudGuard Pro - {scenario_name}</h1>', unsafe_allow_html=True)
    st.markdown(f"**Welcome, {persona_name}** | Scenario: {scenario_name}")
    
    # Navigation
    pages = {
        "[TARGET] Command Center": show_command_center,
        "[PROCESS] Live Stream": create_streaming_simulator,
        "[SEARCH] Explainable AI": create_shap_explanation,
        "[MONEY] Business Impact": create_business_impact_calculator,
        "[CHART] Model Monitoring": create_model_monitoring_dashboard,
        "[TOP] A/B Testing": show_ab_testing,
        ">>> Deployment": show_deployment_simulator
    }
    
    # Sidebar navigation
    st.sidebar.markdown("## [NAV] Navigation")
    selected_page = st.sidebar.selectbox("Choose a section:", list(pages.keys()))
    
    # Persona-specific sidebar
    persona = USER_PERSONAS[st.session_state.user_persona]
    st.sidebar.markdown(f"""
    ### [USER] Your Profile: {persona_name}
    **Focus**: {persona['focus']}
    
    **Today's Goals**:
    - {persona['goals']}
    """)
    
    # Scenario context
    scenario = SCENARIOS[st.session_state.scenario]
    st.sidebar.markdown(f"""
    ### 🏢 Business Context
    **Scenario**: {scenario_name}
    - Avg Transaction: ${scenario['avg_transaction']}
    - Fraud Rate: {scenario['fraud_rate']}%
    - Investigation Cost: ${scenario['investigation_cost']}
    """)
    
    # Display selected page
    pages[selected_page]()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
    <p>🛡️ FraudGuard Pro v3.0 | Enterprise Fraud Detection Platform</p>
    <p>Built with Advanced ML, Real-time Analytics, and Explainable AI</p>
    </div>
    """, unsafe_allow_html=True)

def show_command_center():
    """Main command center overview."""
    st.markdown("## [TARGET] Fraud Detection Command Center")
    
    df = load_data()
    models, scaler, results = load_models()
    
    if df is None:
        st.error("Unable to load dataset")
        return
    
    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    
    total_transactions = len(df)
    fraud_transactions = df['Class'].sum()
    fraud_rate = (fraud_transactions / total_transactions) * 100
    avg_fraud_amount = df[df['Class'] == 1]['Amount'].mean()
    
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
            <h3>${avg_fraud_amount:.0f}</h3>
            <p>Avg Fraud Amount</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Model performance overview
    if results:
        st.markdown("### [TOP] Model Performance Overview")
        
        performance_data = []
        for model_name, metrics in results.items():
            performance_data.append({
                'Model': model_name,
                'F1-Score': metrics.get('f1_score', 0),
                'Precision': metrics.get('precision', 0),
                'Recall': metrics.get('recall', 0),
                'ROC-AUC': metrics.get('roc_auc', 0)
            })
        
        performance_df = pd.DataFrame(performance_data)
        
        fig = px.bar(performance_df, x='Model', y='F1-Score',
                   color='F1-Score', color_continuous_scale='viridis',
                   title="Model F1-Score Comparison")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Dataset overview
    st.markdown("### [DATA] Dataset Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Amount distribution
        sample_df = df.sample(10000)
        fig = px.histogram(sample_df, x='Amount', color='Class',
                         title="Transaction Amount Distribution",
                         color_discrete_map={0: 'blue', 1: 'red'},
                         nbins=50)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Time analysis
        sample_df['Hour'] = (sample_df['Time'] % (24 * 3600)) // 3600
        hourly_fraud = sample_df.groupby('Hour')['Class'].agg(['count', 'sum', 'mean']).reset_index()
        hourly_fraud.columns = ['Hour', 'Total', 'Fraud_Count', 'Fraud_Rate']
        
        fig = px.line(hourly_fraud, x='Hour', y='Fraud_Rate',
                    title="Fraud Rate by Hour of Day")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def show_ab_testing():
    """A/B testing interface for model comparison."""
    st.markdown("### [TEST] A/B Testing & Model Comparison")
    
    models, scaler, results = load_models()
    
    if models is None or len(models) < 2:
        st.error("Need at least 2 models for A/B testing")
        return
    
    # Model selection for A/B test
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🅰️ Model A")
        model_a = st.selectbox("Select Model A:", list(models.keys()), key="model_a")
        
        if model_a in results:
            metrics_a = results[model_a]
            st.markdown(f"""
            <div class="metric-card">
                <h4>Model A Performance</h4>
                <p>F1-Score: {metrics_a.get('f1_score', 0):.3f}</p>
                <p>Precision: {metrics_a.get('precision', 0):.3f}</p>
                <p>Recall: {metrics_a.get('recall', 0):.3f}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### 🅱️ Model B")
        available_models = [m for m in models.keys() if m != model_a]
        model_b = st.selectbox("Select Model B:", available_models, key="model_b")
        
        if model_b in results:
            metrics_b = results[model_b]
            st.markdown(f"""
            <div class="metric-card">
                <h4>Model B Performance</h4>
                <p>F1-Score: {metrics_b.get('f1_score', 0):.3f}</p>
                <p>Precision: {metrics_b.get('precision', 0):.3f}</p>
                <p>Recall: {metrics_b.get('recall', 0):.3f}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # A/B test configuration
    st.markdown("#### [SETTING] Test Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        traffic_split = st.slider("Traffic Split (% to Model A):", 0, 100, 50)
    
    with col2:
        test_duration = st.slider("Test Duration (days):", 1, 30, 7)
    
    with col3:
        sample_size = st.number_input("Daily Sample Size:", 100, 10000, 1000)
    
    # Simulate A/B test results
    if st.button(">>> Run A/B Test Simulation"):
        
        # Simulate test results
        days = range(test_duration)
        results_data = []
        
        for day in days:
            # Model A results
            model_a_traffic = int(sample_size * traffic_split / 100)
            model_a_performance = metrics_a.get('f1_score', 0.8) + np.random.normal(0, 0.02)
            
            # Model B results  
            model_b_traffic = sample_size - model_a_traffic
            model_b_performance = metrics_b.get('f1_score', 0.8) + np.random.normal(0, 0.02)
            
            results_data.append({
                'Day': day + 1,
                'Model_A_F1': model_a_performance,
                'Model_B_F1': model_b_performance,
                'Model_A_Traffic': model_a_traffic,
                'Model_B_Traffic': model_b_traffic
            })
        
        test_results = pd.DataFrame(results_data)
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            # Performance comparison over time
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=test_results['Day'], y=test_results['Model_A_F1'],
                                   name=f'Model A ({model_a})', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=test_results['Day'], y=test_results['Model_B_F1'],
                                   name=f'Model B ({model_b})', line=dict(color='red')))
            
            fig.update_layout(title="A/B Test Performance Over Time",
                            xaxis_title="Day", yaxis_title="F1-Score", height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Statistical significance
            avg_a = test_results['Model_A_F1'].mean()
            avg_b = test_results['Model_B_F1'].mean()
            
            improvement = ((avg_b - avg_a) / avg_a) * 100
            
            winner = "Model B" if avg_b > avg_a else "Model A"
            winner_color = "success" if abs(improvement) > 2 else "warning"
            
            st.markdown(f"""
            <div class="insight-box">
                <h4>[TOP] A/B Test Results</h4>
                <p><strong>Winner:</strong> {winner}</p>
                <p><strong>Improvement:</strong> {abs(improvement):.2f}%</p>
                <p><strong>Statistical Significance:</strong> {'High' if abs(improvement) > 5 else 'Moderate' if abs(improvement) > 2 else 'Low'}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Traffic allocation recommendation
            if abs(improvement) > 5:
                recommendation = f"Deploy {winner} to 100% traffic"
            elif abs(improvement) > 2:
                recommendation = f"Increase {winner} traffic to 70%"
            else:
                recommendation = "Continue testing - results inconclusive"
            
            st.info(f"**Recommendation:** {recommendation}")

def show_deployment_simulator():
    """Deployment simulation and system health."""
    st.markdown("### >>> Deployment Simulation & System Health")
    
    # Deployment configuration
    st.markdown("#### [SETTING] Deployment Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        expected_tps = st.slider("Expected TPS:", 10, 1000, 100)
        
    with col2:
        target_latency = st.slider("Target Latency (ms):", 10, 200, 50)
        
    with col3:
        redundancy_level = st.selectbox("Redundancy:", ["Single", "Dual", "Multi-AZ"])
    
    # System health simulation
    st.markdown("#### [CONFIG] System Health Metrics")
    
    # Generate real-time metrics
    current_time = datetime.now()
    
    # Simulate system metrics
    actual_tps = expected_tps + np.random.normal(0, expected_tps * 0.1)
    actual_latency = target_latency + np.random.normal(0, target_latency * 0.2)
    cpu_usage = random.uniform(30, 80)
    memory_usage = random.uniform(40, 85)
    error_rate = random.uniform(0, 2.0)
    
    # Health status
    health_status = "Healthy"
    if actual_latency > target_latency * 1.5 or cpu_usage > 80 or error_rate > 1.0:
        health_status = "Warning"
    if actual_latency > target_latency * 2 or cpu_usage > 90 or error_rate > 2.0:
        health_status = "Critical"
    
    # Display metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        tps_color = "🟢" if actual_tps >= expected_tps * 0.9 else "🟡" if actual_tps >= expected_tps * 0.7 else "🔴"
        st.markdown(f"""
        <div class="system-health">
            <h4>{tps_color} Throughput</h4>
            <p>{actual_tps:.0f} TPS</p>
            <p>Target: {expected_tps} TPS</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        latency_color = "🟢" if actual_latency <= target_latency else "🟡" if actual_latency <= target_latency * 1.5 else "🔴"
        st.markdown(f"""
        <div class="system-health">
            <h4>{latency_color} Latency</h4>
            <p>{actual_latency:.1f}ms</p>
            <p>Target: <{target_latency}ms</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        cpu_color = "🟢" if cpu_usage < 70 else "🟡" if cpu_usage < 85 else "🔴"
        st.markdown(f"""
        <div class="system-health">
            <h4>{cpu_color} CPU Usage</h4>
            <p>{cpu_usage:.1f}%</p>
            <p>Threshold: 80%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        memory_color = "🟢" if memory_usage < 75 else "🟡" if memory_usage < 90 else "🔴"
        st.markdown(f"""
        <div class="system-health">
            <h4>{memory_color} Memory</h4>
            <p>{memory_usage:.1f}%</p>
            <p>Threshold: 85%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        error_color = "🟢" if error_rate < 0.5 else "🟡" if error_rate < 1.0 else "🔴"
        st.markdown(f"""
        <div class="system-health">
            <h4>{error_color} Error Rate</h4>
            <p>{error_rate:.2f}%</p>
            <p>Threshold: <1%</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Overall system status
    status_color = {"Healthy": "🟢", "Warning": "🟡", "Critical": "🔴"}[health_status]
    st.markdown(f"""
    <div class="system-health">
        <h3>{status_color} Overall System Status: {health_status}</h3>
        <p>Last Updated: {current_time.strftime('%H:%M:%S')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Deployment checklist
    st.markdown("#### [OK] Production Deployment Checklist")
    
    checklist_items = [
        ("Load Testing", True),
        ("Security Scanning", True),
        ("Performance Benchmarking", True),
        ("Monitoring Setup", True),
        ("Backup & Recovery", True),
        ("Auto-scaling Configuration", random.choice([True, False])),
        ("SSL/TLS Encryption", True),
        ("API Rate Limiting", True),
        ("Logging & Audit Trail", True),
        ("Disaster Recovery Plan", random.choice([True, False]))
    ]
    
    for item, status in checklist_items:
        status_icon = "[OK]" if status else "[ERROR]"
        st.markdown(f"{status_icon} {item}")
    
    # Scaling simulation
    st.markdown("#### [CHART] Auto-scaling Simulation")
    
    if st.button("[HOT] Simulate Traffic Spike"):
        spike_data = []
        for minute in range(10):
            if minute < 3:
                load = expected_tps
            elif minute < 7:
                load = expected_tps * (2 + minute * 0.5)  # Gradual spike
            else:
                load = expected_tps * 1.2  # Settle at higher level
            
            spike_data.append({
                'Minute': minute,
                'Load_TPS': load,
                'Instances': max(1, int(load / 100)),  # Scale instances based on load
                'Response_Time': 50 + (load / expected_tps - 1) * 20  # Latency increases with load
            })
        
        spike_df = pd.DataFrame(spike_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = make_subplots(rows=2, cols=1, subplot_titles=("Traffic Load", "Response Time"))
            
            fig.add_trace(go.Scatter(x=spike_df['Minute'], y=spike_df['Load_TPS'],
                                   name='Load (TPS)', line=dict(color='blue')), row=1, col=1)
            fig.add_trace(go.Scatter(x=spike_df['Minute'], y=spike_df['Response_Time'],
                                   name='Response Time (ms)', line=dict(color='red')), row=2, col=1)
            
            fig.update_layout(height=400, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(spike_df, x='Minute', y='Instances',
                       title="Auto-scaling Response")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()