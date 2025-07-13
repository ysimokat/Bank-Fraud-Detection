#!/bin/bash
# Quick Start Script for Bank Fraud Detection System
# Author: Yanhong Simokat

echo "🚀 Bank Fraud Detection System - Quick Start"
echo "==========================================="

# Check Python installation
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✅ Python detected: $(python3 --version)"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    source venv/Scripts/activate
else
    # Linux/Mac
    source venv/bin/activate
fi

# Check if requirements are installed
echo "📚 Checking dependencies..."
if ! python -c "import pandas" &> /dev/null; then
    echo "📥 Installing requirements..."
    pip install -r requirements.txt
fi

# Check for dataset
if [ ! -f "creditcard.csv" ]; then
    echo "⚠️  Dataset not found!"
    echo "Please download creditcard.csv from:"
    echo "https://www.kaggle.com/mlg-ulb/creditcardfraud"
    echo "and place it in the project root directory."
    exit 1
fi

echo "✅ Dataset found"

# Menu
echo ""
echo "What would you like to run?"
echo "1. 🚀 Integrated Pipeline - ALL Models (Recommended)"
echo "2. ⚡ Quick Integrated Pipeline (Faster)"
echo "3. 📊 Basic Fraud Detection Models Only"
echo "4. 🧠 Enhanced Deep Learning Models Only"
echo "5. 📱 Professional Dashboard (Interactive)"
echo "6. 🌐 API Server"
echo "7. 🖥️ GPU Configuration Test"
echo "8. 📓 Jupyter Notebook Tutorials"
echo "9. 📦 Install/Update Requirements"
echo "0. 🚪 Exit"

read -p "Enter your choice (0-9): " choice

case $choice in
    1)
        echo "🚀 Running Integrated Pipeline with ALL Models..."
        echo "This includes: Basic ML, XGBoost, LightGBM, Deep Learning, GNN"
        python integrated_fraud_pipeline.py
        ;;
    2)
        echo "⚡ Running Quick Integrated Pipeline (Basic + Enhanced only)..."
        python integrated_fraud_pipeline.py --quick
        ;;
    3)
        echo "📊 Running Basic Fraud Detection Models Only..."
        python fraud_detection_models.py
        ;;
    4)
        echo "🧠 Running Enhanced Deep Learning Models Only..."
        python enhanced_deep_learning.py
        ;;
    5)
        echo "📱 Starting Professional Dashboard..."
        echo "Open http://localhost:8501 in your browser"
        python professional_fraud_dashboard.py
        ;;
    6)
        echo "🌐 Starting API Server..."
        echo "API will be available at http://localhost:8000"
        echo "Documentation at http://localhost:8000/docs"
        python enhanced_fraud_api.py
        ;;
    7)
        echo "🖥️ Testing GPU Configuration..."
        python gpu_config.py
        ;;
    8)
        echo "📓 Starting Jupyter Notebook..."
        cd tutorials
        jupyter notebook
        ;;
    9)
        echo "📦 Installing/Updating Requirements..."
        pip install -r requirements.txt
        ;;
    0)
        echo "👋 Goodbye!"
        exit 0
        ;;
    *)
        echo "❌ Invalid choice. Please run the script again."
        exit 1
        ;;
esac

echo ""
echo "✅ Task completed!"
echo "Run this script again to try other components."