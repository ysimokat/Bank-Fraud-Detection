# 🛡️ Credit Card Fraud Detection System

A comprehensive machine learning system for detecting credit card fraud using state-of-the-art techniques including deep learning, graph neural networks, and ensemble methods.

## 🎯 Overview

This project implements multiple approaches to fraud detection:
- Traditional ML models (Random Forest, XGBoost, LightGBM)
- Deep Learning with specialized loss functions
- Graph Neural Networks for relationship analysis
- Real-time streaming systems
- Active learning for continuous improvement

## 📊 Performance

| Model Type | F1-Score | Training Time |
|------------|----------|---------------|
| Random Forest | ~0.85 | 2-3 min |
| XGBoost | ~0.86 | 3-5 min |
| Deep Learning | ~0.87 | 10-15 min |
| Graph Neural Network | ~0.87 | 15-20 min |
| Ensemble (All Models) | ~0.91 | 45-60 min |

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+**
2. **Dataset**: Download from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)
   - File: `creditcard.csv`
   - Place in project root directory

### Installation

```bash
# Clone repository
git clone https://github.com/ysimokat/Bank-Fraud-Detection.git
cd Bank-Fraud-Detection

# Create virtual environment
python -m venv venv

# Activate environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the System

#### Option 1: Quick Start (Recommended for first-time users)
```bash
# Windows
run_windows.bat

# Linux/Mac
./RUN_QUICK_START.sh
```

#### Option 2: Command Line

**Basic Pipeline (10 minutes)**
```bash
python integrated_fraud_pipeline.py --quick
```

**Full Pipeline (30-45 minutes)**
```bash
python integrated_fraud_pipeline.py
```

**Advanced Pipeline with All Systems (60+ minutes)**
```bash
python advanced_integrated_pipeline.py
```

**View Results Dashboard**
```bash
python professional_fraud_dashboard.py
# Open http://localhost:8501
```

## 📚 Project Structure

```
Bank_Fraud_Detection/
│
├── 🎯 Main Pipelines
│   ├── integrated_fraud_pipeline.py      # All basic + enhanced models
│   ├── advanced_integrated_pipeline.py   # Includes advanced systems
│   └── professional_fraud_dashboard.py   # Interactive dashboard
│
├── 🧩 Model Components
│   ├── fraud_detection_models.py         # Basic ML models
│   ├── enhanced_fraud_models.py          # XGBoost, LightGBM, CatBoost
│   ├── enhanced_deep_learning.py         # Neural networks
│   ├── graph_neural_network.py           # Graph neural networks
│   └── heterogeneous_gnn.py              # Advanced GNN
│
├── 🔧 Advanced Systems
│   ├── online_streaming_system.py        # Real-time processing
│   ├── hybrid_ensemble_system.py         # Meta-learning ensemble
│   ├── enhanced_active_learning.py       # Human-in-the-loop
│   └── advanced_model_calibration.py     # Probability calibration
│
├── 📱 Deployment
│   ├── enhanced_fraud_api.py             # REST API server
│   └── gpu_config.py                     # GPU optimization
│
└── 📓 Learning Resources
    └── tutorials/                        # Jupyter notebooks
```

## 🎓 Learning Path

### Beginner (Week 1)
1. Run basic pipeline: `python integrated_fraud_pipeline.py --quick`
2. Explore dashboard: `python professional_fraud_dashboard.py`
3. Complete tutorials in `tutorials/` folder

### Intermediate (Week 2)
1. Run full pipeline: `python integrated_fraud_pipeline.py`
2. Study deep learning models
3. Understand ensemble methods

### Advanced (Week 3+)
1. Run advanced pipeline: `python advanced_integrated_pipeline.py`
2. Explore graph neural networks
3. Implement custom modifications

## 💡 Key Features

### 1. Multiple Model Types
- **Traditional ML**: Random Forest, Logistic Regression, SVM
- **Boosting**: XGBoost, LightGBM, CatBoost
- **Deep Learning**: Focal Loss, Weighted BCE, Autoencoders
- **Graph Networks**: GNN, Heterogeneous GNN

### 2. Advanced Techniques
- **Imbalanced Learning**: SMOTE, Focal Loss, Class weights
- **Anomaly Detection**: Isolation Forest, One-Class SVM, Autoencoders
- **Ensemble Methods**: Voting, Stacking, Meta-learning
- **Online Learning**: Streaming updates, Drift detection

### 3. Production Features
- **REST API**: Fast inference endpoint
- **Dashboard**: Real-time monitoring
- **GPU Support**: Automatic GPU detection and optimization
- **Active Learning**: Continuous improvement with feedback

### 4. Comprehensive Evaluation
- Multiple metrics (F1, ROC-AUC, Precision-Recall)
- Business impact analysis
- Model explainability (SHAP)
- A/B testing framework

## 🖥️ GPU Support

The system automatically detects and uses available GPUs:
- NVIDIA GPUs (CUDA)
- AMD GPUs (ROCm)
- Apple Silicon (MPS)

Test GPU configuration:
```bash
python gpu_config.py
```

## 📊 API Usage

Start the API server:
```bash
python enhanced_fraud_api.py
```

Make predictions:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [...]}'
```

API documentation available at: http://localhost:8000/docs

## 🔍 Model Interpretability

The dashboard includes:
- SHAP values for feature importance
- Partial dependence plots
- Individual prediction explanations
- Model performance monitoring

## 📈 Extending the System

### Adding New Models
1. Create new model class in appropriate file
2. Add to pipeline in `integrated_fraud_pipeline.py`
3. Update dashboard to display results

### Custom Features
1. Modify `data_preprocessing.py`
2. Update feature engineering in pipelines
3. Retrain models

## 🐛 Troubleshooting

### Common Issues

**Out of Memory**
- Use `--quick` mode
- Reduce batch sizes
- Enable GPU if available

**Missing Dependencies**
```bash
pip install --upgrade -r requirements.txt
```

**Dataset Not Found**
- Ensure `creditcard.csv` is in project root
- Download from Kaggle link above

## 📝 Citation

If you use this project in research, please cite:
```
@software{fraud_detection_system,
  title = {Credit Card Fraud Detection System},
  author = {Yanhong Simokat},
  year = {2024},
  url = {https://github.com/ysimokat/Bank-Fraud-Detection}
}
```

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## 📧 Contact

- Author: Yanhong Simokat
- Email: yanhong7369@gmail.com
- GitHub: [@ysimokat](https://github.com/ysimokat)

## 🙏 Acknowledgments

- Dataset: [Machine Learning Group - ULB](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- Inspired by recent advances in fraud detection research
- Built with PyTorch, Scikit-learn, XGBoost, and Streamlit