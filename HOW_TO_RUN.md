# Bank Fraud Detection System - How to Run

## Prerequisites

### 1. System Requirements
- Python 3.8 or higher
- 8GB+ RAM (16GB recommended)
- GPU (optional but recommended for deep learning models)
  - NVIDIA GPU with CUDA support
  - AMD GPU with ROCm support
  - Apple Silicon with MPS support

### 2. Dataset
Download the Credit Card Fraud Detection dataset from Kaggle:
- URL: https://www.kaggle.com/mlg-ulb/creditcardfraud
- File: `creditcard.csv` (~150MB)
- Place it in the project root directory

### 3. Install Dependencies
```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# For GPU support (optional):
# NVIDIA GPUs:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# For RAPIDS (GPU-accelerated ML):
# Follow instructions at https://rapids.ai/start.html
```

## Quick Start

### 1. Integrated Pipeline (Recommended) 🚀
```bash
# Run ALL models in one integrated pipeline
python integrated_fraud_pipeline.py

# Quick mode (skip deep learning & GNN for faster results)
python integrated_fraud_pipeline.py --quick

# Custom options
python integrated_fraud_pipeline.py --skip-dl  # Skip deep learning only
python integrated_fraud_pipeline.py --skip-gnn # Skip GNN only
```
This runs:
- Basic ML models (Random Forest, Logistic Regression)
- Enhanced models (XGBoost, LightGBM, CatBoost)
- Deep Learning (Focal Loss, Weighted BCE, Autoencoders)
- Graph Neural Networks
- Model Calibration
- Ensemble Methods
- Generates comprehensive report

### 2. Basic Pipeline Only
```bash
# Run just the basic fraud detection models
python fraud_detection_models.py
```
This will:
- Load and preprocess the data
- Train basic ML models
- Display performance metrics
- Save trained models

### 2. Enhanced Models with Deep Learning
```bash
# Run enhanced deep learning models
python enhanced_deep_learning.py
```
Features:
- Focal Loss for imbalanced data
- Weighted Binary Cross-Entropy
- Autoencoder for anomaly detection
- GPU acceleration (if available)

### 3. Interactive Dashboard
```bash
# Start the professional dashboard
python professional_fraud_dashboard.py
```
Then open http://localhost:8501 in your browser.

Dashboard features:
- Real-time fraud detection
- Model performance metrics
- Transaction analysis
- Pattern visualization

### 4. API Server
```bash
# Start the fraud detection API
python enhanced_fraud_api.py
```
API will be available at http://localhost:8000

Test the API:
```bash
# Health check
curl http://localhost:8000/health

# Make prediction (example)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.5, -1.2, 0.3, ...]}'
```

## Running Specific Components

### Data Exploration
```bash
python data_exploration.py
```
Generates comprehensive EDA report with visualizations.

### Advanced Models
```bash
# Graph Neural Networks
python graph_neural_network.py

# Ensemble System
python hybrid_ensemble_system.py

# Online Learning System
python online_streaming_system.py

# Model Calibration
python advanced_model_calibration.py
```

### Active Learning System
```bash
python enhanced_active_learning.py
```
Implements human-in-the-loop learning for continuous improvement.

## Jupyter Notebook Tutorials

Navigate to the tutorials folder and start Jupyter:
```bash
cd tutorials
jupyter notebook
```

Recommended order:
1. `data_exploration.ipynb` - Understanding the dataset
2. `fraud_detection_models.ipynb` - Basic ML models
3. `enhanced_fraud_models.ipynb` - Advanced techniques
4. `enhanced_deep_learning.ipynb` - Deep learning approaches
5. `heterogeneous_gnn.ipynb` - Graph neural networks
6. `professional_fraud_dashboard.ipynb` - Building dashboards
7. `enhanced_fraud_api.ipynb` - Creating APIs

## GPU Configuration

The system automatically detects and uses available GPUs:
```bash
# Test GPU configuration
python gpu_config.py
```

To force CPU usage:
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

## Performance Benchmarks

Expected performance (may vary based on hardware):
- Random Forest: F1-score ~0.85
- XGBoost: F1-score ~0.86
- Deep Learning: F1-score ~0.87
- Ensemble: F1-score ~0.88+

Training times (with GPU):
- Basic models: 5-10 minutes
- Deep learning: 15-30 minutes
- Full pipeline: 45-60 minutes

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce batch size in deep learning models
   - Use sampling for large datasets
   - Enable GPU memory optimization

2. **Missing Dependencies**
   ```bash
   pip install --upgrade -r requirements.txt
   ```

3. **GPU Not Detected**
   - Check CUDA installation: `nvidia-smi`
   - Verify PyTorch GPU: `python -c "import torch; print(torch.cuda.is_available())"`

4. **Dataset Not Found**
   - Ensure `creditcard.csv` is in the project root
   - Check file permissions

### Logging

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Production Deployment

### Docker Deployment
```bash
# Build Docker image
docker build -t fraud-detection .

# Run container
docker run -p 8000:8000 -p 8501:8501 fraud-detection
```

### Environment Variables
Create `.env` file:
```
MODEL_PATH=./models
DATA_PATH=./data
LOG_LEVEL=INFO
GPU_MEMORY_FRACTION=0.8
```

 📁 Python Files Breakdown

  ✅ Files You RUN (Main Programs)

  These have if __name__ == "__main__": and are complete programs:
  - fraud_detection_models.py - Main ML pipeline
  - enhanced_deep_learning.py - Deep learning models
  - professional_fraud_dashboard.py - Web dashboard
  - enhanced_fraud_api.py - API server
  - gpu_config.py - GPU testing tool

  📦 Files You DON'T RUN (Helper Modules)

  These are imported by other files:
  - data_preprocessing.py - Used by fraud_detection_models.py
  - data_exploration.py - Used for analysis functions
  - active_learning_system.py - Imported by enhanced_active_learning.py
  - setup.py - For package installation only

  🔄 Duplicate/Alternative Versions

  Earlier versions kept for reference:
  - streamlit_dashboard.py → Replaced by professional_fraud_dashboard.py
  - dash_dashboard.py → Alternative to Streamlit
  - fraud_detection_api.py → Replaced by enhanced_fraud_api.py
  - advanced_models.py → Replaced by enhanced_fraud_models.py
  - simplified_advanced_models.py → Simplified version

  🧩  Specialized Components

  Only run if you need specific features:
  - graph_neural_network.py - Advanced GNN model
  - heterogeneous_gnn.py - Another GNN variant
  - online_streaming_system.py - Real-time processing
  - hybrid_ensemble_system.py - Complex ensemble
  - advanced_model_calibration.py - Probability calibration
  - demo_script.py - Quick demonstration

  🎯 What You Actually Need

  Essential files to run (in order):
  1. fraud_detection_models.py - Basic models
  2. enhanced_deep_learning.py - Deep learning
  3. professional_fraud_dashboard.py - View results

  Everything else is either:
  - Helper code (imported automatically)
  - Old versions (kept for reference)
  - Specialized features (optional)


### Monitoring
The system includes built-in monitoring:
- API metrics: http://localhost:8000/metrics
- Model performance tracking
- Resource usage monitoring

## Additional Resources

- Project Documentation: See individual module docstrings
- API Documentation: http://localhost:8000/docs (when API is running)
- Model Architecture: Check tutorial notebooks
- Performance Optimization: See `gpu_config.py` for GPU settings

## Contact

For issues or questions:
- Author: Yanhong Simokat
- Email: yanhong7369@gmail.com