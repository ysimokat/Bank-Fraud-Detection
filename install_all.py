#!/usr/bin/env python3
"""
Quick Install Script for All Dependencies
=========================================

Installs all required packages for the fraud detection system.

Author: Yanhong Simokat
"""

import subprocess
import sys
import os

def run_command(cmd):
    """Run a command and return success status."""
    try:
        subprocess.check_call(cmd, shell=True)
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    print(">>> Fraud Detection System - Complete Installation")
    print("=" * 60)
    
    # Check if in virtual environment
    if sys.prefix == sys.base_prefix:
        print("WARNING:  Warning: Not in a virtual environment!")
        print("   Recommended: python -m venv venv && venv\\Scripts\\activate")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    print("\n[PACKAGE] Installing packages in groups...")
    
    # Core packages
    print("\n1. Installing core ML packages...")
    core_packages = [
        "numpy pandas scikit-learn matplotlib seaborn joblib",
        "imbalanced-learn tqdm"
    ]
    for pkg_group in core_packages:
        run_command(f"pip install {pkg_group}")
    
    # Advanced ML packages
    print("\n2. Installing advanced ML packages...")
    run_command("pip install xgboost lightgbm catboost")
    
    # Dashboard and API packages
    print("\n3. Installing dashboard and API packages...")
    run_command("pip install streamlit dash plotly")
    run_command("pip install fastapi uvicorn python-multipart pydantic")
    
    # Additional analysis packages
    print("\n4. Installing analysis packages...")
    run_command("pip install shap optuna")
    
    # Graph packages
    print("\n5. Installing graph packages...")
    run_command("pip install networkx")
    
    # Online learning
    print("\n6. Installing online learning packages...")
    run_command("pip install river")
    
    # Deep Learning - Ask for GPU support
    print("\n7. Deep Learning packages...")
    print("\n[SYSTEM]  GPU Detection:")
    
    # Try to detect NVIDIA GPU
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("[OK] NVIDIA GPU detected!")
            gpu_available = True
        else:
            print("[ERROR] No NVIDIA GPU detected")
            gpu_available = False
    except:
        print("[ERROR] nvidia-smi not found - assuming no GPU")
        gpu_available = False
    
    if gpu_available:
        print("\n[GPU] Installing PyTorch with GPU support (CUDA 12.1)...")
        run_command("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    else:
        print("\n[CPU] Installing PyTorch (CPU only)...")
        run_command("pip install torch torchvision torchaudio")
    
    # TensorFlow (optional)
    print("\n8. Installing TensorFlow (optional)...")
    response = input("Install TensorFlow? (y/n): ")
    if response.lower() == 'y':
        run_command("pip install tensorflow")
    
    # PyTorch Geometric (for GNN)
    print("\n9. Installing PyTorch Geometric...")
    if gpu_available:
        print("Installing with CUDA support...")
        run_command("pip install torch-geometric")
        # Additional dependencies for PyG
        run_command("pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu121.html")
    else:
        print("Installing CPU version...")
        run_command("pip install torch-geometric")
    
    print("\n" + "=" * 60)
    print("[OK] Installation complete!")
    print("\n[CONFIG] Next steps:")
    print("1. Run: python check_requirements.py")
    print("2. Run: python gpu_config.py")
    print("3. Run: python integrated_fraud_pipeline.py --quick")
    
    # Test imports
    print("\n[TEST] Testing critical imports...")
    test_imports = [
        "import pandas",
        "import numpy", 
        "import sklearn",
        "import xgboost",
        "import torch",
        "import streamlit"
    ]
    
    failed = []
    for imp in test_imports:
        try:
            exec(imp)
            print(f"[OK] {imp}")
        except ImportError:
            print(f"[ERROR] {imp}")
            failed.append(imp)
    
    if failed:
        print(f"\nWARNING:  Some imports failed: {failed}")
    else:
        print("\n[OK] All critical imports successful!")

if __name__ == "__main__":
    main()