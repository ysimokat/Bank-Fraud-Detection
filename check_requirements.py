#!/usr/bin/env python3
"""
Check and Install Missing Requirements
======================================

This script checks for all required packages and helps install missing ones.

Author: Yanhong Simokat
"""

import subprocess
import sys
import importlib
import pkg_resources
from typing import Dict, List, Tuple

# Define all required packages with their import names and pip names
REQUIRED_PACKAGES = {
    # Core ML packages
    'numpy': 'numpy',
    'pandas': 'pandas',
    'scikit-learn': 'scikit-learn',
    'matplotlib': 'matplotlib',
    'seaborn': 'seaborn',
    'joblib': 'joblib',
    
    # Advanced ML
    'xgboost': 'xgboost',
    'lightgbm': 'lightgbm',
    'catboost': 'catboost',
    
    # Deep Learning
    'torch': 'torch',
    'torchvision': 'torchvision',
    'tensorflow': 'tensorflow',
    
    # Graph Neural Networks
    'torch_geometric': 'torch-geometric',
    'networkx': 'networkx',
    
    # Imbalanced learning
    'imblearn': 'imbalanced-learn',
    
    # Streaming/Online learning
    'river': 'river',
    
    # API and Dashboard
    'fastapi': 'fastapi',
    'uvicorn': 'uvicorn',
    'streamlit': 'streamlit',
    'dash': 'dash',
    'plotly': 'plotly',
    
    # Additional utilities
    'shap': 'shap',
    'optuna': 'optuna',
    'tqdm': 'tqdm',
    'python-multipart': 'python-multipart',
    'pydantic': 'pydantic',
}

# GPU-specific packages
GPU_PACKAGES = {
    'torch': 'torch --index-url https://download.pytorch.org/whl/cu121',
    'torchvision': 'torchvision --index-url https://download.pytorch.org/whl/cu121',
    'torchaudio': 'torchaudio --index-url https://download.pytorch.org/whl/cu121',
}

# Optional but recommended packages
OPTIONAL_PACKAGES = {
    'jupyterlab': 'jupyterlab',
    'notebook': 'notebook',
    'ipywidgets': 'ipywidgets',
    'pytest': 'pytest',
    'black': 'black',
    'flake8': 'flake8',
}

def check_package(package_name: str) -> Tuple[bool, str]:
    """Check if a package is installed and return version."""
    try:
        if package_name == 'sklearn':
            package_name = 'scikit-learn'
        
        # Try to import the package
        if package_name == 'scikit-learn':
            importlib.import_module('sklearn')
        elif package_name == 'torch-geometric':
            importlib.import_module('torch_geometric')
        elif package_name == 'imbalanced-learn':
            importlib.import_module('imblearn')
        else:
            importlib.import_module(package_name)
        
        # Get version
        try:
            version = pkg_resources.get_distribution(package_name).version
        except:
            version = "Unknown"
        
        return True, version
    except ImportError:
        return False, "Not installed"

def check_cuda_availability():
    """Check if CUDA is available for PyTorch."""
    try:
        import torch
        if torch.cuda.is_available():
            return True, f"CUDA {torch.version.cuda} - Device: {torch.cuda.get_device_name(0)}"
        else:
            return False, "PyTorch installed but CUDA not available"
    except ImportError:
        return False, "PyTorch not installed"

def install_package(package_name: str, pip_name: str = None) -> bool:
    """Install a package using pip."""
    if pip_name is None:
        pip_name = package_name
    
    try:
        print(f"[PACKAGE] Installing {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + pip_name.split())
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    """Main function to check and install requirements."""
    print("[SEARCH] Fraud Detection System - Requirements Checker")
    print("=" * 60)
    
    # Check Python version
    python_version = sys.version.split()[0]
    print(f"Python version: {python_version}")
    if sys.version_info < (3, 8):
        print("WARNING:  Warning: Python 3.8+ is recommended")
    print()
    
    # Check CUDA availability
    print("[SYSTEM]  GPU/CUDA Status:")
    cuda_available, cuda_info = check_cuda_availability()
    if cuda_available:
        print(f"[OK] {cuda_info}")
    else:
        print(f"[ERROR] {cuda_info}")
    print()
    
    # Check required packages
    print("[CONFIG] Required Packages:")
    print("-" * 60)
    
    missing_packages = []
    installed_packages = []
    
    for import_name, pip_name in REQUIRED_PACKAGES.items():
        installed, version = check_package(pip_name)
        if installed:
            print(f"[OK] {import_name:<20} {version}")
            installed_packages.append(import_name)
        else:
            print(f"[ERROR] {import_name:<20} Not installed")
            missing_packages.append((import_name, pip_name))
    
    print()
    
    # Summary
    print("[DATA] Summary:")
    print(f"   Installed: {len(installed_packages)}/{len(REQUIRED_PACKAGES)}")
    print(f"   Missing: {len(missing_packages)}")
    
    if missing_packages:
        print("\nWARNING:  Missing packages detected!")
        print("\n[PACKAGE] Installation Commands:")
        print("-" * 60)
        
        # Basic installation command
        print("\n1. Install all missing packages at once:")
        missing_pip_names = [pkg[1] for pkg in missing_packages]
        print(f"   pip install {' '.join(missing_pip_names)}")
        
        # GPU-specific installation
        if not cuda_available and 'torch' in [pkg[0] for pkg in missing_packages]:
            print("\n2. For GPU support (NVIDIA), install PyTorch with CUDA:")
            print("   pip uninstall torch torchvision torchaudio -y")
            print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        
        # Install missing packages interactively
        print("\n" + "=" * 60)
        response = input("\n[TIP] Would you like to install missing packages now? (y/n): ")
        
        if response.lower() == 'y':
            print("\n>>> Installing missing packages...")
            
            # Check if we need GPU version of PyTorch
            if not cuda_available and 'torch' in [pkg[0] for pkg in missing_packages]:
                gpu_response = input("Install PyTorch with GPU support? (y/n): ")
                if gpu_response.lower() == 'y':
                    # Remove torch packages from missing list
                    missing_packages = [(name, pip) for name, pip in missing_packages 
                                      if name not in ['torch', 'torchvision', 'torchaudio']]
                    
                    # Install GPU version
                    for name, cmd in GPU_PACKAGES.items():
                        install_package(name, cmd)
            
            # Install remaining packages
            for import_name, pip_name in missing_packages:
                success = install_package(import_name, pip_name)
                if success:
                    print(f"[OK] Successfully installed {import_name}")
                else:
                    print(f"[ERROR] Failed to install {import_name}")
            
            print("\n[OK] Installation complete!")
            print("[TARGET] You can now run: python advanced_integrated_pipeline.py")
        else:
            print("\n[NOTE] To install manually, use the commands shown above.")
    else:
        print("\n[OK] All required packages are installed!")
        print("[TARGET] You can run: python advanced_integrated_pipeline.py")
    
    # Check optional packages
    print("\n[PACKAGE] Optional Packages (for development):")
    print("-" * 60)
    for import_name, pip_name in OPTIONAL_PACKAGES.items():
        installed, version = check_package(pip_name)
        status = "[OK]" if installed else "[-]"
        print(f"{status} {import_name:<20} {version if installed else 'Not installed'}")
    
    # Create requirements file if it doesn't exist
    print("\n[FILE] Generating requirements.txt...")
    with open('requirements_full.txt', 'w') as f:
        f.write("# Core packages\n")
        for _, pip_name in REQUIRED_PACKAGES.items():
            if pip_name not in ['torch', 'torchvision', 'torchaudio']:
                f.write(f"{pip_name}\n")
        
        f.write("\n# PyTorch with CUDA (uncomment for GPU support)\n")
        f.write("# torch --index-url https://download.pytorch.org/whl/cu121\n")
        f.write("# torchvision --index-url https://download.pytorch.org/whl/cu121\n")
        f.write("# torchaudio --index-url https://download.pytorch.org/whl/cu121\n")
        
        f.write("\n# PyTorch CPU only (comment out if using GPU)\n")
        f.write("torch\n")
        f.write("torchvision\n")
        f.write("torchaudio\n")
    
    print("[OK] Created requirements_full.txt with all dependencies")

if __name__ == "__main__":
    main()