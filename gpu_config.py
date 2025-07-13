#!/usr/bin/env python3
"""
GPU Configuration and Setup Module
==================================

Handles GPU detection, configuration, and optimization for all models.
Supports CUDA (NVIDIA), ROCm (AMD), and MPS (Apple Silicon).

Author: Yanhong Simokat (yanhong7369@gmail.com)
"""

import torch
import numpy as np
import os
import logging
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPUConfig:
    """
    Unified GPU configuration for all models in the fraud detection system.
    """
    
    def __init__(self):
        self.device = None
        self.device_name = None
        self.gpu_available = False
        self.gpu_count = 0
        self.gpu_memory = {}
        self._setup_device()
        
    def _setup_device(self):
        """Detect and configure GPU device."""
        # Check for CUDA (NVIDIA GPUs)
        if torch.cuda.is_available():
            self.gpu_available = True
            self.gpu_count = torch.cuda.device_count()
            self.device = torch.device('cuda')
            self.device_name = torch.cuda.get_device_name(0)
            
            # Get GPU memory info
            for i in range(self.gpu_count):
                mem_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
                self.gpu_memory[f'GPU_{i}'] = {
                    'total_gb': round(mem_total, 2),
                    'reserved_gb': round(mem_reserved, 2)
                }
            
            logger.info(f"[OK] CUDA GPU detected: {self.device_name}")
            logger.info(f"   Number of GPUs: {self.gpu_count}")
            logger.info(f"   Memory: {self.gpu_memory}")
            
        # Check for MPS (Apple Silicon)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.gpu_available = True
            self.device = torch.device('mps')
            self.device_name = "Apple Silicon GPU (MPS)"
            self.gpu_count = 1
            logger.info(f"[OK] Apple Silicon GPU detected (MPS)")
            
        # Check for ROCm (AMD GPUs)
        elif hasattr(torch, 'hip') and torch.hip.is_available():
            self.gpu_available = True
            self.device = torch.device('hip')
            self.device_name = "AMD GPU (ROCm)"
            self.gpu_count = 1
            logger.info(f"[OK] AMD GPU detected (ROCm)")
            
        else:
            # Fallback to CPU
            self.device = torch.device('cpu')
            self.device_name = "CPU"
            logger.warning("WARNING: No GPU detected. Using CPU for computation.")
            logger.info("   For better performance, consider using a GPU.")
            
    def get_device(self):
        """Get the configured device."""
        return self.device
    
    def enable_mixed_precision(self):
        """Enable automatic mixed precision for faster training."""
        if self.gpu_available and self.device.type == 'cuda':
            # Enable TF32 on Ampere GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("[OK] Mixed precision training enabled (TF32)")
            return True
        return False
    
    def optimize_memory(self):
        """Optimize GPU memory usage."""
        if self.gpu_available and self.device.type == 'cuda':
            # Enable memory efficient attention
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # Clear cache
            torch.cuda.empty_cache()
            
            logger.info("[OK] GPU memory optimization enabled")
            return True
        return False
    
    def set_random_seeds(self, seed=42):
        """Set random seeds for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        if self.gpu_available:
            if self.device.type == 'cuda':
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            elif self.device.type == 'mps':
                # MPS doesn't have specific seed setting
                pass
                
        logger.info(f"[OK] Random seeds set to {seed}")
    
    def get_optimal_batch_size(self, model_type='standard'):
        """Get recommended batch size based on GPU memory."""
        if not self.gpu_available:
            return 32  # Conservative batch size for CPU
        
        # Get available memory (for CUDA GPUs)
        if self.device.type == 'cuda':
            mem_gb = self.gpu_memory.get('GPU_0', {}).get('total_gb', 4)
            
            # Recommendations based on model type and memory
            batch_sizes = {
                'standard': {
                    2: 16,   # 2GB GPU
                    4: 32,   # 4GB GPU
                    8: 64,   # 8GB GPU
                    12: 128, # 12GB GPU
                    16: 256, # 16GB GPU
                    24: 512  # 24GB+ GPU
                },
                'deep': {
                    2: 8,
                    4: 16,
                    8: 32,
                    12: 64,
                    16: 128,
                    24: 256
                },
                'autoencoder': {
                    2: 16,
                    4: 32,
                    8: 64,
                    12: 128,
                    16: 256,
                    24: 512
                }
            }
            
            # Find appropriate batch size
            sizes = batch_sizes.get(model_type, batch_sizes['standard'])
            for mem_threshold, batch_size in sorted(sizes.items()):
                if mem_gb >= mem_threshold:
                    recommended_size = batch_size
            
            return recommended_size
        
        # Default for other GPU types
        return 64
    
    def move_to_device(self, *tensors):
        """Move tensors to the configured device."""
        moved_tensors = []
        for tensor in tensors:
            if isinstance(tensor, torch.Tensor):
                moved_tensors.append(tensor.to(self.device))
            elif isinstance(tensor, np.ndarray):
                moved_tensors.append(torch.from_numpy(tensor).to(self.device))
            else:
                moved_tensors.append(tensor)
        
        return moved_tensors[0] if len(moved_tensors) == 1 else moved_tensors
    
    def get_info(self):
        """Get GPU configuration information."""
        info = {
            'device': str(self.device),
            'device_name': self.device_name,
            'gpu_available': self.gpu_available,
            'gpu_count': self.gpu_count,
            'gpu_memory': self.gpu_memory
        }
        return info
    
    def print_config(self):
        """Print GPU configuration details."""
        print("\n" + "="*60)
        print("[SYSTEM]  GPU Configuration")
        print("="*60)
        print(f"Device: {self.device}")
        print(f"Device Name: {self.device_name}")
        print(f"GPU Available: {self.gpu_available}")
        print(f"GPU Count: {self.gpu_count}")
        
        if self.gpu_memory:
            print("\nGPU Memory:")
            for gpu_id, mem_info in self.gpu_memory.items():
                print(f"  {gpu_id}: {mem_info['total_gb']} GB total")
        
        print("="*60 + "\n")

# Global GPU configuration instance
gpu_config = GPUConfig()

def get_device():
    """Get the configured device."""
    return gpu_config.get_device()

def setup_gpu_for_sklearn():
    """Configure scikit-learn models to use GPU when available."""
    # Set environment variables for scikit-learn GPU support
    if gpu_config.gpu_available:
        # For Intel GPU support in scikit-learn
        os.environ['SKLEARN_DEVICE'] = 'gpu'
        
        # For RAPIDS cuML (NVIDIA GPU acceleration for scikit-learn)
        try:
            import cuml
            logger.info("[OK] RAPIDS cuML available for GPU-accelerated scikit-learn")
            return True
        except ImportError:
            logger.info("ℹ️ Install RAPIDS cuML for GPU-accelerated scikit-learn algorithms")
            
    return False

def get_xgboost_params():
    """Get XGBoost parameters optimized for GPU."""
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'random_state': 42
    }
    
    if gpu_config.gpu_available and gpu_config.device.type == 'cuda':
        params.update({
            'tree_method': 'gpu_hist',
            'predictor': 'gpu_predictor',
            'gpu_id': 0
        })
        logger.info("[OK] XGBoost GPU acceleration enabled")
    else:
        params['tree_method'] = 'hist'
        
    return params

def get_lightgbm_params():
    """Get LightGBM parameters optimized for GPU."""
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'random_state': 42,
        'verbose': -1
    }
    
    if gpu_config.gpu_available and gpu_config.device.type == 'cuda':
        params.update({
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0
        })
        logger.info("[OK] LightGBM GPU acceleration enabled")
    else:
        params['device'] = 'cpu'
        
    return params

# Print configuration on import
if __name__ == "__main__":
    gpu_config.print_config()
    
    # Test GPU functionality
    if gpu_config.gpu_available:
        print("\n[TEST] Testing GPU functionality...")
        try:
            # Create a small tensor and perform operation
            test_tensor = torch.randn(1000, 1000).to(gpu_config.device)
            result = torch.matmul(test_tensor, test_tensor)
            print("[OK] GPU computation test successful!")
            
            # Memory test
            if gpu_config.device.type == 'cuda':
                print(f"   Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"[ERROR] GPU test failed: {e}")