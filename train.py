import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models.unet import UNet
from utils.trainer import Trainer
from utils.visualizer import Visualizer
from data_loader import create_data_loaders

def main():
    # Detect device first
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    is_cuda = device.type == 'cuda'
    
    # Optimized Configuration
    config = {
        'batch_size': 32,
        'learning_rate': 2e-4,
        'weight_decay': 1e-4,
        'num_epochs': 100,
        'input_channels': 3,
        'num_classes': 1,
        'device': device,
        
        # Performance optimizations (adjusted for CPU/GPU)
        'compile_model': is_cuda,  # Only compile on GPU
        'mixed_precision': is_cuda,  # Only use AMP on GPU
        'gradient_accumulation_steps': 1,
        'num_workers': min(8, os.cpu_count()) if is_cuda else 4,  # Fewer workers on CPU
        'pin_memory': is_cuda,  # Only pin memory for GPU
        'persistent_workers': is_cuda,
        'prefetch_factor': 2 if is_cuda else 1,
        
        # Training optimizations
        'warmup_epochs': 5,
        'cosine_annealing': True,
        'early_stopping_patience': 15,
        'save_best_only': True,
    }
    
    print(f"Using device: {config['device']}")
    
    # Performance optimizations (only for CUDA)
    if is_cuda:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Create optimized data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir='Data/NLM-MontgomeryCXRSet/MontgomerySet',
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    
    # Create model with optimizations
    model = UNet(
        n_channels=config['input_channels'],
        n_classes=config['num_classes']
    ).to(config['device'])
    
    # Compile model for PyTorch 2.0+ (only on GPU)
    if config['compile_model'] and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode='reduce-overhead')
            print("✓ Model compiled for optimized execution")
        except Exception as e:
            print(f"Model compilation failed: {e}")
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create optimized trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=config['device'],
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        mixed_precision=config['mixed_precision'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        warmup_epochs=config['warmup_epochs'],
        cosine_annealing=config['cosine_annealing'],
        early_stopping_patience=config['early_stopping_patience']
    )
    
    # Train model
    trainer.train(config['num_epochs'])
    
    # Visualize results (optional - comment out for faster execution)
    visualizer = Visualizer(model, config['device'])
    visualizer.visualize_predictions(val_loader)
    visualizer.visualize_overlay(val_loader)
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()