# ============================================================================
#  train.py 
# ============================================================================

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models.unet import UNet
from utils.trainer import Trainer
from utils.visualizer import Visualizer
from data_loader import create_data_loaders  # Assuming this exists from your setup

def main():
    # Configuration
    config = {
        'batch_size': 16,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'num_epochs': 100,
        'input_channels': 3,  # RGB images
        'num_classes': 1,     # Binary segmentation
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    }
    
    print(f"Using device: {config['device']}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir='Data/NLM-MontgomeryCXRSet/MontgomerySet',
        batch_size=config['batch_size'],
        num_workers=4
    )
    
    # Create model
    model = UNet(
        n_channels=config['input_channels'],
        n_classes=config['num_classes']
    ).to(config['device'])
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=config['device'],
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Train model
    trainer.train(config['num_epochs'])
    
    # Visualize results
    visualizer = Visualizer(model, config['device'])
    visualizer.visualize_predictions(val_loader)
    visualizer.visualize_overlay(val_loader)
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()


