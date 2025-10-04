import torch
import torch.nn as nn
from pathlib import Path
import sys
import os

sys.path.append(str(Path(__file__).parent))

from models.unet import UNet
from utils.trainer import Trainer
from utils.visualizer import Visualizer
from data_loader import create_data_loaders

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    is_cuda = device.type == 'cuda'
    
    config = {
        'batch_size': 16 if is_cuda else 4,
        'learning_rate': 2e-4,
        'weight_decay': 1e-4,
        'num_epochs': 20,
        'input_channels': 3,
        'num_classes': 1,
        'device': device,
        'mixed_precision': is_cuda,
        'gradient_accumulation_steps': 1,
        'num_workers': min(8, os.cpu_count()) if is_cuda else 2,
        'warmup_epochs': 5,
        'cosine_annealing': True,
        'early_stopping_patience': 15,
    }
    
    print(f"Configuration:")
    for key, val in config.items():
        print(f"  {key}: {val}")
    
    if is_cuda:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir='Data/NLM-MontgomeryCXRSet/MontgomerySet',
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    
    model = UNet(
        n_channels=config['input_channels'],
        n_classes=config['num_classes']
    ).to(config['device'])
    
    if is_cuda and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode='reduce-overhead')
            print("Model compiled successfully")
        except Exception as e:
            print(f"Model compilation skipped: {e}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
    
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
    
    train_losses, val_losses = trainer.train(config['num_epochs'])
    
    visualizer = Visualizer(model, config['device'])
    visualizer.visualize_predictions(val_loader)
    visualizer.visualize_overlay(val_loader)
    
    print("\nTraining completed successfully!")
    print(f"Results saved to: {trainer.results_dir}")

if __name__ == "__main__":
    main()