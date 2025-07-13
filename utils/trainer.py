# ============================================================================
#  utils/trainer.py
# ============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import json

class Trainer:
    def __init__(self, model, train_loader, val_loader, device, 
                 learning_rate=1e-4, weight_decay=1e-4, results_dir='results'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Import losses and metrics
        from models.losses import CombinedLoss
        from utils.metrics import dice_coefficient, iou_score, pixel_accuracy
        
        self.criterion = CombinedLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, factor=0.5)
        
        # Metrics functions
        self.dice_coefficient = dice_coefficient
        self.iou_score = iou_score
        self.pixel_accuracy = pixel_accuracy
        
        # History
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_dice': [], 'val_dice': [],
            'train_iou': [], 'val_iou': [],
            'train_acc': [], 'val_acc': []
        }
        
        self.best_val_dice = 0.0
        self.best_model_path = self.results_dir / 'best_model.pth'
    
    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        running_dice = 0.0
        running_iou = 0.0
        running_acc = 0.0
        
        for batch_idx, (images, masks) in enumerate(tqdm(self.train_loader, desc="Training")):
            images, masks = images.to(self.device), masks.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            running_dice += self.dice_coefficient(outputs, masks)
            running_iou += self.iou_score(outputs, masks)
            running_acc += self.pixel_accuracy(outputs, masks)
        
        num_batches = len(self.train_loader)
        return {
            'loss': running_loss / num_batches,
            'dice': running_dice / num_batches,
            'iou': running_iou / num_batches,
            'acc': running_acc / num_batches
        }
    
    def validate_epoch(self):
        self.model.eval()
        running_loss = 0.0
        running_dice = 0.0
        running_iou = 0.0
        running_acc = 0.0
        
        with torch.no_grad():
            for images, masks in tqdm(self.val_loader, desc="Validation"):
                images, masks = images.to(self.device), masks.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                running_loss += loss.item()
                running_dice += self.dice_coefficient(outputs, masks)
                running_iou += self.iou_score(outputs, masks)
                running_acc += self.pixel_accuracy(outputs, masks)
        
        num_batches = len(self.val_loader)
        return {
            'loss': running_loss / num_batches,
            'dice': running_dice / num_batches,
            'iou': running_iou / num_batches,
            'acc': running_acc / num_batches
        }
    
    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'best_val_dice': self.best_val_dice
        }
        
        if is_best:
            torch.save(checkpoint, self.best_model_path)
        
        torch.save(checkpoint, self.results_dir / f'checkpoint_epoch_{epoch}.pth')
    
    def plot_metrics(self):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss
        ax1.plot(epochs, self.history['train_loss'], 'b-', label='Training Loss')
        ax1.plot(epochs, self.history['val_loss'], 'r-', label='Validation Loss')
        ax1.set_title('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Dice
        ax2.plot(epochs, self.history['train_dice'], 'b-', label='Training Dice')
        ax2.plot(epochs, self.history['val_dice'], 'r-', label='Validation Dice')
        ax2.set_title('Dice Coefficient')
        ax2.legend()
        ax2.grid(True)
        
        # IoU
        ax3.plot(epochs, self.history['train_iou'], 'b-', label='Training IoU')
        ax3.plot(epochs, self.history['val_iou'], 'r-', label='Validation IoU')
        ax3.set_title('IoU Score')
        ax3.legend()
        ax3.grid(True)
        
        # Accuracy
        ax4.plot(epochs, self.history['train_acc'], 'b-', label='Training Accuracy')
        ax4.plot(epochs, self.history['val_acc'], 'r-', label='Validation Accuracy')
        ax4.set_title('Pixel Accuracy')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'training_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def train(self, num_epochs):
        print(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 20)
            
            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = self.validate_epoch()
            
            # Scheduler step
            self.scheduler.step(val_metrics['loss'])
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_dice'].append(train_metrics['dice'])
            self.history['val_dice'].append(val_metrics['dice'])
            self.history['train_iou'].append(train_metrics['iou'])
            self.history['val_iou'].append(val_metrics['iou'])
            self.history['train_acc'].append(train_metrics['acc'])
            self.history['val_acc'].append(val_metrics['acc'])
            
            # Print metrics
            print(f"Train - Loss: {train_metrics['loss']:.4f}, Dice: {train_metrics['dice']:.4f}, IoU: {train_metrics['iou']:.4f}, Acc: {train_metrics['acc']:.4f}")
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, Dice: {val_metrics['dice']:.4f}, IoU: {val_metrics['iou']:.4f}, Acc: {val_metrics['acc']:.4f}")
            
            # Save best model
            is_best = val_metrics['dice'] > self.best_val_dice
            if is_best:
                self.best_val_dice = val_metrics['dice']
                print(f"New best model! Validation Dice: {self.best_val_dice:.4f}")
            
            # Save checkpoint
            if epoch % 10 == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            
            # Plot metrics every 10 epochs
            if epoch % 10 == 0:
                self.plot_metrics()
        
        # Final plots and save history
        self.plot_metrics()
        with open(self.results_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"\nTraining completed! Best validation Dice: {self.best_val_dice:.4f}")


