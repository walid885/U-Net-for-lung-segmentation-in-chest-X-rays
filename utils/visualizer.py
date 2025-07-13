# ============================================================================
#  utils/visualizer.py
# ============================================================================

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

class Visualizer:
    def __init__(self, model, device, results_dir='results'):
        self.model = model
        self.device = device
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
    
    def visualize_predictions(self, data_loader, num_samples=8):
        self.model.eval()
        
        fig, axes = plt.subplots(3, num_samples, figsize=(20, 8))
        
        with torch.no_grad():
            for i, (images, masks) in enumerate(data_loader):
                if i >= num_samples:
                    break
                
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = self.model(images)
                predictions = torch.sigmoid(outputs)
                predictions = (predictions > 0.5).float()
                
                # Move to CPU for visualization
                image = images[0].cpu().numpy().transpose(1, 2, 0)
                mask = masks[0].cpu().numpy().squeeze()
                prediction = predictions[0].cpu().numpy().squeeze()
                
                # Normalize image for display
                image = (image - image.min()) / (image.max() - image.min())
                
                # Original image
                axes[0, i].imshow(image, cmap='gray' if image.shape[2] == 1 else None)
                axes[0, i].set_title('Original')
                axes[0, i].axis('off')
                
                # Ground truth
                axes[1, i].imshow(mask, cmap='gray')
                axes[1, i].set_title('Ground Truth')
                axes[1, i].axis('off')
                
                # Prediction
                axes[2, i].imshow(prediction, cmap='gray')
                axes[2, i].set_title('Prediction')
                axes[2, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'predictions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def visualize_overlay(self, data_loader, num_samples=4):
        self.model.eval()
        
        fig, axes = plt.subplots(2, num_samples, figsize=(16, 8))
        
        with torch.no_grad():
            for i, (images, masks) in enumerate(data_loader):
                if i >= num_samples:
                    break
                
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = self.model(images)
                predictions = torch.sigmoid(outputs)
                predictions = (predictions > 0.5).float()
                
                # Move to CPU
                image = images[0].cpu().numpy().transpose(1, 2, 0)
                mask = masks[0].cpu().numpy().squeeze()
                prediction = predictions[0].cpu().numpy().squeeze()
                
                # Normalize image
                image = (image - image.min()) / (image.max() - image.min())
                if image.shape[2] == 1:
                    image = image.squeeze()
                
                # Ground truth overlay
                axes[0, i].imshow(image, cmap='gray')
                axes[0, i].imshow(mask, alpha=0.3, cmap='Reds')
                axes[0, i].set_title('GT Overlay')
                axes[0, i].axis('off')
                
                # Prediction overlay
                axes[1, i].imshow(image, cmap='gray')
                axes[1, i].imshow(prediction, alpha=0.3, cmap='Blues')
                axes[1, i].set_title('Pred Overlay')
                axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'overlays.png', dpi=300, bbox_inches='tight')
        plt.close()


