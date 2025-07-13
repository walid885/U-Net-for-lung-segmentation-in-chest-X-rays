# ============================================================================
# inference.py 
# ============================================================================

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from models.unet import UNet
from utils.visualizer import Visualizer

class LungSegmentationInference:
    def __init__(self, model_path, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = UNet(n_channels=3, n_classes=1)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded from {model_path}")
        print(f"Best validation Dice: {checkpoint.get('best_val_dice', 'N/A')}")
    
    def preprocess_image(self, image_path, target_size=(256, 256)):
        """Preprocess image for inference"""
        image = Image.open(image_path).convert('RGB')
        image = image.resize(target_size)
        
        # Convert to tensor and normalize
        image_array = np.array(image) / 255.0
        image_tensor = torch.from_numpy(image_array).float()
        image_tensor = image_tensor.permute(2, 0, 1)  # HWC to CHW
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        
        return image_tensor
    
    def predict(self, image_path, threshold=0.5):
        """Predict lung segmentation mask"""
        # Preprocess
        image_tensor = self.preprocess_image(image_path)
        image_tensor = image_tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            output = self.model(image_tensor)
            prediction = torch.sigmoid(output)
            binary_mask = (prediction > threshold).float()
        
        return {
            'probability_mask': prediction.cpu().numpy().squeeze(),
            'binary_mask': binary_mask.cpu().numpy().squeeze(),
            'original_image': image_tensor.cpu().numpy().squeeze().transpose(1, 2, 0)
        }
    
    def visualize_result(self, image_path, save_path=None):
        """Visualize segmentation result"""
        result = self.predict(image_path)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(result['original_image'])
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Probability mask
        axes[1].imshow(result['probability_mask'], cmap='hot')
        axes[1].set_title('Probability Mask')
        axes[1].axis('off')
        
        # Binary mask overlay
        axes[2].imshow(result['original_image'], cmap='gray')
        axes[2].imshow(result['binary_mask'], alpha=0.3, cmap='Reds')
        axes[2].set_title('Segmentation Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

# Usage example
if __name__ == "__main__":
    # Initialize inference
    inference = LungSegmentationInference('results/best_model.pth')
    
    # Run inference on a test image
    test_image_path = 'Data/NLM-MontgomeryCXRSet/MontgomerySet/CXR_png/MCUCXR_0001_0.png'
    inference.visualize_result(test_image_path, 'results/inference_result.png')