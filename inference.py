import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import time

sys.path.append(str(Path(__file__).parent))

from models.unet import UNet
from utils.metrics import dice_coefficient, iou_score, pixel_accuracy

class LungSegmentationInference:
    def __init__(self, model_path, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = UNet(n_channels=3, n_classes=1)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded from {model_path}")
        print(f"Device: {self.device}")
        if 'best_val_dice' in checkpoint:
            print(f"Best validation Dice: {checkpoint['best_val_dice']:.4f}")
        if 'metrics' in checkpoint:
            print(f"Validation metrics: {checkpoint['metrics']}")
    
    def preprocess_image(self, image_path, target_size=(256, 256)):
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        image = image.resize(target_size)
        
        image_array = np.array(image) / 255.0
        
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_array = (image_array - mean) / std
        
        image_tensor = torch.from_numpy(image_array).float()
        image_tensor = image_tensor.permute(2, 0, 1)
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor, original_size
    
    def predict(self, image_path, threshold=0.5):
        start_time = time.time()
        
        image_tensor, original_size = self.preprocess_image(image_path)
        image_tensor = image_tensor.to(self.device)
        
        with torch.no_grad():
            output = self.model(image_tensor)
            prediction = torch.sigmoid(output)
            binary_mask = (prediction > threshold).float()
        
        inference_time = time.time() - start_time
        
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        original_image = image_tensor.cpu() * std + mean
        original_image = torch.clamp(original_image, 0, 1)
        
        return {
            'probability_mask': prediction.cpu().numpy().squeeze(),
            'binary_mask': binary_mask.cpu().numpy().squeeze(),
            'original_image': original_image.numpy().squeeze().transpose(1, 2, 0),
            'inference_time': inference_time,
            'original_size': original_size
        }
    
    def visualize_result(self, image_path, save_path=None):
        result = self.predict(image_path)
        
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        axes[0].imshow(result['original_image'])
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(result['probability_mask'], cmap='hot')
        axes[1].set_title('Probability Map')
        axes[1].axis('off')
        
        axes[2].imshow(result['binary_mask'], cmap='gray')
        axes[2].set_title('Binary Mask')
        axes[2].axis('off')
        
        axes[3].imshow(result['original_image'])
        axes[3].imshow(result['binary_mask'], alpha=0.4, cmap='Reds')
        axes[3].set_title(f'Overlay (Time: {result["inference_time"]*1000:.2f}ms)')
        axes[3].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Result saved to {save_path}")
        plt.show()
        
        return result
    
    def benchmark_inference(self, image_path, num_runs=100):
        print(f"\nBenchmarking inference on {num_runs} runs...")
        
        image_tensor, _ = self.preprocess_image(image_path)
        image_tensor = image_tensor.to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = self.model(image_tensor)
        
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.time()
                _ = self.model(image_tensor)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                times.append(time.time() - start)
        
        times = np.array(times) * 1000  # Convert to ms
        
        print(f"Inference Benchmark Results:")
        print(f"  Mean: {np.mean(times):.2f}ms")
        print(f"  Median: {np.median(times):.2f}ms")
        print(f"  Std: {np.std(times):.2f}ms")
        print(f"  Min: {np.min(times):.2f}ms")
        print(f"  Max: {np.max(times):.2f}ms")
        print(f"  FPS: {1000/np.mean(times):.2f}")
        
        return times

if __name__ == "__main__":
    inference = LungSegmentationInference('results/best_model.pth')
    
    test_image_path = 'Data/NLM-MontgomeryCXRSet/MontgomerySet/CXR_png/MCUCXR_0001_0.png'
    
    result = inference.visualize_result(test_image_path, 'results/inference_result.png')
    
    inference.benchmark_inference(test_image_path, num_runs=100)