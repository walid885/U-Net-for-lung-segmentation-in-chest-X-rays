import os
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class MontgomeryDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None, target_size=(256, 256)):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.target_size = target_size
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, self.target_size)
        image = image.astype(np.float32) / 255.0
        
        # Load mask
        mask_path = self.mask_paths[idx]
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, self.target_size)
        mask = (mask > 0).astype(np.float32)  # Binarize mask
        
        # Convert to tensors
        image = torch.from_numpy(image).unsqueeze(0)  # Add channel dimension
        mask = torch.from_numpy(mask).unsqueeze(0)
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        return image, mask

def load_montgomery_data(data_dir):
    """
    Load Montgomery County chest X-ray dataset
    
    Args:
        data_dir: Path to Data/NLM-MontgomeryCXRSet/MontgomerySet
    
    Returns:
        image_paths, mask_paths: Lists of file paths
    """
    image_dir = os.path.join(data_dir, "CXR_png")
    mask_dir = os.path.join(data_dir, "ManualMask")
    
    image_paths = []
    mask_paths = []
    
    # Get all image files
    for filename in os.listdir(image_dir):
        if filename.endswith('.png'):
            image_path = os.path.join(image_dir, filename)
            
            # Find corresponding mask files
            base_name = filename.replace('.png', '')
            
            # Look for left and right lung masks
            left_mask = os.path.join(mask_dir, f"{base_name}_left.png")
            right_mask = os.path.join(mask_dir, f"{base_name}_right.png")
            
            if os.path.exists(left_mask) and os.path.exists(right_mask):
                # Combine left and right masks
                combined_mask_path = combine_lung_masks(left_mask, right_mask, mask_dir, base_name)
                image_paths.append(image_path)
                mask_paths.append(combined_mask_path)
    
    return image_paths, mask_paths

def combine_lung_masks(left_mask_path, right_mask_path, output_dir, base_name):
    """
    Combine left and right lung masks into single mask
    """
    left_mask = cv2.imread(left_mask_path, cv2.IMREAD_GRAYSCALE)
    right_mask = cv2.imread(right_mask_path, cv2.IMREAD_GRAYSCALE)
    
    # Combine masks
    combined_mask = np.logical_or(left_mask > 0, right_mask > 0).astype(np.uint8) * 255
    
    # Save combined mask
    combined_path = os.path.join(output_dir, f"{base_name}_combined.png")
    cv2.imwrite(combined_path, combined_mask)
    
    return combined_path

def create_data_loaders(data_dir, batch_size=4, val_split=0.2, test_split=0.1):
    """
    Create training, validation, and test data loaders
    """
    # Load data paths
    image_paths, mask_paths = load_montgomery_data(data_dir)
    
    print(f"Total samples: {len(image_paths)}")
    
    # Split data
    train_imgs, temp_imgs, train_masks, temp_masks = train_test_split(
        image_paths, mask_paths, test_size=val_split+test_split, random_state=42
    )
    
    val_imgs, test_imgs, val_masks, test_masks = train_test_split(
        temp_imgs, temp_masks, test_size=test_split/(val_split+test_split), random_state=42
    )
    
    print(f"Train samples: {len(train_imgs)}")
    print(f"Val samples: {len(val_imgs)}")
    print(f"Test samples: {len(test_imgs)}")
    
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
    ])
    
    # Create datasets
    train_dataset = MontgomeryDataset(train_imgs, train_masks, transform=train_transform)
    val_dataset = MontgomeryDataset(val_imgs, val_masks)
    test_dataset = MontgomeryDataset(test_imgs, test_masks)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

def visualize_samples(data_loader, num_samples=4):
    """
    Visualize sample images and masks
    """
    dataiter = iter(data_loader)
    images, masks = next(dataiter)
    
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    
    for i in range(num_samples):
        # Display image
        axes[0, i].imshow(images[i].squeeze(), cmap='gray')
        axes[0, i].set_title(f'Image {i+1}')
        axes[0, i].axis('off')
        
        # Display mask
        axes[1, i].imshow(masks[i].squeeze(), cmap='gray')
        axes[1, i].set_title(f'Mask {i+1}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()

def get_dataset_statistics(data_loader):
    """
    Calculate dataset statistics
    """
    total_samples = 0
    total_pixels = 0
    positive_pixels = 0
    
    for images, masks in data_loader:
        total_samples += images.size(0)
        total_pixels += masks.numel()
        positive_pixels += masks.sum().item()
    
    positive_ratio = positive_pixels / total_pixels
    
    print(f"Dataset Statistics:")
    print(f"Total samples: {total_samples}")
    print(f"Total pixels: {total_pixels}")
    print(f"Positive pixels (lung): {positive_pixels}")
    print(f"Positive ratio: {positive_ratio:.4f}")
    
    return positive_ratio

if __name__ == "__main__":
    # Example usage
    data_dir = "Data/NLM-MontgomeryCXRSet/MontgomerySet"
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(data_dir)
    
    # Visualize samples
    print("Training samples:")
    visualize_samples(train_loader)
    
    # Get dataset statistics
    get_dataset_statistics(train_loader)