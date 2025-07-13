import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class LungDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None, target_size=(256, 256)):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.target_size = target_size
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, self.target_size)
        image = image.astype(np.float32) / 255.0
        
        # Load mask
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, self.target_size)
        mask = (mask > 127).astype(np.float32)  # Binary threshold
        
        # Convert to tensor
        image = torch.from_numpy(image).unsqueeze(0)  # Add channel dimension
        mask = torch.from_numpy(mask).unsqueeze(0)
        
        if self.transform:
            # Apply same transform to both image and mask
            seed = torch.randint(0, 2147483647, (1,)).item()
            torch.manual_seed(seed)
            image = self.transform(image)
            torch.manual_seed(seed)
            mask = self.transform(mask)
            
        return image, mask

def get_data_paths(data_dir):
    """Get image and mask file paths from Montgomery dataset"""
    image_dir = os.path.join(data_dir, 'CXR_png')
    mask_dir = os.path.join(data_dir, 'ManualMask', 'GT')
    
    image_paths = []
    mask_paths = []
    
    for filename in os.listdir(image_dir):
        if filename.endswith('.png'):
            image_path = os.path.join(image_dir, filename)
            # Montgomery masks have specific naming convention
            mask_filename = filename.replace('.png', '_mask.png')
            mask_path = os.path.join(mask_dir, mask_filename)
            
            if os.path.exists(mask_path):
                image_paths.append(image_path)
                mask_paths.append(mask_path)
    
    return image_paths, mask_paths

def create_data_loaders(data_dir, batch_size=16, test_size=0.3, val_size=0.5):
    """Create train, validation, and test data loaders"""
    
    # Get file paths
    image_paths, mask_paths = get_data_paths(data_dir)
    print(f"Found {len(image_paths)} image-mask pairs")
    
    # Split data
    train_images, temp_images, train_masks, temp_masks = train_test_split(
        image_paths, mask_paths, test_size=test_size, random_state=42
    )
    
    val_images, test_images, val_masks, test_masks = train_test_split(
        temp_images, temp_masks, test_size=val_size, random_state=42
    )
    
    print(f"Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}")
    
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2)
    ])
    
    # No augmentation for validation and test
    val_test_transform = None
    
    # Create datasets
    train_dataset = LungDataset(train_images, train_masks, transform=train_transform)
    val_dataset = LungDataset(val_images, val_masks, transform=val_test_transform)
    test_dataset = LungDataset(test_images, test_masks, transform=val_test_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader

def visualize_sample(data_loader, num_samples=4):
    """Visualize sample images and masks"""
    data_iter = iter(data_loader)
    images, masks = next(data_iter)
    
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    
    for i in range(num_samples):
        # Original image
        axes[0, i].imshow(images[i].squeeze(), cmap='gray')
        axes[0, i].set_title(f'Image {i+1}')
        axes[0, i].axis('off')
        
        # Mask
        axes[1, i].imshow(masks[i].squeeze(), cmap='gray')
        axes[1, i].set_title(f'Mask {i+1}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()

def check_dataset_info(data_dir):
    """Check dataset statistics"""
    image_paths, mask_paths = get_data_paths(data_dir)
    
    print(f"Dataset Statistics:")
    print(f"Total samples: {len(image_paths)}")
    
    # Check image dimensions
    sample_image = cv2.imread(image_paths[0], cv2.IMREAD_GRAYSCALE)
    print(f"Original image shape: {sample_image.shape}")
    
    # Check mask statistics
    sample_mask = cv2.imread(mask_paths[0], cv2.IMREAD_GRAYSCALE)
    print(f"Original mask shape: {sample_mask.shape}")
    print(f"Mask unique values: {np.unique(sample_mask)}")
    
    return len(image_paths)

if __name__ == "__main__":
    # Test the data loader
    data_dir = "data/Montgomery"  # Adjust path as needed
    
    # Check dataset info
    check_dataset_info(data_dir)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(data_dir, batch_size=8)
    
    # Visualize samples
    print("\nVisualizing training samples:")
    visualize_sample(train_loader)