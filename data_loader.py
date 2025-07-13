import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import time

class OptimizedLungDataset(Dataset):
    def __init__(self, image_dir, left_mask_dir, right_mask_dir, samples, image_transform=None, mask_transform=None, preload=False):
        self.image_dir = image_dir
        self.left_mask_dir = left_mask_dir
        self.right_mask_dir = right_mask_dir
        self.samples = samples
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.preload = preload
        self.cache = {}
        
        # Preload small datasets into memory
        if preload and len(samples) < 100:
            print(f"Preloading {len(samples)} samples...")
            self._preload_data()
    
    def _preload_data(self):
        """Preload data into memory for small datasets"""
        for i in range(len(self.samples)):
            self.cache[i] = self._load_sample(i)
    
    def _load_sample(self, idx):
        """Load a single sample from disk"""
        sample_id = self.samples[idx]
        
        # Load image
        image_path = os.path.join(self.image_dir, sample_id)
        image = Image.open(image_path).convert('RGB')
        
        # Load masks
        left_mask_path = os.path.join(self.left_mask_dir, sample_id)
        right_mask_path = os.path.join(self.right_mask_dir, sample_id)
        
        left_mask = Image.open(left_mask_path).convert('L')
        right_mask = Image.open(right_mask_path).convert('L')
        
        return image, left_mask, right_mask
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Check cache first
        if idx in self.cache:
            image, left_mask, right_mask = self.cache[idx]
        else:
            image, left_mask, right_mask = self._load_sample(idx)
        
        # Apply transforms separately
        if self.image_transform:
            image = self.image_transform(image)
        
        if self.mask_transform:
            left_mask = self.mask_transform(left_mask)
            right_mask = self.mask_transform(right_mask)
        
        # Combine masks
        combined_mask = torch.max(left_mask, right_mask)
        
        return image, combined_mask

def create_optimized_dataloader(dataset, batch_size=4, shuffle=True, is_train=True):
    """Create optimized DataLoader with multiprocessing"""
    # Use optimal number of workers
    num_workers = min(mp.cpu_count() - 1, 8)  # Leave 1 CPU free, max 8 workers
    
    # Reduce workers for small datasets to avoid overhead
    if len(dataset) < 50:
        num_workers = min(num_workers, 2)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),  # Faster GPU transfer
        persistent_workers=True if is_train and num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else 2,
        drop_last=True if is_train else False,
        multiprocessing_context='spawn' if mp.get_start_method() != 'spawn' else None
    )

def prepare_data_parallel():
    """Prepare data directories and file lists in parallel"""
    # Directories - updated paths
    base_dir = "Data/NLM-MontgomeryCXRSet/MontgomerySet"
    image_dir = os.path.join(base_dir, "CXR_png")
    mask_dir = os.path.join(base_dir, "ManualMask")
    
    # Check if directories exist
    for dir_path in [image_dir, mask_dir]:
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"Directory not found: {dir_path}")
    
    # Check what's in ManualMask directory
    print(f"Contents of {mask_dir}:")
    for item in os.listdir(mask_dir):
        print(f"  {item}")
    
    # Try different possible mask directory names
    possible_left_dirs = ["leftMask", "left", "LeftMask", "left_mask", "left lung"]
    possible_right_dirs = ["rightMask", "right", "RightMask", "right_mask", "right lung"]
    
    left_mask_dir = None
    right_mask_dir = None
    
    for left_dir in possible_left_dirs:
        if os.path.exists(os.path.join(mask_dir, left_dir)):
            left_mask_dir = os.path.join(mask_dir, left_dir)
            break
    
    for right_dir in possible_right_dirs:
        if os.path.exists(os.path.join(mask_dir, right_dir)):
            right_mask_dir = os.path.join(mask_dir, right_dir)
            break
    
    if not left_mask_dir or not right_mask_dir:
        raise FileNotFoundError(f"Could not find left/right mask directories in {mask_dir}")
    
    print(f"Found left masks: {left_mask_dir}")
    print(f"Found right masks: {right_mask_dir}")
    
    print("Checking file structure...")
    
    # Use ThreadPoolExecutor for I/O operations
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Submit tasks
        left_future = executor.submit(os.listdir, left_mask_dir)
        right_future = executor.submit(os.listdir, right_mask_dir)
        image_future = executor.submit(os.listdir, image_dir)
        
        # Get results
        left_files = left_future.result()
        right_files = right_future.result()
        image_files = image_future.result()
    
    # Filter and find common files
    left_files = [f for f in left_files if f.endswith('.png')]
    right_files = [f for f in right_files if f.endswith('.png')]
    image_files = [f for f in image_files if f.endswith('.png')]
    
    # Find intersection
    common_files = list(set(left_files) & set(right_files) & set(image_files))
    common_files.sort()
    
    print(f"Left mask files: {len(left_files)}")
    print(f"Right mask files: {len(right_files)}")
    print(f"Image files: {len(image_files)}")
    print(f"Common files: {len(common_files)}")
    
    return image_dir, left_mask_dir, right_mask_dir, common_files

def create_transforms():
    """Create optimized transforms for images and masks separately"""
    # Image transforms (RGB, 3 channels)
    image_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Mask transforms (Grayscale, 1 channel)
    mask_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    return image_transform, mask_transform

def visualize_samples_non_blocking(dataloader, max_samples=4):
    """Non-blocking visualization"""
    print("Visualizing samples...")
    
    # Get first batch
    batch = next(iter(dataloader))
    images, masks = batch
    
    # Plot without blocking
    fig, axes = plt.subplots(2, min(max_samples, len(images)), figsize=(15, 6))
    
    for i in range(min(max_samples, len(images))):
        # Original image
        img = images[i].permute(1, 2, 0)
        img = img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
        img = torch.clamp(img, 0, 1)
        
        axes[0, i].imshow(img)
        axes[0, i].set_title(f'Image {i+1}')
        axes[0, i].axis('off')
        
        # Mask
        axes[1, i].imshow(masks[i].squeeze(), cmap='gray')
        axes[1, i].set_title(f'Mask {i+1}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show(block=False)  # Non-blocking display
    plt.pause(0.001)  # Brief pause to render

def benchmark_dataloader(dataloader, name, num_batches=10):
    """Benchmark dataloader performance"""
    print(f"\nBenchmarking {name}...")
    
    start_time = time.time()
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
    
    end_time = time.time()
    avg_time = (end_time - start_time) / num_batches
    print(f"{name}: {avg_time:.3f}s per batch")
    
    return avg_time

def main():
    # Set multiprocessing start method
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)
    
    print("Starting optimized data loading...")
    start_total = time.time()
    
    # Prepare data in parallel
    image_dir, left_mask_dir, right_mask_dir, common_files = prepare_data_parallel()
    
    # Create transforms
    image_transform, mask_transform = create_transforms()
    
    # Split data
    train_files, temp_files = train_test_split(common_files, test_size=0.3, random_state=42)
    val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)
    
    print(f"Total samples: {len(common_files)}")
    print(f"Train samples: {len(train_files)}")
    print(f"Val samples: {len(val_files)}")
    print(f"Test samples: {len(test_files)}")
    
    # Create datasets
    train_dataset = OptimizedLungDataset(
        image_dir, left_mask_dir, right_mask_dir, 
        train_files, image_transform=image_transform, mask_transform=mask_transform, preload=len(train_files) < 50
    )
    
    val_dataset = OptimizedLungDataset(
        image_dir, left_mask_dir, right_mask_dir,
        val_files, image_transform=image_transform, mask_transform=mask_transform, preload=len(val_files) < 50
    )
    
    test_dataset = OptimizedLungDataset(
        image_dir, left_mask_dir, right_mask_dir,
        test_files, image_transform=image_transform, mask_transform=mask_transform, preload=len(test_files) < 50
    )
    
    # Create optimized dataloaders
    print("\nCreating optimized dataloaders...")
    train_loader = create_optimized_dataloader(train_dataset, batch_size=4, is_train=True)
    val_loader = create_optimized_dataloader(val_dataset, batch_size=4, shuffle=False, is_train=False)
    test_loader = create_optimized_dataloader(test_dataset, batch_size=1, shuffle=False, is_train=False)
    
    # Benchmark performance
    benchmark_dataloader(train_loader, "Training DataLoader")
    benchmark_dataloader(val_loader, "Validation DataLoader")
    
    # Quick visualization (non-blocking)
    visualize_samples_non_blocking(train_loader)
    
    end_total = time.time()
    print(f"\nTotal setup time: {end_total - start_total:.2f}s")
    print(f"Using {mp.cpu_count()} CPU cores")
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    train_loader, val_loader, test_loader = main()
    
    # Keep the program running briefly to see the plot
    print("\nPress Ctrl+C to exit...")
    try:
        time.sleep(5)
    except KeyboardInterrupt:
        print("Exiting...")