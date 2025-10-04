import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import multiprocessing as mp

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
        
        if preload and len(samples) < 100:
            print(f"Preloading {len(samples)} samples...")
            self._preload_data()
    
    def _preload_data(self):
        for i in range(len(self.samples)):
            self.cache[i] = self._load_sample(i)
    
    def _load_sample(self, idx):
        sample_id = self.samples[idx]
        
        image_path = os.path.join(self.image_dir, sample_id)
        image = Image.open(image_path).convert('RGB')
        
        left_mask_path = os.path.join(self.left_mask_dir, sample_id)
        right_mask_path = os.path.join(self.right_mask_dir, sample_id)
        
        left_mask = Image.open(left_mask_path).convert('L')
        right_mask = Image.open(right_mask_path).convert('L')
        
        return image, left_mask, right_mask
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        if idx in self.cache:
            image, left_mask, right_mask = self.cache[idx]
        else:
            image, left_mask, right_mask = self._load_sample(idx)
        
        if self.image_transform:
            image = self.image_transform(image)
        
        if self.mask_transform:
            left_mask = self.mask_transform(left_mask)
            right_mask = self.mask_transform(right_mask)
        
        combined_mask = torch.max(left_mask, right_mask)
        
        return image, combined_mask

def create_optimized_dataloader(dataset, batch_size=4, shuffle=True, is_train=True):
    num_workers = min(mp.cpu_count() // 2, 4)
    
    if len(dataset) < 50:
        num_workers = min(num_workers, 2)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True if is_train and num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else 2,
        drop_last=True if is_train else False
    )

def create_transforms():
    image_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    mask_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    return image_transform, mask_transform

def create_data_loaders(data_dir, batch_size=4, num_workers=4):
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)
    
    base_dir = data_dir
    image_dir = os.path.join(base_dir, "CXR_png")
    mask_dir = os.path.join(base_dir, "ManualMask")
    
    for dir_path in [image_dir, mask_dir]:
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"Directory not found: {dir_path}")
    
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
        raise FileNotFoundError(f"Could not find mask directories in {mask_dir}")
    
    left_files = [f for f in os.listdir(left_mask_dir) if f.endswith('.png')]
    right_files = [f for f in os.listdir(right_mask_dir) if f.endswith('.png')]
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
    
    common_files = list(set(left_files) & set(right_files) & set(image_files))
    common_files.sort()
    
    image_transform, mask_transform = create_transforms()
    
    train_files, temp_files = train_test_split(common_files, test_size=0.3, random_state=42)
    val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)
    
    train_dataset = OptimizedLungDataset(
        image_dir, left_mask_dir, right_mask_dir, 
        train_files, image_transform=image_transform, mask_transform=mask_transform, 
        preload=len(train_files) < 50
    )
    
    val_dataset = OptimizedLungDataset(
        image_dir, left_mask_dir, right_mask_dir,
        val_files, image_transform=image_transform, mask_transform=mask_transform, 
        preload=len(val_files) < 50
    )
    
    test_dataset = OptimizedLungDataset(
        image_dir, left_mask_dir, right_mask_dir,
        test_files, image_transform=image_transform, mask_transform=mask_transform, 
        preload=len(test_files) < 50
    )
    
    train_loader = create_optimized_dataloader(train_dataset, batch_size=batch_size, is_train=True)
    val_loader = create_optimized_dataloader(val_dataset, batch_size=batch_size, shuffle=False, is_train=False)
    test_loader = create_optimized_dataloader(test_dataset, batch_size=1, shuffle=False, is_train=False)
    
    return train_loader, val_loader, test_loader