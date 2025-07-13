import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import GradScaler, autocast
import time
import numpy as np
from pathlib import Path

class Trainer:
    def __init__(self, model, train_loader, val_loader, device, learning_rate=1e-4, 
                 weight_decay=1e-4, mixed_precision=True, gradient_accumulation_steps=1,
                 warmup_epochs=5, cosine_annealing=True, early_stopping_patience=15):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.mixed_precision = mixed_precision
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.early_stopping_patience = early_stopping_patience
        
        # Optimizer with optimized settings
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay,
            eps=1e-8,
            foreach=True  # Faster multi-tensor operations
        )
        
        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Mixed precision scaler
        self.scaler = GradScaler() if mixed_precision else None
        
        # Learning rate scheduler
        if cosine_annealing:
            warmup_scheduler = LinearLR(
                self.optimizer, 
                start_factor=0.1, 
                end_factor=1.0, 
                total_iters=warmup_epochs
            )
            cosine_scheduler = CosineAnnealingLR(
                self.optimizer, 
                T_max=100 - warmup_epochs,
                eta_min=learning_rate * 0.01
            )
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs]
            )
        else:
            self.scheduler = None
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        
        # Create results directory
        self.results_dir = Path('results')
        self.results_dir.mkdir(exist_ok=True)
        
        print(f"Trainer initialized with mixed_precision={mixed_precision}")
        print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
        print(f"Using scheduler: {cosine_annealing}")
    
    def train_epoch(self):
        """Optimized training epoch with mixed precision and gradient accumulation"""
        self.model.train()
        epoch_loss = 0
        num_batches = len(self.train_loader)
        
        self.optimizer.zero_grad()
        
        for batch_idx, (images, masks) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)
            
            # Mixed precision forward pass
            if self.mixed_precision:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                    loss = loss / self.gradient_accumulation_steps
                
                self.scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                loss = loss / self.gradient_accumulation_steps
                
                loss.backward()
                
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            epoch_loss += loss.item() * self.gradient_accumulation_steps
            
            # Progress update (every 10% of batches)
            if batch_idx % max(1, num_batches // 10) == 0:
                print(f'Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}')
        
        # Handle remaining gradients
        if num_batches % self.gradient_accumulation_steps != 0:
            if self.mixed_precision:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()
        
        return epoch_loss / num_batches
    
    def validate(self):
        """Optimized validation with mixed precision"""
        self.model.eval()
        val_loss = 0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for images, masks in self.val_loader:
                images = images.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)
                
                if self.mixed_precision:
                    with autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, masks)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                
                val_loss += loss.item()
        
        return val_loss / num_batches
    
    def train(self, num_epochs):
        """Main training loop with all optimizations"""
        print(f"Starting training for {num_epochs} epochs...")
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Training
            train_loss = self.train_epoch()
            
            # Validation
            val_loss = self.validate()
            
            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step()
            
            # Track metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, self.results_dir / 'best_model.pth')
            else:
                self.patience_counter += 1
            
            epoch_time = time.time() - epoch_start
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'  Train Loss: {train_loss:.4f}')
            print(f'  Val Loss: {val_loss:.4f}')
            print(f'  LR: {current_lr:.2e}')
            print(f'  Time: {epoch_time:.2f}s')
            print(f'  Patience: {self.patience_counter}/{self.early_stopping_patience}')
            print('-' * 50)
            
            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f} seconds")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        # Load best model
        checkpoint = torch.load(self.results_dir / 'best_model.pth')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded best model weights")
        
        return self.train_losses, self.val_losses