import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import GradScaler, autocast
import time
import numpy as np
from pathlib import Path
from .metrics import dice_coefficient, iou_score, pixel_accuracy, sensitivity, specificity

class Trainer:
    def __init__(self, model, train_loader, val_loader, device, learning_rate=1e-4, 
                 weight_decay=1e-4, mixed_precision=True, gradient_accumulation_steps=1,
                 warmup_epochs=5, cosine_annealing=True, early_stopping_patience=15):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.mixed_precision = mixed_precision and device.type == 'cuda'
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.early_stopping_patience = early_stopping_patience
        
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay,
            eps=1e-8,
            fused=True if device.type == 'cuda' else False
        )
        
        self.criterion = nn.BCEWithLogitsLoss()
        self.scaler = GradScaler() if self.mixed_precision else None
        
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
        
        self.best_val_loss = float('inf')
        self.best_val_dice = 0.0
        self.patience_counter = 0
        
        self.train_losses = []
        self.val_losses = []
        self.val_dices = []
        self.val_ious = []
        
        self.results_dir = Path('results')
        self.results_dir.mkdir(exist_ok=True)
        
        # Benchmarking metrics
        self.epoch_times = []
        self.batch_times = []
        
        print(f"Trainer initialized:")
        print(f"  Device: {device}")
        print(f"  Mixed precision: {self.mixed_precision}")
        print(f"  Gradient accumulation: {gradient_accumulation_steps}")
        print(f"  Scheduler: {cosine_annealing}")
    
    def train_epoch(self):
        self.model.train()
        epoch_loss = 0
        num_batches = len(self.train_loader)
        batch_times = []
        
        self.optimizer.zero_grad()
        
        for batch_idx, (images, masks) in enumerate(self.train_loader):
            batch_start = time.time()
            
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)
            
            if self.mixed_precision:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                    loss = loss / self.gradient_accumulation_steps
                
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                loss = loss / self.gradient_accumulation_steps
                
                loss.backward()
                
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            epoch_loss += loss.item() * self.gradient_accumulation_steps
            batch_times.append(time.time() - batch_start)
            
            if batch_idx % max(1, num_batches // 10) == 0:
                print(f'  Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}')
        
        if num_batches % self.gradient_accumulation_steps != 0:
            if self.mixed_precision:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()
        
        self.batch_times.extend(batch_times)
        return epoch_loss / num_batches
    
    def validate(self):
        self.model.eval()
        val_loss = 0
        dice_scores = []
        iou_scores = []
        pixel_accs = []
        sens_scores = []
        spec_scores = []
        
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
                
                dice_scores.append(dice_coefficient(outputs, masks))
                iou_scores.append(iou_score(outputs, masks))
                pixel_accs.append(pixel_accuracy(outputs, masks))
                sens_scores.append(sensitivity(outputs, masks).item())
                spec_scores.append(specificity(outputs, masks).item())
        
        metrics = {
            'loss': val_loss / num_batches,
            'dice': np.mean(dice_scores),
            'iou': np.mean(iou_scores),
            'pixel_acc': np.mean(pixel_accs),
            'sensitivity': np.mean(sens_scores),
            'specificity': np.mean(spec_scores)
        }
        
        return metrics
    
    def train(self, num_epochs):
        print(f"Starting training for {num_epochs} epochs...")
        training_start = time.time()
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            train_loss = self.train_epoch()
            val_metrics = self.validate()
            
            if self.scheduler:
                self.scheduler.step()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_metrics['loss'])
            self.val_dices.append(val_metrics['dice'])
            self.val_ious.append(val_metrics['iou'])
            
            epoch_time = time.time() - epoch_start
            self.epoch_times.append(epoch_time)
            
            if val_metrics['dice'] > self.best_val_dice:
                self.best_val_dice = val_metrics['dice']
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_metrics['loss'],
                    'best_val_dice': self.best_val_dice,
                    'best_val_iou': val_metrics['iou'],
                    'metrics': val_metrics
                }, self.results_dir / 'best_model.pth')
            else:
                self.patience_counter += 1
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'  Train Loss: {train_loss:.4f}')
            print(f'  Val Loss: {val_metrics["loss"]:.4f}')
            print(f'  Val Dice: {val_metrics["dice"]:.4f}')
            print(f'  Val IoU: {val_metrics["iou"]:.4f}')
            print(f'  Val Pixel Acc: {val_metrics["pixel_acc"]:.4f}')
            print(f'  Sensitivity: {val_metrics["sensitivity"]:.4f}')
            print(f'  Specificity: {val_metrics["specificity"]:.4f}')
            print(f'  LR: {current_lr:.2e}')
            print(f'  Time: {epoch_time:.2f}s')
            print(f'  Patience: {self.patience_counter}/{self.early_stopping_patience}')
            print('-' * 60)
            
            if self.patience_counter >= self.early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        total_time = time.time() - training_start
        
        print(f"\n{'='*60}")
        print("TRAINING COMPLETE - PERFORMANCE METRICS")
        print(f"{'='*60}")
        print(f"Total training time: {total_time:.2f}s ({total_time/60:.2f}min)")
        print(f"Average epoch time: {np.mean(self.epoch_times):.2f}s")
        print(f"Average batch time: {np.mean(self.batch_times)*1000:.2f}ms")
        print(f"Best validation Dice: {self.best_val_dice:.4f}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Final validation IoU: {self.val_ious[-1]:.4f}")
        print(f"{'='*60}\n")
        
        checkpoint = torch.load(self.results_dir / 'best_model.pth', weights_only=False)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        return self.train_losses, self.val_losses