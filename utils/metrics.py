# ============================================================================
#  utils/metrics.py
# ============================================================================

import torch
import numpy as np

def dice_coefficient(predictions, targets, smooth=1e-7):
    """Calculate Dice coefficient"""
    predictions = torch.sigmoid(predictions)
    predictions = (predictions > 0.5).float()
    
    intersection = (predictions * targets).sum()
    dice = (2. * intersection + smooth) / (predictions.sum() + targets.sum() + smooth)
    
    return dice.item()

def iou_score(predictions, targets, smooth=1e-7):
    """Calculate IoU score"""
    predictions = torch.sigmoid(predictions)
    predictions = (predictions > 0.5).float()
    
    intersection = (predictions * targets).sum()
    union = predictions.sum() + targets.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()

def pixel_accuracy(predictions, targets):
    """Calculate pixel accuracy"""
    predictions = torch.sigmoid(predictions)
    predictions = (predictions > 0.5).float()
    
    correct = (predictions == targets).float().sum()
    total = targets.numel()
    
    return (correct / total).item()


