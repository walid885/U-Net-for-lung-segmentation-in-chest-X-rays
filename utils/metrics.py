import torch
import numpy as np

def dice_coefficient(predictions, targets, smooth=1e-7):
    """Calculate Dice coefficient"""
    predictions = torch.sigmoid(predictions)
    predictions = (predictions > 0.5).float()
    
    predictions = predictions.contiguous().view(-1)
    targets = targets.contiguous().view(-1)
    
    intersection = (predictions * targets).sum()
    dice = (2. * intersection + smooth) / (predictions.sum() + targets.sum() + smooth)
    
    return dice.item()

def iou_score(predictions, targets, smooth=1e-7):
    """Calculate IoU score"""
    predictions = torch.sigmoid(predictions)
    predictions = (predictions > 0.5).float()
    
    predictions = predictions.contiguous().view(-1)
    targets = targets.contiguous().view(-1)
    
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

def sensitivity(predictions, targets, smooth=1e-7):
    """Calculate sensitivity (recall/true positive rate)"""
    predictions = torch.sigmoid(predictions)
    predictions = (predictions > 0.5).float()
    
    predictions = predictions.contiguous().view(-1)
    targets = targets.contiguous().view(-1)
    
    true_positive = (predictions * targets).sum()
    false_negative = ((1 - predictions) * targets).sum()
    
    return (true_positive + smooth) / (true_positive + false_negative + smooth)

def specificity(predictions, targets, smooth=1e-7):
    """Calculate specificity (true negative rate)"""
    predictions = torch.sigmoid(predictions)
    predictions = (predictions > 0.5).float()
    
    predictions = predictions.contiguous().view(-1)
    targets = targets.contiguous().view(-1)
    
    true_negative = ((1 - predictions) * (1 - targets)).sum()
    false_positive = (predictions * (1 - targets)).sum()
    
    return (true_negative + smooth) / (true_negative + false_positive + smooth)