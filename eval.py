import torch
from data_loader import create_data_loaders
from models.unet import UNet
from utils.metrics import dice_coefficient, iou_score, pixel_accuracy, sensitivity, specificity
import numpy as np

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = UNet(n_channels=3, n_classes=1).to(device)
    checkpoint = torch.load('results/best_model.pth', weights_only=False, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("Model loaded successfully")
    print(f"Best validation Dice: {checkpoint.get('best_val_dice', 'N/A')}")
    
    # Load test data
    _, _, test_loader = create_data_loaders(
        'Data/NLM-MontgomeryCXRSet/MontgomerySet', 
        batch_size=1,
        num_workers=0  # Set to 0 to avoid multiprocessing issues
    )
    
    # Evaluate
    dice_scores = []
    iou_scores = []
    pixel_accs = []
    sens_scores = []
    spec_scores = []
    
    print(f"\nEvaluating on {len(test_loader)} test samples...")
    
    with torch.no_grad():
        for i, (images, masks) in enumerate(test_loader):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            
            dice_scores.append(dice_coefficient(outputs, masks))
            iou_scores.append(iou_score(outputs, masks))
            pixel_accs.append(pixel_accuracy(outputs, masks))
            sens_scores.append(sensitivity(outputs, masks).item())
            spec_scores.append(specificity(outputs, masks).item())
            
            if (i + 1) % 5 == 0:
                print(f"  Processed {i+1}/{len(test_loader)} samples")
    
    # Print results
    print(f"\n{'='*60}")
    print("TEST SET EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Dice Score:      {np.mean(dice_scores):.4f} ± {np.std(dice_scores):.4f}")
    print(f"IoU Score:       {np.mean(iou_scores):.4f} ± {np.std(iou_scores):.4f}")
    print(f"Pixel Accuracy:  {np.mean(pixel_accs):.4f} ± {np.std(pixel_accs):.4f}")
    print(f"Sensitivity:     {np.mean(sens_scores):.4f} ± {np.std(sens_scores):.4f}")
    print(f"Specificity:     {np.mean(spec_scores):.4f} ± {np.std(spec_scores):.4f}")
    print(f"{'='*60}\n")
    
    return {
        'dice': np.mean(dice_scores),
        'iou': np.mean(iou_scores),
        'pixel_acc': np.mean(pixel_accs),
        'sensitivity': np.mean(sens_scores),
        'specificity': np.mean(spec_scores)
    }

if __name__ == '__main__':
    results = evaluate()