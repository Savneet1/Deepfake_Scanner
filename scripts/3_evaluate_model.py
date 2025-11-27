import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import json

# Import dataset from training script
import sys
from pathlib import Path

# Ensure scripts directory is on path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.append(str(SCRIPT_DIR))

from train_model import DeepfakeDataset, create_model  # we will rename the training file

# Configuration
BATCH_SIZE = 4
IMG_SIZE = 224
DEVICE = 'cpu'
MODEL_PATH = 'models/checkpoints/best_model.pt'
DATA_DIR = Path('extracted_frames')

def evaluate_model():
    print("="*60)
    print("Model Evaluation")
    print("="*60)
    
    # Load model
    print("\nLoading model...")
    model = create_model().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("✓ Model loaded successfully")
    
    # Prepare data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    dataset = DeepfakeDataset(DATA_DIR, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Evaluate
    print("\nEvaluating...")
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    
    # Calculate metrics
    print("\n" + "="*60)
    print("Classification Report")
    print("="*60)
    print(classification_report(
        all_labels, all_preds, 
        target_names=['Real', 'Fake'],
        digits=4
    ))
    
    print("\n" + "="*60)
    print("Confusion Matrix")
    print("="*60)
    cm = confusion_matrix(all_labels, all_preds)
    print(f"              Predicted")
    print(f"              Real    Fake")
    print(f"Actual Real   {cm[0][0]:<7} {cm[0][1]:<7}")
    print(f"       Fake   {cm[1][0]:<7} {cm[1][1]:<7}")
    
    # Calculate accuracy
    accuracy = 100.0 * (cm[0][0] + cm[1][1]) / np.sum(cm)
    print(f"\nOverall Accuracy: {accuracy:.2f}%")
    
    # Save results
    results = {
        'accuracy': float(accuracy),
        'confusion_matrix': cm.tolist(),
        'classification_report': classification_report(
            all_labels, all_preds, 
            target_names=['Real', 'Fake'],
            output_dict=True
        )
    }
    
    results_path = Path('logs/evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\n✓ Results saved to {results_path}")
    print("="*60)

if __name__ == "__main__":
    evaluate_model()
