"""
Step 1: Test Pretrained FCN-ResNet50 on PASCAL VOC 2012
MAC-COMPATIBLE VERSION - Fixes multiprocessing and lambda issues
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import VOCSegmentation
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
import numpy as np
from tqdm import tqdm
import os
import sys


class MeanIoU:
    """Calculate Mean Intersection over Union for semantic segmentation"""
    
    def __init__(self, num_classes=21):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
    
    def update(self, pred, target):
        """
        Update confusion matrix
        pred: predicted segmentation (H, W) with class indices
        target: ground truth segmentation (H, W) with class indices
        """
        # Flatten arrays
        pred = pred.flatten()
        target = target.flatten()
        
        # Filter out ignore index (255 in VOC)
        mask = (target != 255)
        pred = pred[mask]
        target = target[mask]
        
        # Update confusion matrix
        for t, p in zip(target, pred):
            self.confusion_matrix[t, p] += 1
    
    def compute(self):
        """Compute mean IoU from confusion matrix"""
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusion_matrix)
        union = (self.confusion_matrix.sum(axis=1) + 
                 self.confusion_matrix.sum(axis=0) - 
                 intersection)
        
        # Avoid division by zero
        iou = intersection / (union + 1e-10)
        
        # Mean IoU (excluding background if desired)
        mean_iou = np.nanmean(iou)
        
        return mean_iou, iou


# Define target transform as a proper class (not lambda) for Mac compatibility
class TargetTransform:
    """Transform target segmentation mask"""
    def __init__(self, size=(520, 520)):
        self.size = size
    
    def __call__(self, target):
        # Resize with nearest neighbor interpolation
        target = transforms.functional.resize(
            target, 
            self.size, 
            interpolation=transforms.InterpolationMode.NEAREST
        )
        # Convert to tensor and squeeze
        target = transforms.functional.pil_to_tensor(target)
        target = target.squeeze(0).long()
        return target


def evaluate_model(model, dataloader, device):
    """Evaluate model on dataset"""
    model.eval()
    miou_calculator = MeanIoU(num_classes=21)
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            
            # Forward pass
            outputs = model(images)['out']  # FCN returns dict with 'out' key
            predictions = outputs.argmax(dim=1)  # (B, H, W)
            
            # Update metrics
            for pred, target in zip(predictions.cpu().numpy(), targets.numpy()):
                miou_calculator.update(pred, target)
    
    mean_iou, class_iou = miou_calculator.compute()
    return mean_iou, class_iou


def check_dataset_exists(data_root='./data'):
    """Check if PASCAL VOC 2012 dataset already exists"""
    voc_path = os.path.join(data_root, 'VOCdevkit', 'VOC2012')
    if os.path.exists(voc_path):
        # Check for essential files
        imagesets_path = os.path.join(voc_path, 'ImageSets', 'Segmentation', 'val.txt')
        if os.path.exists(imagesets_path):
            print(f"✓ Found existing PASCAL VOC 2012 dataset at: {voc_path}")
            return True
    return False


def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    print(f"Using device: {device}")
    
    # Define transforms (no lambda functions for Mac compatibility)
    transform = transforms.Compose([
        transforms.Resize((520, 520)),  # Slightly larger for better results
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Use proper class instead of lambda
    target_transform = TargetTransform(size=(520, 520))
    
    # Check if dataset already exists
    print("Checking for existing PASCAL VOC 2012 dataset...")
    dataset_exists = check_dataset_exists('./data')
    
    if not dataset_exists:
        print("❌ Dataset not found!")
        print("Please ensure the dataset is at: ./data/VOCdevkit/VOC2012/")
        sys.exit(1)
    
    # Load PASCAL VOC 2012 dataset
    print("Loading PASCAL VOC 2012 dataset...")
    try:
        test_dataset = VOCSegmentation(
            root='./data',
            year='2012',
            image_set='val',  # Using validation set as test set
            download=False,  # Don't try to download since we have it
            transform=transform,
            target_transform=target_transform
        )
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        sys.exit(1)
    
    print(f"✓ Dataset loaded successfully!")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # IMPORTANT: Set num_workers=0 for Mac to avoid multiprocessing issues
    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,  # Set to 0 for Mac compatibility
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Load pretrained FCN-ResNet50
    print("\nLoading pretrained FCN-ResNet50...")
    try:
        model = fcn_resnet50(weights=FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1)
    except Exception as e:
        print(f"⚠️  Error loading with new API: {e}")
        print("Trying alternative method...")
        try:
            model = fcn_resnet50(pretrained=True)
        except Exception as e2:
            print(f"❌ Failed to load model: {e2}")
            sys.exit(1)
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Evaluate
    print("\nStarting evaluation...")
    print("(This may take a few minutes...)")
    mean_iou, class_iou = evaluate_model(model, test_loader, device)
    
    # Print results
    print("\n" + "="*50)
    print(f"Mean IoU: {mean_iou:.4f}")
    print("="*50)
    
    # VOC class names
    voc_classes = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
        'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
        'sofa', 'train', 'tvmonitor'
    ]
    
    print("\nPer-class IoU:")
    for i, (cls_name, iou) in enumerate(zip(voc_classes, class_iou)):
        print(f"{cls_name:15s}: {iou:.4f}")
    
    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()