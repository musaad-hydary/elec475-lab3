import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import VOCSegmentation
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import time
import sys

from step2_model import StudentSegmentationModel
from step1_test_pretrained import MeanIoU


# Target transform class (no lambda for Mac compatibility)
class TargetTransform:
    """Transform target segmentation mask"""
    def __init__(self, size=(512, 512)):
        self.size = size
    
    def __call__(self, target):
        target = transforms.functional.resize(
            target, 
            self.size, 
            interpolation=transforms.InterpolationMode.NEAREST
        )
        target = transforms.functional.pil_to_tensor(target)
        return target.squeeze(0).long()


class SegmentationTrainer:
    """Training manager for segmentation models"""
    
    def __init__(self, model, train_loader, val_loader, device, 
                 learning_rate=0.001, num_epochs=50, save_dir='checkpoints'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.save_dir = save_dir
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)
        
        # Optimizer with different learning rates for backbone and decoder
        params = [
            {'params': model.features.parameters(), 'lr': learning_rate * 0.1},
            {'params': model.aspp.parameters(), 'lr': learning_rate},
            {'params': model.low_level_project.parameters(), 'lr': learning_rate},
            {'params': model.decoder.parameters(), 'lr': learning_rate},
            {'params': model.classifier.parameters(), 'lr': learning_rate}
        ]
        self.optimizer = optim.Adam(params, lr=learning_rate)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.PolynomialLR(
            self.optimizer, total_iters=num_epochs, power=0.9
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_mious = []
        self.best_miou = 0.0
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for images, targets in pbar:
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update statistics
            running_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        epoch_loss = running_loss / len(self.train_loader)
        return epoch_loss
    
    def validate(self):
        """Validate model"""
        self.model.eval()
        running_loss = 0.0
        miou_calculator = MeanIoU(num_classes=21)
        
        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc="Validating"):
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                
                # Update statistics
                running_loss += loss.item()
                
                # Calculate mIoU
                predictions = outputs.argmax(dim=1)
                for pred, target in zip(predictions.cpu().numpy(), targets.cpu().numpy()):
                    miou_calculator.update(pred, target)
        
        val_loss = running_loss / len(self.val_loader)
        mean_iou, _ = miou_calculator.compute()
        
        return val_loss, mean_iou
    
    def train(self):
        """Full training loop"""
        print("Starting training...")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.num_epochs}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print("-" * 50)
        
        for epoch in range(self.num_epochs):
            start_time = time.time()
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss, val_miou = self.validate()
            
            # Update learning rate
            self.scheduler.step()
            
            # Save statistics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_mious.append(val_miou)
            
            # Print progress
            epoch_time = time.time() - start_time
            print(f"Epoch [{epoch+1}/{self.num_epochs}] - "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Val mIoU: {val_miou:.4f}, "
                  f"Time: {epoch_time:.2f}s")
            
            # Save best model
            if val_miou > self.best_miou:
                self.best_miou = val_miou
                self.save_checkpoint(epoch, 'best_model.pth')
                print(f"  → New best model! mIoU: {val_miou:.4f}")
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, f'checkpoint_epoch_{epoch+1}.pth')
        
        print("\nTraining completed!")
        print(f"Best validation mIoU: {self.best_miou:.4f}")
        
        # Plot training curves
        self.plot_training_curves()
    
    def save_checkpoint(self, epoch, filename):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_mious': self.val_mious,
            'best_miou': self.best_miou
        }
        filepath = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, filepath)
    
    def plot_training_curves(self):
        """Plot and save training curves"""
        epochs = range(1, len(self.train_losses) + 1)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss curves
        ax1.plot(epochs, self.train_losses, label='Train Loss')
        ax1.plot(epochs, self.val_losses, label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # mIoU curve
        ax2.plot(epochs, self.val_mious, label='Val mIoU', color='green')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('mIoU')
        ax2.set_title('Validation mIoU')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_curves.png'), dpi=300)
        print(f"Training curves saved to {self.save_dir}/training_curves.png")
        plt.close()


def check_dataset_exists():
    """Check if dataset exists"""
    voc_path = './data/VOCdevkit/VOC2012'
    if os.path.exists(voc_path):
        print(f"✓ Found dataset at: {voc_path}")
        return True
    print(f"❌ Dataset not found at: {voc_path}")
    print("Please ensure you have downloaded the VOC 2012 dataset.")
    return False


def main():
    # Check dataset first
    if not check_dataset_exists():
        sys.exit(1)
    
    # Set device (MPS for Mac M1/M2, CUDA for NVIDIA, CPU otherwise)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # Hyperparameters
    BATCH_SIZE = 8
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    
    # Data transforms (no lambda for Mac)
    train_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    target_transform = TargetTransform(size=(512, 512))
    
    # Load datasets
    print("Loading datasets...")
    try:
        train_dataset = VOCSegmentation(
            root='./data',
            year='2012',
            image_set='train',
            download=False,  # Don't auto-download
            transform=train_transform,
            target_transform=target_transform
        )
        
        val_dataset = VOCSegmentation(
            root='./data',
            year='2012',
            image_set='val',
            download=False,  # Don't auto-download
            transform=val_transform,
            target_transform=target_transform
        )
    except Exception as e:
        print(f"❌ Error loading datasets: {e}")
        print("\nPlease ensure the VOC 2012 dataset is at ./data/VOCdevkit/VOC2012/")
        sys.exit(1)
    
    # IMPORTANT: num_workers=0 for Mac compatibility
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Mac compatibility
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,  # Mac compatibility
        pin_memory=True if device.type == 'cuda' else False
    )
    
    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")
    
    # Create model
    print("\nCreating student model...")
    model = StudentSegmentationModel(num_classes=21, pretrained=True)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = SegmentationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=LEARNING_RATE,
        num_epochs=NUM_EPOCHS,
        save_dir='checkpoints_baseline'
    )
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()