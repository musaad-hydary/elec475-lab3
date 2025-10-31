import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import VOCSegmentation
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
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


class FeatureProjection(nn.Module):
    """Project features to match dimensions for distillation"""
    def __init__(self, in_channels, out_channels):
        super(FeatureProjection, self).__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        return self.projection(x)


class DistillationLoss(nn.Module):
    """Combined distillation loss: Response-based + Feature-based"""
    
    def __init__(self, alpha=0.5, beta=0.5, temperature=4.0, 
                 feature_weight=0.3, num_classes=21, device='mps'):
        super(DistillationLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.feature_weight = feature_weight
        self.num_classes = num_classes
        self.device = device
        
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=255)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        
        # Feature projections to match teacher dimensions
        # Student: low=24, mid=40, high=576
        # Teacher: low=256, mid=512, high=2048
        self.projections = nn.ModuleDict({
            'low': FeatureProjection(24, 256),
            'mid': FeatureProjection(40, 512),
            'high': FeatureProjection(576, 2048)
        }).to(device)
    
    def response_based_loss(self, student_logits, teacher_logits, targets):
        """Response-based distillation loss"""
        # Ground truth loss
        gt_loss = self.ce_loss(student_logits, targets)
        
        # Distillation loss (KL divergence)
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        
        # Create mask for valid pixels
        mask = (targets != 255).unsqueeze(1).float()
        student_soft = student_soft * mask
        teacher_soft = teacher_soft * mask
        
        distill_loss = self.kl_loss(student_soft, teacher_soft) * (self.temperature ** 2)
        
        # Combined loss
        total_loss = self.alpha * gt_loss + self.beta * distill_loss
        
        return total_loss, gt_loss, distill_loss
    
    def cosine_similarity_loss(self, student_feat, teacher_feat):
        """Feature-based distillation using cosine similarity"""
        # Normalize features channel-wise
        student_norm = F.normalize(student_feat, p=2, dim=1)
        teacher_norm = F.normalize(teacher_feat, p=2, dim=1)
        
        # Cosine similarity per spatial location
        cos_sim = (student_norm * teacher_norm).sum(dim=1)
        
        # Loss is 1 - similarity (minimize distance)
        loss = (1 - cos_sim).mean()
        return loss
    
    def feature_based_loss(self, student_features, teacher_features):
        """
        Feature-based distillation loss with proper dimension matching
        Projects student features to match teacher dimensions
        """
        total_loss = 0.0
        num_levels = 0
        
        for level in ['low', 'mid', 'high']:
            if level in student_features and level in teacher_features:
                student_feat = student_features[level]
                teacher_feat = teacher_features[level]
                
                # Resize spatially to match
                if student_feat.shape[2:] != teacher_feat.shape[2:]:
                    teacher_feat = F.interpolate(
                        teacher_feat, 
                        size=student_feat.shape[2:],
                        mode='bilinear', 
                        align_corners=False
                    )
                
                # Project student features to match teacher channels
                student_feat_proj = self.projections[level](student_feat)
                
                # Now dimensions should match
                loss = self.cosine_similarity_loss(student_feat_proj, teacher_feat)
                total_loss += loss
                num_levels += 1
        
        if num_levels > 0:
            total_loss /= num_levels
        
        return total_loss


class TeacherModel(nn.Module):
    """Wrapper for teacher model to extract intermediate features"""
    
    def __init__(self):
        super(TeacherModel, self).__init__()
        self.model = fcn_resnet50(weights=FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1)
        
        # Freeze teacher
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.model.eval()
    
    def forward(self, x, return_features=False):
        """Forward pass through teacher"""
        if not return_features:
            with torch.no_grad():
                return self.model(x)['out']
        
        features = {}
        backbone = self.model.backbone
        
        with torch.no_grad():
            x = backbone.conv1(x)
            x = backbone.bn1(x)
            x = backbone.relu(x)
            x = backbone.maxpool(x)
            
            # Layer 1 (low-level, stride 4, 256 channels)
            x = backbone.layer1(x)
            features['low'] = x
            
            # Layer 2 (mid-level, stride 8, 512 channels)
            x = backbone.layer2(x)
            features['mid'] = x
            
            # Layer 3 and 4 (high-level, stride 16, 2048 channels)
            x = backbone.layer3(x)
            x = backbone.layer4(x)
            features['high'] = x
            
            output = self.model.classifier(x)
        
        return output, features


class KnowledgeDistillationTrainer:
    """Training manager with knowledge distillation"""
    
    def __init__(self, student_model, teacher_model, train_loader, val_loader, 
                 device, method='response', learning_rate=0.001, num_epochs=50, 
                 alpha=0.5, beta=0.5, temperature=4.0, feature_weight=0.3,
                 save_dir='checkpoints_kd'):
        
        self.student = student_model
        self.teacher = teacher_model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.method = method
        self.num_epochs = num_epochs
        self.save_dir = save_dir
        
        os.makedirs(save_dir, exist_ok=True)
        
        self.distill_loss = DistillationLoss(
            alpha=alpha, beta=beta, temperature=temperature, 
            feature_weight=feature_weight, device=device
        )
        
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=255)
        
        # Optimizer - include projection layers for feature-based methods
        if method == 'feature' or method == 'both':
            params = [
                {'params': student_model.features.parameters(), 'lr': learning_rate * 0.1},
                {'params': student_model.aspp.parameters(), 'lr': learning_rate},
                {'params': student_model.low_level_project.parameters(), 'lr': learning_rate},
                {'params': student_model.decoder.parameters(), 'lr': learning_rate},
                {'params': student_model.classifier.parameters(), 'lr': learning_rate},
                {'params': self.distill_loss.projections.parameters(), 'lr': learning_rate}  # Add projections
            ]
        else:
            params = [
                {'params': student_model.features.parameters(), 'lr': learning_rate * 0.1},
                {'params': student_model.aspp.parameters(), 'lr': learning_rate},
                {'params': student_model.low_level_project.parameters(), 'lr': learning_rate},
                {'params': student_model.decoder.parameters(), 'lr': learning_rate},
                {'params': student_model.classifier.parameters(), 'lr': learning_rate}
            ]
        
        self.optimizer = optim.Adam(params, lr=learning_rate)
        
        self.scheduler = optim.lr_scheduler.PolynomialLR(
            self.optimizer, total_iters=num_epochs, power=0.9
        )
        
        self.train_losses = []
        self.val_losses = []
        self.val_mious = []
        self.best_miou = 0.0
    
    def train_epoch(self):
        """Train for one epoch with knowledge distillation"""
        self.student.train()
        self.teacher.eval()
        self.distill_loss.projections.train()  # Ensure projections are trainable
        
        running_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Training with {self.method} KD")
        for images, targets in pbar:
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # Student forward
            if self.method == 'feature' or self.method == 'both':
                student_logits, student_features = self.student(images, return_features=True)
            else:
                student_logits = self.student(images, return_features=False)
            
            # Teacher forward
            if self.method == 'feature' or self.method == 'both':
                teacher_logits, teacher_features = self.teacher(images, return_features=True)
            else:
                teacher_logits = self.teacher(images, return_features=False)
            
            # Calculate loss
            if self.method == 'response':
                loss, gt_loss, distill_loss = self.distill_loss.response_based_loss(
                    student_logits, teacher_logits, targets
                )
                feature_loss = torch.tensor(0.0)
                
            elif self.method == 'feature':
                gt_loss = self.ce_loss(student_logits, targets)
                feature_loss = self.distill_loss.feature_based_loss(
                    student_features, teacher_features
                )
                loss = gt_loss + self.distill_loss.feature_weight * feature_loss
                distill_loss = torch.tensor(0.0)
                
            else:  # 'both'
                loss, gt_loss, distill_loss = self.distill_loss.response_based_loss(
                    student_logits, teacher_logits, targets
                )
                feature_loss = self.distill_loss.feature_based_loss(
                    student_features, teacher_features
                )
                loss = loss + self.distill_loss.feature_weight * feature_loss
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'gt': f'{gt_loss.item():.4f}'
            })
        
        return running_loss / len(self.train_loader)
    
    def validate(self):
        """Validate model"""
        self.student.eval()
        running_loss = 0.0
        miou_calculator = MeanIoU(num_classes=21)
        
        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc="Validating"):
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.student(images, return_features=False)
                loss = self.ce_loss(outputs, targets)
                
                running_loss += loss.item()
                
                predictions = outputs.argmax(dim=1)
                for pred, target in zip(predictions.cpu().numpy(), targets.cpu().numpy()):
                    miou_calculator.update(pred, target)
        
        val_loss = running_loss / len(self.val_loader)
        mean_iou, _ = miou_calculator.compute()
        
        return val_loss, mean_iou
    
    def train(self):
        """Full training loop"""
        print(f"Starting training with {self.method} distillation...")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.num_epochs}")
        print("-" * 50)
        
        for epoch in range(self.num_epochs):
            start_time = time.time()
            
            train_loss = self.train_epoch()
            val_loss, val_miou = self.validate()
            
            self.scheduler.step()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_mious.append(val_miou)
            
            epoch_time = time.time() - start_time
            print(f"Epoch [{epoch+1}/{self.num_epochs}] - "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Val mIoU: {val_miou:.4f}, "
                  f"Time: {epoch_time:.2f}s")
            
            if val_miou > self.best_miou:
                self.best_miou = val_miou
                self.save_checkpoint(epoch, 'best_model.pth')
                print(f"  → New best model! mIoU: {val_miou:.4f}")
            
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, f'checkpoint_epoch_{epoch+1}.pth')
        
        print("\nTraining completed!")
        print(f"Best validation mIoU: {self.best_miou:.4f}")
        self.plot_training_curves()
    
    def save_checkpoint(self, epoch, filename):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.student.state_dict(),
            'projection_state_dict': self.distill_loss.projections.state_dict() if self.method in ['feature', 'both'] else None,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_mious': self.val_mious,
            'best_miou': self.best_miou,
            'method': self.method
        }
        torch.save(checkpoint, os.path.join(self.save_dir, filename))
    
    def plot_training_curves(self):
        """Plot training curves"""
        epochs = range(1, len(self.train_losses) + 1)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(epochs, self.train_losses, label='Train Loss')
        ax1.plot(epochs, self.val_losses, label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'Training Loss ({self.method} KD)')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(epochs, self.val_mious, label='Val mIoU', color='green')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('mIoU')
        ax2.set_title(f'Validation mIoU ({self.method} KD)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'training_curves_{self.method}.png'), dpi=300)
        print(f"Training curves saved to {self.save_dir}/training_curves_{self.method}.png")
        plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='response', choices=['response', 'feature', 'both'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--temperature', type=float, default=4.0)
    parser.add_argument('--feature_weight', type=float, default=0.3)
    
    args = parser.parse_args()
    
    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    target_transform = TargetTransform(size=(512, 512))
    
    # Datasets
    print("Loading datasets...")
    train_dataset = VOCSegmentation(
        root='./data', year='2012', image_set='train', download=False,
        transform=train_transform, target_transform=target_transform
    )
    
    val_dataset = VOCSegmentation(
        root='./data', year='2012', image_set='val', download=False,
        transform=val_transform, target_transform=target_transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")
    
    # Models
    print("\nCreating models...")
    student = StudentSegmentationModel(num_classes=21, pretrained=True).to(device)
    teacher = TeacherModel().to(device)
    
    student_params = sum(p.numel() for p in student.parameters())
    teacher_params = sum(p.numel() for p in teacher.parameters())
    
    print(f"Student parameters: {student_params:,}")
    print(f"Teacher parameters: {teacher_params:,}")
    
    # Trainer
    save_dir = f'checkpoints_kd_{args.method}'
    trainer = KnowledgeDistillationTrainer(
        student_model=student, teacher_model=teacher,
        train_loader=train_loader, val_loader=val_loader,
        device=device, method=args.method,
        learning_rate=args.lr, num_epochs=args.epochs,
        alpha=args.alpha, beta=args.beta, 
        temperature=args.temperature, feature_weight=args.feature_weight,
        save_dir=save_dir
    )
    
    trainer.train()


if __name__ == "__main__":
    main()