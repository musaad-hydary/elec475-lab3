import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling module"""
    
    def __init__(self, in_channels, out_channels=256, rates=[1, 6, 12, 18]):
        super(ASPP, self).__init__()
        
        self.convs = nn.ModuleList()
        
        # 1x1 convolution
        self.convs.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        
        # Atrous convolutions with different rates
        for rate in rates[1:]:
            self.convs.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        
        # Global average pooling branch
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Output projection
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (len(rates) + 1), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
    
    def forward(self, x):
        res = []
        
        # Apply all atrous convolutions
        for conv in self.convs:
            res.append(conv(x))
        
        # Global pooling branch
        pool = self.global_avg_pool(x)
        pool = F.interpolate(pool, size=x.shape[2:], mode='bilinear', align_corners=False)
        res.append(pool)
        
        # Concatenate all branches
        res = torch.cat(res, dim=1)
        
        return self.project(res)


class StudentSegmentationModel(nn.Module):
    """
    Lightweight segmentation model for knowledge distillation
    Architecture: MobileNetV3-Small backbone + ASPP + Decoder
    """
    
    def __init__(self, num_classes=21, pretrained=True):
        super(StudentSegmentationModel, self).__init__()
        
        # Load pretrained MobileNetV3-Small as backbone
        if pretrained:
            weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
            mobilenet = mobilenet_v3_small(weights=weights)
        else:
            mobilenet = mobilenet_v3_small(weights=None)
        
        # Extract features from MobileNetV3
        self.features = mobilenet.features
        
        # Feature dimensions at different strides
        # Low-level: stride 4, channels=24 (after layer 3)
        # Mid-level: stride 8, channels=40 (after layer 6)
        # High-level: stride 16, channels=576 (final)
        
        self.low_level_channels = 24
        self.mid_level_channels = 40
        self.high_level_channels = 576
        
        # ASPP module on high-level features
        self.aspp = ASPP(self.high_level_channels, out_channels=256, rates=[1, 6, 12, 18])
        
        # Low-level feature projection
        self.low_level_project = nn.Sequential(
            nn.Conv2d(self.low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        # Final classifier
        self.classifier = nn.Conv2d(256, num_classes, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize decoder and classifier weights"""
        for m in [self.aspp, self.low_level_project, self.decoder, self.classifier]:
            if isinstance(m, nn.ModuleList):
                for module in m:
                    self._init_module(module)
            else:
                self._init_module(m)
    
    def _init_module(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x, return_features=False):
        """
        Forward pass
        Args:
            x: Input image (B, 3, H, W)
            return_features: If True, return intermediate features for distillation
        Returns:
            output: Segmentation logits (B, num_classes, H, W)
            features (optional): Dict of intermediate features
        """
        input_shape = x.shape[-2:]
        
        # Extract features at different levels
        features = {}
        
        # Low-level features (stride 4)
        for i in range(4):
            x = self.features[i](x)
        low_level = x  # (B, 24, H/4, W/4)
        features['low'] = low_level
        
        # Mid-level features (stride 8)
        for i in range(4, 7):
            x = self.features[i](x)
        mid_level = x  # (B, 40, H/8, W/8)
        features['mid'] = mid_level
        
        # High-level features (stride 16)
        for i in range(7, len(self.features)):
            x = self.features[i](x)
        high_level = x  # (B, 576, H/16, W/16)
        features['high'] = high_level
        
        # ASPP on high-level features
        aspp_out = self.aspp(high_level)  # (B, 256, H/16, W/16)
        
        # Upsample ASPP output
        aspp_up = F.interpolate(aspp_out, size=low_level.shape[-2:], 
                                mode='bilinear', align_corners=False)
        
        # Project low-level features
        low_level_proj = self.low_level_project(low_level)
        
        # Concatenate and decode
        decoder_input = torch.cat([aspp_up, low_level_proj], dim=1)
        decoder_out = self.decoder(decoder_input)
        
        # Final classification
        output = self.classifier(decoder_out)
        
        # Upsample to original input size
        output = F.interpolate(output, size=input_shape, 
                              mode='bilinear', align_corners=False)
        
        if return_features:
            return output, features
        return output


def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_model():
    """Test model instantiation and forward pass"""
    model = StudentSegmentationModel(num_classes=21, pretrained=True)
    
    # Print model info
    total_params = count_parameters(model)
    print(f"Total trainable parameters: {total_params:,}")
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 512, 512)
    
    # Without features
    output = model(dummy_input, return_features=False)
    print(f"Output shape: {output.shape}")  # Should be (2, 21, 512, 512)
    
    # With features
    output, features = model(dummy_input, return_features=True)
    print(f"\nIntermediate features:")
    for name, feat in features.items():
        print(f"  {name}: {feat.shape}")
    
    print("\nModel test passed!")


if __name__ == "__main__":
    test_model()