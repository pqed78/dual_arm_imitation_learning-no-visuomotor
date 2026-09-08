import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

class VisionEncoder(nn.Module):
    def __init__(self, backbone_type="resnet18", feature_dim=512, pretrained=True):
        super().__init__()
        self.backbone_type = backbone_type.lower()
        
        # 1. Select the backbone network
        if self.backbone_type == "resnet18":
            net = models.resnet18(pretrained=pretrained)
            self.backbone = nn.Sequential(*list(net.children())[:-1])
            backbone_out_dim = 512
            
        elif self.backbone_type == "resnet50":
            net = models.resnet50(pretrained=pretrained)
            self.backbone = nn.Sequential(*list(net.children())[:-1])
            backbone_out_dim = 2048
            
        elif self.backbone_type == "mobilenet_v3_small":
            net = models.mobilenet_v3_small(pretrained=pretrained)
            self.backbone = nn.Sequential(net.features, nn.AdaptiveAvgPool2d(1))
            backbone_out_dim = 576
            
        elif self.backbone_type == "efficientnet_b0":
            net = models.efficientnet_b0(pretrained=pretrained)
            self.backbone = nn.Sequential(net.features, nn.AdaptiveAvgPool2d(1))
            backbone_out_dim = 1280
            
        elif self.backbone_type == "vit_b_16":
            net = models.vit_b_16(pretrained=pretrained)
            # Remove classification head to output the raw 768-dim class token
            net.heads = nn.Identity()
            self.backbone = net
            backbone_out_dim = 768
            
        else:
            raise ValueError(f"Unsupported backbone_type: {self.backbone_type}. Supported types: resnet18, resnet50, mobilenet_v3_small, efficientnet_b0, vit_b_16")
        
        # 2. Projection layer to match the requested feature_dim (e.g., 512)
        self.proj = nn.Linear(backbone_out_dim, feature_dim) if feature_dim != backbone_out_dim else nn.Identity()
        
        # 3. Standard ImageNet normalization
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def forward(self, x):
        # x shape: (B, C, H, W) or (B, T, C, H, W)
        is_sequence = x.ndim == 5
        if is_sequence:
            B, T, C, H, W = x.shape
            x = x.reshape(B * T, C, H, W)
            
        # ViT requires exactly 224x224 image size
        if self.backbone_type == "vit_b_16":
            x = TF.resize(x, [224, 224], antialias=True)
            
        # Normalize
        x = self.normalize(x)
        
        # Extract features
        features = self.backbone(x)
        
        # CNNs return (B*T, Channels, 1, 1), ViT returns (B*T, Channels)
        if features.dim() > 2:
            features = torch.flatten(features, 1)
        
        # Project to target dimension
        features = self.proj(features)
        
        if is_sequence:
            features = features.reshape(B, T, -1)
            
        return features
