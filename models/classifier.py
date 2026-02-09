"""
MIX-MLP: Multi-Path Classification Head

Dual-pathway architecture combining residual connections (for linearly separable patterns)
with expansion paths (for modeling complex label co-occurrences). Designed for
multi-label CheXpert classification with 14 pathology categories.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Optional


class MixMLP(nn.Module):
    """
    Dual-path MLP head for multi-label classification.
    
    Combines two complementary pathways:
    - Residual path: preserves linear separability for simple patterns
      (e.g., cardiomegaly from cardiac silhouette size)
    - Expansion path: models complex label interactions via higher-dimensional space
      (e.g., CHF presenting with both edema and effusion)
    
    The paths are fused with layer normalization before final classification.
    """
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int = 1024, 
        num_labels: int = 14, 
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Residual path: identity + shallow transform
        self.res_block = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Expansion path: dimension expansion for complex interactions
        self.exp_block = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Fusion and classification
        self.fuse = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(input_dim, num_labels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, D] input features (from backbone global pool or view fusion)
            
        Returns:
            logits: [B, num_labels] raw classification scores
        """
        residual = self.res_block(x)
        expanded = self.exp_block(x)
        fused = self.fuse(x + residual + expanded)
        return self.classifier(fused)


class ViewAttention(nn.Module):
    """
    Learns attention weights across multiple CXR views.
    
    Handles variable view availability (frontal only vs. frontal + lateral)
    via masking. Produces a unified representation by weighted aggregation.
    """
    
    def __init__(self, dim: int):
        super().__init__()
        self.score_net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1)
        )
    
    def forward(self, view_features: torch.Tensor, view_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            view_features: [B, V, D] features from each view
            view_mask: [B, V] binary mask (1 = view exists, 0 = missing)
            
        Returns:
            fused: [B, D] attention-weighted combination
        """
        scores = self.score_net(view_features).squeeze(-1)  # [B, V]
        scores = scores.masked_fill(view_mask <= 0, float('-inf'))
        weights = F.softmax(scores, dim=1)  # [B, V]
        weights = torch.nan_to_num(weights, nan=0.0)  # Handle all-masked edge case
        return (weights.unsqueeze(-1) * view_features).sum(dim=1)


class MultiViewClassifier(nn.Module):
    """
    End-to-end multi-view CXR classifier.
    
    Processes frontal and lateral views through a shared backbone,
    fuses them via learned attention, and classifies using MIX-MLP.
    Handles missing views gracefully via attention masking.
    """
    
    def __init__(
        self, 
        backbone_name: str = "convnext_tiny",
        pretrained: bool = True,
        num_labels: int = 14,
        hidden_dim: int = 1024,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Shared backbone for all views
        self.backbone = timm.create_model(
            backbone_name, 
            pretrained=pretrained, 
            num_classes=0, 
            global_pool='avg'
        )
        feat_dim = self.backbone.num_features
        
        # View fusion
        self.view_attn = ViewAttention(feat_dim)
        
        # Classification head
        self.head = MixMLP(
            input_dim=feat_dim,
            hidden_dim=hidden_dim,
            num_labels=num_labels,
            dropout=dropout
        )
        
        self.feat_dim = feat_dim
    
    def forward(
        self, 
        views: torch.Tensor, 
        view_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            views: [B, V, C, H, W] multi-view images (V=2: frontal, lateral)
            view_mask: [B, V] binary mask indicating view availability
            
        Returns:
            logits: [B, num_labels] classification scores
        """
        B, V, C, H, W = views.shape
        
        # Process all views through shared backbone
        flat = views.view(B * V, C, H, W)
        features = self.backbone(flat)  # [B*V, D]
        features = features.view(B, V, -1)  # [B, V, D]
        
        # Fuse views
        fused = self.view_attn(features, view_mask)  # [B, D]
        
        # Classify
        return self.head(fused)
    
    def freeze_backbone(self) -> None:
        """Freeze backbone for head-only training."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self, stages: Optional[list] = None) -> None:
        """
        Unfreeze backbone parameters.
        
        Args:
            stages: if provided, only unfreeze specified stages (for ConvNeXt)
                   None = unfreeze everything
        """
        if stages is None:
            for param in self.backbone.parameters():
                param.requires_grad = True
            return
        
        # ConvNeXt-specific: unfreeze selected stages
        if hasattr(self.backbone, 'stages'):
            for idx in stages:
                if idx < len(self.backbone.stages):
                    for param in self.backbone.stages[idx].parameters():
                        param.requires_grad = True
        
        # Also unfreeze final norm if present
        if hasattr(self.backbone, 'norm'):
            for param in self.backbone.norm.parameters():
                param.requires_grad = True


def build_classifier(
    backbone: str = "convnext_tiny",
    pretrained: bool = True,
    num_labels: int = 14,
) -> MultiViewClassifier:
    """
    Factory function for multi-view classifier.
    
    Args:
        backbone: timm model name
        pretrained: load ImageNet weights
        num_labels: number of pathology classes
        
    Returns:
        Configured MultiViewClassifier
    """
    return MultiViewClassifier(
        backbone_name=backbone,
        pretrained=pretrained,
        num_labels=num_labels,
    )
