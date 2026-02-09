"""
PRO-FA: Progressive Feature Alignment Module

Implements hierarchical visual perception with concept-aligned region tokens.
Extracts multi-scale features and aligns them to anatomical/pathological concepts
using BioClinicalBERT embeddings for interpretable feature learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class RegionAttention(nn.Module):
    """
    Learns K region tokens via cross-attention over spatial feature maps.
    
    Each region token aggregates information from relevant spatial locations,
    enabling pathology localization without explicit bounding box supervision.
    """
    
    def __init__(self, dim: int, num_regions: int = 8, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_regions = num_regions
        self.queries = nn.Parameter(torch.randn(num_regions, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
    
    def forward(self, pixel_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pixel_tokens: [B, HW, D] flattened spatial features
            
        Returns:
            region_tokens: [B, K, D] aggregated region representations
            attention_weights: [B, K, HW] for interpretability
        """
        B = pixel_tokens.size(0)
        queries = self.queries.unsqueeze(0).expand(B, -1, -1)
        region_tokens, attn_weights = self.attn(queries, pixel_tokens, pixel_tokens, need_weights=True)
        return region_tokens, attn_weights


class ConceptProjector(nn.Module):
    """
    Projects visual tokens and text concept embeddings into a shared space.
    
    Enables direct comparison between image regions and semantic concepts
    for interpretable pathology detection.
    """
    
    def __init__(self, visual_dim: int, text_dim: int = 768, proj_dim: int = 512):
        super().__init__()
        self.visual_proj = nn.Linear(visual_dim, proj_dim)
        self.text_proj = nn.Linear(text_dim, proj_dim, bias=False)
    
    def forward(self, visual: torch.Tensor, text: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            visual: [B, N, D_v] visual tokens (pixel, region, or organ level)
            text: [C, D_t] concept embeddings from BioClinicalBERT
            
        Returns:
            visual_proj: [B, N, D] L2-normalized projected visual tokens
            text_proj: [C, D] L2-normalized projected concept embeddings
        """
        v = F.normalize(self.visual_proj(visual), dim=-1)
        t = F.normalize(self.text_proj(text), dim=-1)
        return v, t


class PROFA(nn.Module):
    """
    PRO-FA: Progressive Region-Organ Feature Alignment
    
    Extracts hierarchical visual features and aligns them to a curated concept bank.
    Supports multi-level pathology detection via organ-level global features and
    region-level MIL (Multiple Instance Learning) aggregation.
    
    Architecture:
        backbone (ConvNeXt) -> pixel tokens -> region tokens (via attention)
                           -> organ token (global pool)
        
        Both region and organ tokens are projected to concept space for:
        1. Direct classification via linear heads
        2. Concept-aligned MIL scoring
    """
    
    def __init__(
        self,
        backbone_name: str = "convnext_tiny",
        pretrained: bool = True,
        num_regions: int = 8,
        proj_dim: int = 512,
        num_labels: int = 14,
        num_anatomy: int = 17,
    ):
        super().__init__()
        
        # Feature extractor (final stage only for efficiency)
        self.backbone = timm.create_model(
            backbone_name, 
            pretrained=pretrained, 
            features_only=True, 
            out_indices=(3,)  # Last stage
        )
        backbone_dim = self.backbone.feature_info.channels()[0]
        
        # Pixel-to-region aggregation
        self.pixel_proj = nn.Linear(backbone_dim, proj_dim)
        self.region_attn = RegionAttention(proj_dim, num_regions=num_regions)
        
        # Global organ-level representation
        self.organ_proj = nn.Linear(backbone_dim, proj_dim)
        
        # Concept alignment
        self.concept_proj = ConceptProjector(proj_dim, text_dim=768, proj_dim=proj_dim)
        
        # Classification heads
        self.organ_classifier = nn.Linear(proj_dim, num_labels)
        
        # Store constants
        self.num_anatomy = num_anatomy
        self.num_labels = num_labels
        self.proj_dim = proj_dim
    
    def forward(
        self, 
        images: torch.Tensor, 
        concept_embeddings: torch.Tensor,
        return_attention: bool = False
    ) -> dict:
        """
        Args:
            images: [B, 3, H, W] input CXR images
            concept_embeddings: [C, 768] BioClinicalBERT concept embeddings
                               (anatomy concepts first, then pathology concepts)
            return_attention: if True, include attention maps in output
            
        Returns:
            Dictionary containing:
                - logits_organ: [B, 14] from global features
                - logits_mil: [B, 14] from region-concept alignment
                - region_tokens: [B, K, D] for downstream use
                - organ_token: [B, D] global representation
        """
        B = images.size(0)
        
        # Extract backbone features
        feat = self.backbone(images)[0]  # [B, C, H, W]
        _, C, H, W = feat.shape
        
        # Pixel tokens
        pixels = feat.permute(0, 2, 3, 1).reshape(B, H * W, C)
        pixels_proj = self.pixel_proj(pixels)  # [B, HW, D]
        
        # Region tokens via attention pooling
        region_tokens, region_attn = self.region_attn(pixels_proj)  # [B, K, D]
        
        # Organ token (global average)
        organ_feat = feat.mean(dim=(2, 3))  # [B, C]
        organ_token = self.organ_proj(organ_feat)  # [B, D]
        
        # Project concepts to shared space
        concept_embeddings = concept_embeddings.to(images.device)
        _, concept_proj = self.concept_proj(
            organ_token.unsqueeze(1), 
            concept_embeddings
        )
        
        # Split pathology concepts (after anatomy)
        pathology_concepts = concept_proj[self.num_anatomy:]  # [14, D]
        
        # Organ-level classification
        logits_organ = self.organ_classifier(organ_token)  # [B, 14]
        
        # MIL: max pooling over region-concept similarities
        region_norm = F.normalize(region_tokens, dim=-1)
        similarities = torch.einsum('bkd,cd->bkc', region_norm, pathology_concepts)
        logits_mil = similarities.max(dim=1).values * 10.0  # [B, 14], scaled for BCE
        
        output = {
            'logits_organ': logits_organ,
            'logits_mil': logits_mil,
            'region_tokens': region_tokens,
            'organ_token': organ_token,
        }
        
        if return_attention:
            output['region_attention'] = region_attn
            output['concept_similarities'] = similarities
        
        return output


def build_encoder(
    backbone: str = "convnext_tiny",
    pretrained: bool = True,
    num_regions: int = 8,
    proj_dim: int = 512,
    num_labels: int = 14,
) -> PROFA:
    """
    Factory function for PRO-FA encoder.
    
    Args:
        backbone: timm model name (convnext_tiny, convnext_small, etc.)
        pretrained: whether to load ImageNet weights
        num_regions: number of learnable region queries
        proj_dim: projection dimension for concept alignment
        num_labels: number of CheXpert pathology classes
        
    Returns:
        Configured PROFA module
    """
    return PROFA(
        backbone_name=backbone,
        pretrained=pretrained,
        num_regions=num_regions,
        proj_dim=proj_dim,
        num_labels=num_labels,
    )
