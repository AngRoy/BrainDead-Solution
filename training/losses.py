"""
Loss functions for multi-label classification.

Implements masked BCE loss to handle missing labels, with optional
class weighting for imbalanced datasets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MaskedBCELoss(nn.Module):
    """
    Binary cross-entropy with label masking.
    
    Handles datasets where some labels are missing (NaN in CheXbert output).
    Loss is computed only for valid label positions.
    """
    
    def __init__(self, pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: [B, C] raw predictions
            targets: [B, C] binary labels
            mask: [B, C] validity mask (1=valid, 0=ignore)
            
        Returns:
            Scalar loss averaged over valid positions
        """
        if self.pos_weight is not None:
            loss = F.binary_cross_entropy_with_logits(
                logits, targets,
                pos_weight=self.pos_weight,
                reduction="none"
            )
        else:
            loss = F.binary_cross_entropy_with_logits(
                logits, targets, reduction="none"
            )
        
        masked_loss = loss * mask
        return masked_loss.sum() / (mask.sum() + 1e-8)


class PROFALoss(nn.Module):
    """
    Combined loss for PRO-FA training.
    
    Components:
    - Organ-level classification loss (from global features)
    - MIL loss (from region-concept alignment)
    - Attention entropy regularization (prevents degenerate attention)
    """
    
    def __init__(
        self,
        pos_weight: Optional[torch.Tensor] = None,
        organ_weight: float = 0.5,
        mil_weight: float = 0.5,
        entropy_weight: float = 0.01,
    ):
        super().__init__()
        self.organ_loss = MaskedBCELoss(pos_weight)
        self.mil_loss = MaskedBCELoss(pos_weight)
        
        self.organ_weight = organ_weight
        self.mil_weight = mil_weight
        self.entropy_weight = entropy_weight
    
    def forward(
        self,
        organ_logits: torch.Tensor,
        mil_logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
        region_attention: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Args:
            organ_logits: [B, C] from global pooling
            mil_logits: [B, C] from MIL aggregation
            targets: [B, C] binary labels
            mask: [B, C] valid labels
            region_attention: [B, K, HW] attention weights (optional)
            
        Returns:
            Dictionary with total loss and component breakdown
        """
        l_organ = self.organ_loss(organ_logits, targets, mask)
        l_mil = self.mil_loss(mil_logits, targets, mask)
        
        l_entropy = torch.tensor(0.0, device=organ_logits.device)
        if region_attention is not None:
            # Encourage peaked attention (low entropy = confident localization)
            eps = 1e-8
            p = region_attention.clamp(min=eps)
            entropy = -(p * p.log()).sum(dim=-1).mean()
            l_entropy = -entropy  # Maximize negative entropy (minimize entropy)
        
        total = (
            self.organ_weight * l_organ +
            self.mil_weight * l_mil +
            self.entropy_weight * l_entropy
        )
        
        return {
            "loss": total,
            "organ_loss": l_organ.detach(),
            "mil_loss": l_mil.detach(),
            "entropy_loss": l_entropy.detach(),
        }
