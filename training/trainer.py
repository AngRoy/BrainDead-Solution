"""
Training loop implementation for multi-view classification.

Supports:
- Progressive backbone unfreezing
- Mixed precision training
- Gradient accumulation
- Threshold tuning
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from sklearn.metrics import f1_score, average_precision_score
from typing import Optional

from .losses import MaskedBCELoss


class Trainer:
    """
    Training loop for multi-label classification.
    
    Handles mixed precision, gradient accumulation, and checkpoint saving.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        pos_weight: Optional[torch.Tensor] = None,
        device: str = "cuda",
        accumulation_steps: int = 1,
        checkpoint_dir: str = "checkpoints",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.accum_steps = accumulation_steps
        self.checkpoint_dir = checkpoint_dir
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.criterion = MaskedBCELoss(pos_weight.to(device) if pos_weight is not None else None)
        self.scaler = GradScaler('cuda')
        
        self.best_f1 = 0.0
        self.thresholds = np.full(14, 0.5)
    
    def train_epoch(self, epoch: int) -> dict:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        self.optimizer.zero_grad()
        progress = tqdm(self.train_loader, desc=f"Epoch {epoch} [train]")
        
        for step, (views, view_mask, targets, label_mask) in enumerate(progress):
            views = views.to(self.device)
            view_mask = view_mask.to(self.device)
            targets = targets.to(self.device)
            label_mask = label_mask.to(self.device)
            
            with autocast('cuda'):
                logits = self.model(views, view_mask)
                loss = self.criterion(logits, targets, label_mask)
                loss = loss / self.accum_steps
            
            self.scaler.scale(loss).backward()
            
            if (step + 1) % self.accum_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * self.accum_steps
            num_batches += 1
            progress.set_postfix(loss=total_loss / num_batches)
        
        if self.scheduler is not None:
            self.scheduler.step()
        
        return {"train_loss": total_loss / num_batches}
    
    @torch.no_grad()
    def validate(self, tune_thresholds: bool = True) -> dict:
        """
        Run validation and optionally tune thresholds.
        
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        
        all_probs = []
        all_targets = []
        all_masks = []
        total_loss = 0.0
        num_batches = 0
        
        for views, view_mask, targets, label_mask in tqdm(self.val_loader, desc="Validating"):
            views = views.to(self.device)
            view_mask = view_mask.to(self.device)
            targets = targets.to(self.device)
            label_mask = label_mask.to(self.device)
            
            with autocast('cuda'):
                logits = self.model(views, view_mask)
                loss = self.criterion(logits, targets, label_mask)
            
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu())
            all_targets.append(targets.cpu())
            all_masks.append(label_mask.cpu())
            total_loss += loss.item()
            num_batches += 1
        
        probs = torch.cat(all_probs).numpy()
        targets = torch.cat(all_targets).numpy()
        masks = torch.cat(all_masks).numpy()
        
        # Tune per-label thresholds
        if tune_thresholds:
            self.thresholds = tune_thresholds_per_label(probs, targets, masks)
        
        # Compute metrics with tuned thresholds
        preds = (probs > self.thresholds).astype(np.float32)
        
        metrics = compute_multilabel_metrics(preds, probs, targets, masks)
        metrics["val_loss"] = total_loss / num_batches
        
        # Save best model
        if metrics["macro_f1"] > self.best_f1:
            self.best_f1 = metrics["macro_f1"]
            self.save_checkpoint("best.pt")
        
        return metrics
    
    def save_checkpoint(self, name: str) -> None:
        """Save model checkpoint."""
        path = os.path.join(self.checkpoint_dir, name)
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "thresholds": self.thresholds,
            "best_f1": self.best_f1,
        }, path)
    
    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.thresholds = ckpt.get("thresholds", np.full(14, 0.5))
        self.best_f1 = ckpt.get("best_f1", 0.0)


def tune_thresholds_per_label(
    probs: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
    candidates: np.ndarray = None,
) -> np.ndarray:
    """
    Grid search for optimal per-label thresholds.
    
    Args:
        probs: [N, C] prediction probabilities
        targets: [N, C] ground truth
        masks: [N, C] valid labels
        candidates: threshold values to try
        
    Returns:
        best_thresholds: [C] optimal threshold per label
    """
    if candidates is None:
        candidates = np.arange(0.1, 0.9, 0.05)
    
    num_labels = probs.shape[1]
    best_thresholds = np.full(num_labels, 0.5)
    
    for c in range(num_labels):
        valid = masks[:, c] > 0
        if valid.sum() < 10:
            continue
        
        p = probs[valid, c]
        t = targets[valid, c]
        
        best_f1 = 0.0
        for thr in candidates:
            preds = (p > thr).astype(np.float32)
            f1 = f1_score(t, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresholds[c] = thr
    
    return best_thresholds


def compute_multilabel_metrics(
    preds: np.ndarray,
    probs: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
) -> dict:
    """
    Compute micro/macro F1 and mean AP.
    
    Only considers valid (unmasked) labels.
    """
    # Flatten valid entries for micro metrics
    valid = masks > 0
    flat_preds = preds[valid]
    flat_targets = targets[valid]
    flat_probs = probs[valid]
    
    micro_f1 = f1_score(flat_targets, flat_preds, average="micro", zero_division=0)
    
    # Per-label metrics for macro average
    per_label_f1 = []
    per_label_ap = []
    
    for c in range(preds.shape[1]):
        c_valid = masks[:, c] > 0
        if c_valid.sum() < 10:
            continue
        
        c_preds = preds[c_valid, c]
        c_targets = targets[c_valid, c]
        c_probs = probs[c_valid, c]
        
        f1 = f1_score(c_targets, c_preds, zero_division=0)
        per_label_f1.append(f1)
        
        if c_targets.sum() > 0:
            ap = average_precision_score(c_targets, c_probs)
            per_label_ap.append(ap)
    
    macro_f1 = np.mean(per_label_f1) if per_label_f1 else 0.0
    mean_ap = np.mean(per_label_ap) if per_label_ap else 0.0
    
    return {
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "mean_ap": mean_ap,
    }
