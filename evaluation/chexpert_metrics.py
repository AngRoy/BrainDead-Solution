"""
Evaluation metrics for CheXpert multi-label classification.

Provides comprehensive metrics including per-label breakdown,
confidence calibration, and threshold-sensitivity analysis.
"""

import numpy as np
from sklearn.metrics import (
    f1_score, 
    precision_score, 
    recall_score, 
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
)
from typing import Optional


CHEXPERT_LABELS = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
    "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices", "No Finding"
]


def compute_metrics(
    probs: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> dict:
    """
    Compute classification metrics from predictions.
    
    Args:
        probs: [N, 14] prediction probabilities
        targets: [N, 14] ground truth labels
        masks: [N, 14] valid label mask
        thresholds: [14] per-label thresholds (default 0.5)
        
    Returns:
        Dictionary with aggregate and per-label metrics
    """
    if thresholds is None:
        thresholds = np.full(14, 0.5)
    
    preds = (probs > thresholds).astype(np.float32)
    num_labels = probs.shape[1]
    
    # Aggregate metrics (over all valid entries)
    valid = masks > 0
    flat_preds = preds[valid]
    flat_targets = targets[valid]
    flat_probs = probs[valid]
    
    aggregate = {
        "micro_f1": f1_score(flat_targets, flat_preds, average="micro", zero_division=0),
        "micro_precision": precision_score(flat_targets, flat_preds, average="micro", zero_division=0),
        "micro_recall": recall_score(flat_targets, flat_preds, average="micro", zero_division=0),
    }
    
    # Per-label breakdown
    per_label = {}
    auc_list = []
    ap_list = []
    f1_list = []
    
    for c in range(num_labels):
        label_name = CHEXPERT_LABELS[c]
        c_valid = masks[:, c] > 0
        
        if c_valid.sum() < 10:
            per_label[label_name] = {"valid_samples": int(c_valid.sum()), "skipped": True}
            continue
        
        c_preds = preds[c_valid, c]
        c_targets = targets[c_valid, c]
        c_probs = probs[c_valid, c]
        
        c_f1 = f1_score(c_targets, c_preds, zero_division=0)
        c_precision = precision_score(c_targets, c_preds, zero_division=0)
        c_recall = recall_score(c_targets, c_preds, zero_division=0)
        
        f1_list.append(c_f1)
        
        # AUC and AP need positive samples
        c_auc = 0.0
        c_ap = 0.0
        if c_targets.sum() > 0 and c_targets.sum() < len(c_targets):
            c_auc = roc_auc_score(c_targets, c_probs)
            c_ap = average_precision_score(c_targets, c_probs)
            auc_list.append(c_auc)
            ap_list.append(c_ap)
        
        per_label[label_name] = {
            "threshold": float(thresholds[c]),
            "f1": float(c_f1),
            "precision": float(c_precision),
            "recall": float(c_recall),
            "auc": float(c_auc),
            "ap": float(c_ap),
            "prevalence": float(c_targets.mean()),
            "valid_samples": int(c_valid.sum()),
        }
    
    # Macro averages
    aggregate["macro_f1"] = float(np.mean(f1_list)) if f1_list else 0.0
    aggregate["mean_auc"] = float(np.mean(auc_list)) if auc_list else 0.0
    aggregate["mean_ap"] = float(np.mean(ap_list)) if ap_list else 0.0
    
    return {
        "aggregate": aggregate,
        "per_label": per_label,
    }


def format_metrics_table(metrics: dict) -> str:
    """
    Format metrics as a human-readable table.
    
    Args:
        metrics: output from compute_metrics
        
    Returns:
        Formatted string table
    """
    lines = []
    lines.append("=" * 80)
    lines.append("Aggregate Metrics")
    lines.append("-" * 40)
    
    agg = metrics["aggregate"]
    lines.append(f"  Micro-F1:    {agg['micro_f1']:.4f}")
    lines.append(f"  Macro-F1:    {agg['macro_f1']:.4f}")
    lines.append(f"  Mean AUC:    {agg['mean_auc']:.4f}")
    lines.append(f"  Mean AP:     {agg['mean_ap']:.4f}")
    
    lines.append("")
    lines.append("Per-Label Breakdown")
    lines.append("-" * 80)
    header = f"{'Label':<30} {'Thr':>5} {'F1':>6} {'Prec':>6} {'Rec':>6} {'AUC':>6} {'Prev':>6}"
    lines.append(header)
    lines.append("-" * 80)
    
    for label in CHEXPERT_LABELS:
        if label not in metrics["per_label"]:
            continue
        
        m = metrics["per_label"][label]
        if m.get("skipped"):
            lines.append(f"{label:<30} {'--':>5} {'--':>6} {'--':>6} {'--':>6} {'--':>6} {'--':>6}")
        else:
            lines.append(
                f"{label:<30} {m['threshold']:>5.2f} {m['f1']:>6.3f} {m['precision']:>6.3f} "
                f"{m['recall']:>6.3f} {m['auc']:>6.3f} {m['prevalence']:>6.2f}"
            )
    
    lines.append("=" * 80)
    return "\n".join(lines)
