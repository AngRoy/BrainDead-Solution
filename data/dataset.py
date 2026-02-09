"""
Data loading and preprocessing utilities for MIMIC-CXR.

Handles:
- CheXbert weak label extraction
- Multi-view dataset construction
- Label preprocessing with uncertainty handling
"""

import os
import ast
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Callable
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


CHEXPERT_LABELS = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
    "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices", "No Finding"
]


def parse_list_field(value) -> list:
    """
    Parse stringified Python list from CSV field.
    
    Handles various edge cases: None, NaN, empty strings, malformed lists.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        return value
    
    s = str(value).strip()
    if s in ("", "[]", "nan", "None"):
        return []
    
    try:
        parsed = ast.literal_eval(s)
        return parsed if isinstance(parsed, list) else []
    except (ValueError, SyntaxError):
        return []


def find_existing_path(rel_paths: list, base_dir: str) -> Optional[str]:
    """Return first existing path from list of relative paths."""
    for rel in rel_paths:
        if rel is None:
            continue
        rel = str(rel).strip()
        if not rel:
            continue
        full = os.path.join(base_dir, rel)
        if os.path.exists(full):
            return full
    return None


def build_image_paths(df: pd.DataFrame, base_dir: str) -> pd.DataFrame:
    """
    Add best_frontal_path and best_lateral_path columns.
    
    Prioritizes PA over AP for frontal views.
    """
    frontal_paths = []
    lateral_paths = []
    
    for _, row in df.iterrows():
        pa = parse_list_field(row.get("PA"))
        ap = parse_list_field(row.get("AP"))
        lat = parse_list_field(row.get("Lateral"))
        
        # Prefer PA, fall back to AP
        front = find_existing_path(pa, base_dir) or find_existing_path(ap, base_dir)
        lateral = find_existing_path(lat, base_dir)
        
        frontal_paths.append(front)
        lateral_paths.append(lateral)
    
    df = df.copy()
    df["best_frontal_path"] = frontal_paths
    df["best_lateral_path"] = lateral_paths
    return df


def chexbert_to_binary(
    df: pd.DataFrame,
    labels: list = CHEXPERT_LABELS,
    policy: str = "u_ones"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert CheXbert labels to binary targets with validity mask.
    
    Args:
        df: DataFrame with label columns
        labels: list of label column names
        policy: how to handle uncertain labels
               "u_ones" = treat uncertain as positive (default)
               "u_zeros" = treat uncertain as negative
               "u_ignore" = mask uncertain labels
    
    Returns:
        targets: [N, 14] binary labels
        mask: [N, 14] validity mask (1 = valid, 0 = ignore)
    """
    n = len(df)
    num_labels = len(labels)
    targets = np.zeros((n, num_labels), dtype=np.float32)
    mask = np.ones((n, num_labels), dtype=np.float32)
    
    for j, label in enumerate(labels):
        if label not in df.columns:
            mask[:, j] = 0.0
            continue
        
        values = pd.to_numeric(df[label], errors="coerce").to_numpy()
        
        # Handle NaN (not mentioned in report)
        is_nan = np.isnan(values)
        mask[is_nan, j] = 0.0
        
        # Convert to binary based on policy
        values_clean = np.nan_to_num(values, nan=0.0)
        
        if policy == "u_ones":
            # Positive (1) and uncertain (-1) both become 1
            targets[:, j] = ((values_clean == 1.0) | (values_clean == -1.0)).astype(np.float32)
        elif policy == "u_zeros":
            # Only positive (1) becomes 1
            targets[:, j] = (values_clean == 1.0).astype(np.float32)
        elif policy == "u_ignore":
            # Positive = 1, uncertain = masked
            targets[:, j] = (values_clean == 1.0).astype(np.float32)
            mask[values_clean == -1.0, j] = 0.0
        else:
            raise ValueError(f"Unknown policy: {policy}")
    
    return targets, mask


def compute_class_weights(targets: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Compute inverse-frequency class weights for imbalanced labels.
    
    Returns weight for positive class (for use with pos_weight in BCE).
    """
    eps = 1e-6
    pos_count = (targets * mask).sum(axis=0)
    neg_count = ((1 - targets) * mask).sum(axis=0)
    return neg_count / (pos_count + eps)


class MIMICDataset(Dataset):
    """
    MIMIC-CXR dataset for multi-view classification.
    
    Returns frontal and lateral views (if available) with labels.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        targets: np.ndarray,
        mask: np.ndarray,
        transform: Optional[Callable] = None,
        img_size: int = 224,
    ):
        self.df = df.reset_index(drop=True)
        self.targets = torch.from_numpy(targets).float()
        self.mask = torch.from_numpy(mask).float()
        self.img_size = img_size
        
        if transform is None:
            self.transform = T.Compose([
                T.Resize((img_size, img_size)),
                T.ToTensor(),
            ])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def _load_image(self, path: Optional[str]) -> Optional[torch.Tensor]:
        if path is None or not os.path.exists(path):
            return None
        img = Image.open(path).convert("RGB")
        return self.transform(img)
    
    def __getitem__(self, idx: int) -> tuple:
        row = self.df.iloc[idx]
        
        frontal = self._load_image(row.get("best_frontal_path"))
        lateral = self._load_image(row.get("best_lateral_path"))
        
        # Stack views [2, C, H, W]
        views = []
        view_mask = []
        
        if frontal is None:
            frontal = torch.zeros(3, self.img_size, self.img_size)
            view_mask.append(0.0)
        else:
            view_mask.append(1.0)
        views.append(frontal)
        
        if lateral is None:
            lateral = torch.zeros(3, self.img_size, self.img_size)
            view_mask.append(0.0)
        else:
            view_mask.append(1.0)
        views.append(lateral)
        
        return (
            torch.stack(views),           # [2, 3, H, W]
            torch.tensor(view_mask),      # [2]
            self.targets[idx],            # [14]
            self.mask[idx],               # [14]
        )


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    train_targets: np.ndarray,
    train_mask: np.ndarray,
    val_targets: np.ndarray,
    val_mask: np.ndarray,
    batch_size: int = 32,
    num_workers: int = 2,
    img_size: int = 224,
) -> tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Returns:
        train_loader, val_loader
    """
    train_ds = MIMICDataset(train_df, train_targets, train_mask, img_size=img_size)
    val_ds = MIMICDataset(val_df, val_targets, val_mask, img_size=img_size)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader


def subject_split(
    df: pd.DataFrame,
    val_fraction: float = 0.08,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset by subject to avoid data leakage.
    
    Studies from the same patient always go to the same split.
    """
    subjects = df["subject_id"].unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(subjects)
    
    n_val = int(len(subjects) * val_fraction)
    val_subjects = set(subjects[:n_val])
    
    train_df = df[~df["subject_id"].isin(val_subjects)].reset_index(drop=True)
    val_df = df[df["subject_id"].isin(val_subjects)].reset_index(drop=True)
    
    return train_df, val_df
