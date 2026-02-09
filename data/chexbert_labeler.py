"""
CheXbert labeler integration for extracting pathology labels from report text.

Uses the CheXbert model from Stanford AIMI to extract 14 CheXpert labels
from radiology report text with support for positive, negative, and uncertain mentions.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertModel, AutoTokenizer
from huggingface_hub import hf_hub_download
from typing import Optional


CHEXPERT_LABELS = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
    "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices", "No Finding"
]


class CheXbertModel(nn.Module):
    """
    CheXbert encoder for radiology report labeling.
    
    Wraps BERT with 14 classification heads (one per CheXpert label).
    Each head outputs class probabilities for blank/positive/negative/uncertain.
    """
    
    def __init__(self, head_dims: list, base_model: str = "bert-base-uncased"):
        super().__init__()
        self.bert = BertModel.from_pretrained(base_model)
        hidden = self.bert.config.hidden_size
        self.heads = nn.ModuleList([nn.Linear(hidden, d) for d in head_dims])
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> list[torch.Tensor]:
        """
        Args:
            input_ids: [B, L] tokenized text
            attention_mask: [B, L] attention mask
            
        Returns:
            List of [B, num_classes] logits for each label head
        """
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_emb = out.last_hidden_state[:, 0]  # [B, H]
        return [head(cls_emb) for head in self.heads]


def load_chexbert(device: torch.device) -> tuple[CheXbertModel, AutoTokenizer]:
    """
    Download and load CheXbert checkpoint from HuggingFace.
    
    Returns:
        model: Loaded CheXbert model
        tokenizer: BERT tokenizer
    """
    ckpt_path = hf_hub_download(repo_id="StanfordAIMI/RRG_scorers", filename="chexbert.pth")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    
    # Extract state dict
    state_dict = ckpt
    for key in ["state_dict", "model_state_dict", "model"]:
        if isinstance(state_dict, dict) and key in state_dict:
            state_dict = state_dict[key]
            break
    
    # Strip module prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    # Infer head dimensions
    head_dims = []
    for i in range(14):
        weight_key = f"linear_heads.{i}.weight"
        if weight_key in state_dict:
            head_dims.append(state_dict[weight_key].shape[0])
    
    model = CheXbertModel(head_dims=head_dims).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    return model, tokenizer


def class_id_to_value(class_id: int) -> float:
    """
    Convert CheXbert class prediction to label value.
    
    CheXbert outputs: 0=blank, 1=positive, 2=negative, 3=uncertain
    Returns: NaN (blank), 1.0 (positive), 0.0 (negative), -1.0 (uncertain)
    """
    mapping = {0: np.nan, 1: 1.0, 2: 0.0, 3: -1.0}
    return mapping.get(class_id, np.nan)


@torch.no_grad()
def label_reports(
    texts: list[str],
    model: CheXbertModel,
    tokenizer: AutoTokenizer,
    device: torch.device,
    batch_size: int = 32,
    max_length: int = 256,
) -> np.ndarray:
    """
    Extract CheXpert labels from report texts.
    
    Args:
        texts: list of report strings
        model: loaded CheXbert model
        tokenizer: BERT tokenizer
        device: computation device
        batch_size: batch size for inference
        max_length: max token length
        
    Returns:
        labels: [N, 14] array with values in {NaN, 0.0, 1.0, -1.0}
    """
    model.eval()
    all_labels = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        tokens = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        
        logits_list = model(**tokens)
        class_ids = np.stack([lg.argmax(dim=-1).cpu().numpy() for lg in logits_list], axis=1)
        
        # Convert to label values
        values = np.vectorize(class_id_to_value)(class_ids)
        all_labels.append(values)
    
    return np.concatenate(all_labels, axis=0)
