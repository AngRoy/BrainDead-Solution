"""
Concept bank construction for PRO-FA alignment.

Creates BioClinicalBERT embeddings for anatomical and pathological concepts
used in the concept alignment training objective.
"""

import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import Optional


ANATOMY_TERMS = [
    "lung", "left lung", "right lung", "upper lobe", "lower lobe", "middle lobe",
    "pleura", "costophrenic angle", "diaphragm", "heart", "cardiomediastinal silhouette",
    "mediastinum", "aorta", "hilum", "rib", "spine", "clavicle"
]

PATHOLOGY_TERMS = [
    "enlarged cardiomediastinum", "cardiomegaly", "lung opacity", "lung lesion",
    "edema", "consolidation", "pneumonia", "atelectasis", "pneumothorax",
    "pleural effusion", "pleural other", "fracture", "support devices", "no finding"
]


def build_concept_prompts() -> tuple[list[str], list[str]]:
    """
    Create natural language prompts for each concept.
    
    Format: "anatomy: {term}" or "finding: {term}"
    
    Returns:
        texts: list of prompt strings
        types: list of concept types ("anatomy" or "pathology")
    """
    texts = []
    types = []
    
    for term in ANATOMY_TERMS:
        texts.append(f"anatomy: {term}")
        types.append("anatomy")
    
    for term in PATHOLOGY_TERMS:
        texts.append(f"finding: {term}")
        types.append("pathology")
    
    return texts, types


@torch.no_grad()
def encode_concepts(
    device: torch.device,
    model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
    batch_size: int = 32,
) -> torch.Tensor:
    """
    Encode concept prompts using BioClinicalBERT.
    
    Returns:
        embeddings: [num_concepts, 768] L2-normalized concept embeddings
    """
    texts, _ = build_concept_prompts()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        tokens = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=32,
        ).to(device)
        
        out = model(**tokens)
        cls_emb = out.last_hidden_state[:, 0]  # [B, 768]
        cls_emb = F.normalize(cls_emb, dim=-1)
        embeddings.append(cls_emb.cpu())
    
    return torch.cat(embeddings, dim=0)


def save_concept_bank(output_path: str, embeddings: torch.Tensor) -> None:
    """
    Save concept bank to disk.
    
    Saves both JSON metadata and embeddings tensor.
    """
    texts, types = build_concept_prompts()
    
    metadata = {
        "concept_texts": texts,
        "concept_types": types,
        "num_anatomy": len(ANATOMY_TERMS),
        "num_pathology": len(PATHOLOGY_TERMS),
    }
    
    with open(output_path.replace(".pt", ".json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    torch.save(embeddings, output_path)


def load_concept_bank(path: str) -> tuple[torch.Tensor, dict]:
    """
    Load precomputed concept embeddings.
    
    Args:
        path: path to .pt file
        
    Returns:
        embeddings: [num_concepts, 768] tensor
        metadata: concept metadata dict
    """
    embeddings = torch.load(path, map_location="cpu")
    
    json_path = path.replace(".pt", ".json")
    with open(json_path) as f:
        metadata = json.load(f)
    
    return embeddings, metadata
