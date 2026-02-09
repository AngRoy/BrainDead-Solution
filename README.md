# CogRRG: Cognitive Radiology Report Generation

A cognitive framework for chest X-ray analysis with multi-label pathology classification.

```mermaid
graph TD
    %% Node Styles
    classDef tensor fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef module fill:#f3e5f5,stroke:#4a148c,stroke-width:2px,rx:10,ry:10;
    classDef process fill:#fff3e0,stroke:#e65100,stroke-width:2px;

    subgraph Inputs
        Img[Multi-View Images]:::tensor
        Ind[Clinical Indication]:::tensor
        Concepts[Concept Bank]:::tensor
    end

    subgraph "Phase 1: Perception (PRO-FA)"
        Backbone[ConvNeXt Backbone]:::module
        Img --> Backbone
        
        Pixel[Pixel Tokens]:::tensor
        Organ[Organ Token]:::tensor
        Backbone --> Pixel
        Backbone --> Organ
        
        RegionAttn(Region Attention):::process
        Pixel --> RegionAttn
        RegionAttn --> Region[Region Tokens]:::tensor
        
        Align(Concept Alignment):::process
        Concepts --> Align
        Region <--> Align
    end

    subgraph "Phase 2: Reasoning (MIX-MLP)"
        ViewAttn(View Fusion):::process
        Organ --> ViewAttn
        ViewAttn --> Fused[Fused Features]:::tensor
        
        Residual(Residual Path):::process
        Expansion(Expansion Path):::process
        
        Fused --> Residual
        Fused --> Expansion
        
        Add(Add & Norm):::process
        Residual --> Add
        Expansion --> Add
        
        ClassHead(Classification Head):::module
        Add --> ClassHead
        ClassHead --> Logits[Pathology Probabilities]:::tensor
    end

    subgraph "Phase 3: Verification (RCTA)"
        Prompter(Label Prompter):::process
        Logits --> Prompter
        Prompter --> LabelEmb[Label Embeddings]:::tensor
        
        CTA1(Step A: Context):::process
        Organ --> CTA1
        Ind --> CTA1
        
        CTA2(Step B: Hypothesis):::process
        CTA1 --> CTA2
        LabelEmb --> CTA2
        
        CTA3(Step C: Verify):::process
        CTA2 --> CTA3
        Pixel --> CTA3
        
        CTA3 --> Verified[Verified Features]:::tensor
    end

    subgraph Output
        Decoder(Report Decoder):::module
        Verified --> Decoder
        Decoder --> Report[Structured Report]:::tensor
    end

    %% Cross-phase connections
    Region -.->|MIL Supervision| ClassHead
```

## Overview

CogRRG implements a three-stage cognitive pipeline for CXR analysis:

1. **PRO-FA** - Progressive Region-Organ Feature Alignment for hierarchical visual perception
2. **MIX-MLP** - Dual-path classifier for multi-label pathology detection
3. **RCTA** - Recursive Closed-loop Triangular Attention for hypothesis verification

## Results

| Metric | Validation | Holdout |
|--------|------------|---------|
| Micro-F1 | 0.778 | 0.774 |
| Macro-F1 | 0.730 | 0.714 |
| Mean AP | 0.735 | 0.760 |

## Installation

```bash
git clone https://github.com/your-repo/BrainDead-Solution.git
cd BrainDead-Solution
pip install -r requirements.txt
```

## Project Structure

```
BrainDead-Solution/
├── data/
│   ├── dataset.py          # MIMIC-CXR data loading
│   ├── chexbert_labeler.py  # CheXpert label extraction
│   └── concepts.py          # Concept bank construction
├── models/
│   ├── encoder.py           # PRO-FA implementation
│   ├── classifier.py        # MIX-MLP with view attention
│   └── decoder.py           # RCTA verification module
├── training/
│   ├── losses.py            # Masked BCE and PRO-FA losses
│   └── trainer.py           # Training loop with AMP
├── evaluation/
│   └── chexpert_metrics.py  # Evaluation metrics
├── notebooks/
│   └── inference_demo.ipynb # Inference example
├── requirements.txt
└── README.md
```

## Quick Start

### Training

```python
from models import build_classifier
from data import create_dataloaders, subject_split, chexbert_to_binary
from training import Trainer

# Load data
train_df, val_df = subject_split(df, val_fraction=0.08)
targets, mask = chexbert_to_binary(train_df)
train_loader, val_loader = create_dataloaders(train_df, val_df, ...)

# Build model
model = build_classifier(backbone='convnext_tiny')

# Train
trainer = Trainer(model, train_loader, val_loader, optimizer)
for epoch in range(3):
    trainer.train_epoch(epoch)
    metrics = trainer.validate()
    print(f"Epoch {epoch}: macro-F1 = {metrics['macro_f1']:.4f}")
```

### Inference

```python
from models import build_classifier
import torch

model = build_classifier(pretrained=False)
model.load_state_dict(torch.load('checkpoints/best.pt')['model'])
model.eval()

# prediction
with torch.no_grad():
    logits = model(views, view_mask)
    probs = torch.sigmoid(logits)
```

## Training Details

- **Backbone**: ConvNeXt-Tiny (pretrained on ImageNet-1K)
- **Input**: 224×224 RGB, frontal + lateral views
- **Training**: Mixed precision, gradient accumulation
- **Hardware**: Single T4 GPU (16GB)
- **Labels**: 14 CheXpert pathologies via CheXbert weak supervision

### Training Phases

1. **Phase 1**: Smoke test on 12K samples (label validation)
2. **Phase 2**: Full training with progressive backbone unfreezing
3. **Phase 3**: PRO-FA concept alignment training

## Citation

```bibtex
@article{cogrrrg2024,
  title={CogRRG: A Cognitive Framework for Structured Chest X-Ray Report Generation},
  author={Anonymous},
  year={2024}
}
```

## License

MIT
