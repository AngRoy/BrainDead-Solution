"""
Model package for CogRRG.

Contains:
- encoder: PRO-FA hierarchical visual perception
- classifier: MIX-MLP multi-label classification
- decoder: RCTA hypothesis verification
"""

from .encoder import PROFA, build_encoder
from .classifier import MultiViewClassifier, MixMLP, build_classifier
from .decoder import RCTA, build_decoder

__all__ = [
    "PROFA",
    "build_encoder",
    "MultiViewClassifier", 
    "MixMLP",
    "build_classifier",
    "RCTA",
    "build_decoder",
]
