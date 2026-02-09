"""
RCTA: Recursive Closed-loop Triangular Attention

Implements hypothesis verification through a three-step attention loop:
1. Context Formation: align visual features with clinical indication
2. Hypothesis Formation: form diagnostic hypotheses from predicted labels
3. Verification: re-attend to fine-grained image regions for confirmation

This module ensures generated findings are grounded in actual image evidence,
reducing hallucination in report generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class TriangularAttention(nn.Module):
    """
    Three-step attention mechanism for hypothesis verification.
    
    Implements the cognitive cycle:
    image features -> clinical context -> diagnostic hypothesis -> verification
    
    Each step uses multi-head attention with residual connections.
    """
    
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        # Step A: Image-Indication contextualization
        self.ctx_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ctx_norm = nn.LayerNorm(dim)
        
        # Step B: Context-Label hypothesis formation
        self.hyp_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.hyp_norm = nn.LayerNorm(dim)
        
        # Step C: Hypothesis-Image verification
        self.ver_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ver_norm = nn.LayerNorm(dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        organ_tokens: torch.Tensor,
        indication_emb: torch.Tensor,
        label_emb: torch.Tensor,
        pixel_tokens: torch.Tensor,
        indication_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            organ_tokens: [B, 1, D] global visual representation
            indication_emb: [B, T, D] encoded clinical indication text
            label_emb: [B, 14, D] label prompt embeddings (from predicted labels)
            pixel_tokens: [B, HW, D] fine-grained spatial features
            indication_mask: [B, T] padding mask for indication (optional)
            
        Returns:
            verified: [B, D] verified representation for report generation
        """
        # Step A: Contextualize with clinical indication
        ctx, _ = self.ctx_attn(
            organ_tokens, indication_emb, indication_emb,
            key_padding_mask=indication_mask
        )
        ctx = self.ctx_norm(organ_tokens + self.dropout(ctx))  # [B, 1, D]
        
        # Step B: Form hypothesis from label embeddings
        hyp, _ = self.hyp_attn(ctx, label_emb, label_emb)
        hyp = self.hyp_norm(ctx + self.dropout(hyp))  # [B, 1, D]
        
        # Step C: Verify against fine-grained image evidence
        ver, _ = self.ver_attn(hyp, pixel_tokens, pixel_tokens)
        verified = self.ver_norm(hyp + self.dropout(ver))  # [B, 1, D]
        
        return verified.squeeze(1)  # [B, D]


class LabelPrompter(nn.Module):
    """
    Generates label prompt embeddings from predicted probabilities.
    
    Converts classifier predictions into structured text prompts like:
    "Pneumonia: present, confidence 0.87"
    "Pleural Effusion: absent"
    
    These prompts are encoded for use in hypothesis formation.
    """
    
    LABEL_NAMES = [
        "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
        "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
        "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices", "No Finding"
    ]
    
    def __init__(self, dim: int, num_labels: int = 14):
        super().__init__()
        self.num_labels = num_labels
        
        # Learnable embeddings for each label (base)
        self.label_emb = nn.Embedding(num_labels, dim)
        
        # Modulation for present/absent/confidence
        self.present_mod = nn.Linear(1, dim)
        self.absent_emb = nn.Parameter(torch.randn(dim) * 0.02)
        
        self.out_proj = nn.Linear(dim, dim)
    
    def forward(self, probs: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Args:
            probs: [B, 14] predicted probabilities
            threshold: classification threshold
            
        Returns:
            label_embeddings: [B, 14, D] structured label prompt embeddings
        """
        B = probs.size(0)
        device = probs.device
        
        # Base label embeddings
        indices = torch.arange(self.num_labels, device=device)
        base = self.label_emb(indices)  # [14, D]
        base = base.unsqueeze(0).expand(B, -1, -1)  # [B, 14, D]
        
        # Modulate by prediction
        is_present = (probs > threshold).float()  # [B, 14]
        present_mod = self.present_mod(probs.unsqueeze(-1))  # [B, 14, D]
        absent_mod = self.absent_emb.unsqueeze(0).unsqueeze(0).expand(B, self.num_labels, -1)
        
        # Combine: present labels get modulated, absent get fixed embedding
        modulated = is_present.unsqueeze(-1) * present_mod + (1 - is_present.unsqueeze(-1)) * absent_mod
        
        return self.out_proj(base + modulated)


class ReportDecoder(nn.Module):
    """
    Transformer decoder for structured report generation.
    
    Generates Findings and Impression sections conditioned on:
    - Verified visual evidence (from RCTA)
    - Label predictions (for consistency checking)
    
    Uses causal self-attention with cross-attention to verified features.
    """
    
    def __init__(
        self,
        vocab_size: int,
        dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        max_length: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_length, dim)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        self.output_proj = nn.Linear(dim, vocab_size)
        self.max_length = max_length
        self.dim = dim
    
    def forward(
        self,
        input_ids: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: [B, L] token indices
            memory: [B, M, D] encoded visual/verified features
            memory_mask: [B, M] attention mask for memory
            
        Returns:
            logits: [B, L, vocab_size] token prediction scores
        """
        B, L = input_ids.shape
        device = input_ids.device
        
        # Embeddings
        positions = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        
        # Causal mask for autoregressive generation
        causal_mask = nn.Transformer.generate_square_subsequent_mask(L, device=device)
        
        # Decode
        out = self.decoder(
            x, memory,
            tgt_mask=causal_mask,
            memory_key_padding_mask=memory_mask,
        )
        
        return self.output_proj(out)
    
    @torch.no_grad()
    def generate(
        self,
        memory: torch.Tensor,
        start_token: int,
        end_token: int,
        max_length: Optional[int] = None,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """
        Autoregressive generation with nucleus sampling.
        
        Args:
            memory: [B, M, D] conditioning features
            start_token: BOS token id
            end_token: EOS token id
            max_length: maximum tokens to generate
            temperature: sampling temperature
            top_p: nucleus sampling threshold
            
        Returns:
            generated: [B, L] generated token ids
        """
        B = memory.size(0)
        device = memory.device
        max_len = max_length or self.max_length
        
        generated = torch.full((B, 1), start_token, dtype=torch.long, device=device)
        
        for _ in range(max_len - 1):
            logits = self.forward(generated, memory)[:, -1, :]  # [B, vocab]
            logits = logits / temperature
            
            # Nucleus sampling
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            cumsum = torch.cumsum(probs, dim=-1)
            
            # Remove tokens outside nucleus
            mask = cumsum - probs > top_p
            sorted_logits[mask] = float('-inf')
            probs = F.softmax(sorted_logits, dim=-1)
            
            # Sample
            next_idx = torch.multinomial(probs, 1)
            next_token = torch.gather(sorted_idx, 1, next_idx)
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop if all sequences hit EOS
            if (next_token == end_token).all():
                break
        
        return generated


class RCTA(nn.Module):
    """
    Complete RCTA module combining all components.
    
    Integrates:
    - TriangularAttention for hypothesis verification
    - LabelPrompter for structured label encoding
    - ReportDecoder for text generation (if enabled)
    """
    
    def __init__(
        self,
        dim: int = 512,
        num_labels: int = 14,
        vocab_size: Optional[int] = None,
        enable_decoder: bool = False,
    ):
        super().__init__()
        
        self.triangular = TriangularAttention(dim)
        self.label_prompter = LabelPrompter(dim, num_labels)
        
        self.enable_decoder = enable_decoder
        if enable_decoder and vocab_size:
            self.decoder = ReportDecoder(vocab_size, dim)
    
    def forward(
        self,
        organ_tokens: torch.Tensor,
        pixel_tokens: torch.Tensor,
        label_probs: torch.Tensor,
        indication_emb: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Args:
            organ_tokens: [B, D] global visual representation
            pixel_tokens: [B, HW, D] fine-grained features
            label_probs: [B, 14] predicted label probabilities
            indication_emb: [B, T, D] optional indication encoding
            
        Returns:
            Dictionary with verified features and optional generation outputs
        """
        B, D = organ_tokens.shape
        
        # Default indication if not provided
        if indication_emb is None:
            indication_emb = organ_tokens.unsqueeze(1)  # [B, 1, D]
        
        # Generate label prompt embeddings
        label_emb = self.label_prompter(label_probs)  # [B, 14, D]
        
        # Run triangular verification
        verified = self.triangular(
            organ_tokens.unsqueeze(1),
            indication_emb,
            label_emb,
            pixel_tokens,
        )  # [B, D]
        
        return {
            'verified': verified,
            'label_embeddings': label_emb,
        }


def build_decoder(
    dim: int = 512,
    num_labels: int = 14,
    vocab_size: Optional[int] = None,
    enable_generation: bool = False,
) -> RCTA:
    """
    Factory function for RCTA module.
    
    Args:
        dim: feature dimension
        num_labels: number of pathology classes
        vocab_size: tokenizer vocabulary size (required if enable_generation=True)
        enable_generation: whether to include text decoder
        
    Returns:
        Configured RCTA module
    """
    return RCTA(
        dim=dim,
        num_labels=num_labels,
        vocab_size=vocab_size,
        enable_decoder=enable_generation,
    )
