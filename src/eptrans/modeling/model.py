"""LoRA-adapted ESM-2 backbone with two heads (design doc Section 11).

One shared foundation, two heads:
  - Head 1 (generation): the MLM head (ESM-2's own lm_head) — mask-fill +
    pseudo-likelihood re-ranker.
  - Head 2 (scoring): a per-phenotype classifier head (mean-pool -> 2-layer MLP)
    on the same adapted backbone.

The backbone is ESM-2, adapted with LoRA (rank 8-16 on q/v proj), base frozen.
Multiple phenotype adapters stay swappable on one backbone (no N x full copies).

Config note: the doc's §11 body settled on ESM-2 650M for ecosystem maturity,
but the user explicitly chose the full 3B (t36) — resources available. The
default here is 3B; ``ESM2_CHECKPOINTS`` maps sizes to HF ids so a smaller model
can be used for CPU smoke-tests (35M) without touching call sites.

Heavy deps (torch/transformers/peft) are imported lazily inside the factory so
this module imports on a torch-less box; the pure classifier-head class is
defined via a factory to keep module import light.
"""
from __future__ import annotations

__all__ = [
    "ESM2_CHECKPOINTS",
    "DEFAULT_BACKBONE",
    "build_classifier_head",
    "build_lora_backbone",
    "add_classifier_head",
    "lora_target_modules",
]

# HF checkpoint ids by ESM-2 size. 3B (t36) is the chosen production backbone;
# 35M/150M are for fast CPU smoke-tests of the scaffold.
ESM2_CHECKPOINTS = {
    "8M": "facebook/esm2_t6_8M_UR50D",
    "35M": "facebook/esm2_t12_35M_UR50D",
    "150M": "facebook/esm2_t30_150M_UR50D",
    "650M": "facebook/esm2_t33_650M_UR50D",
    "3B": "facebook/esm2_t36_3B_UR50D",
}
DEFAULT_BACKBONE = "3B"


def lora_target_modules() -> list[str]:
    """ESM-2 attention projection module names LoRA adapts (q_proj / v_proj)."""
    return ["query", "value"]  # HF EsmSelfAttention uses .query/.key/.value


def build_classifier_head(hidden_size: int, n_hidden: int = 512, dropout: float = 0.1):
    """2-layer MLP classifier head on a mean-pooled backbone representation.

    Returns an ``nn.Module`` producing a single logit per sequence. Mean-pooling
    masks out pad/special tokens via the attention mask.
    """
    import torch
    import torch.nn as nn

    class MeanPoolClassifierHead(nn.Module):
        def __init__(self, d_in, d_hidden, p):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d_in, d_hidden), nn.GELU(), nn.Dropout(p),
                nn.Linear(d_hidden, 1),
            )

        def forward(self, hidden_states, attention_mask=None):
            # hidden_states: (B, L, d); attention_mask: (B, L) 1=token 0=pad
            if attention_mask is not None:
                m = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
                pooled = (hidden_states * m).sum(1) / m.sum(1).clamp_min(1.0)
            else:
                pooled = hidden_states.mean(1)
            return self.net(pooled).squeeze(-1)  # (B,)

    return MeanPoolClassifierHead(hidden_size, n_hidden, dropout)


def build_lora_backbone(size: str = DEFAULT_BACKBONE, lora_rank: int = 16,
                        lora_alpha: int = 32, lora_dropout: float = 0.05,
                        for_mlm: bool = True, gradient_checkpointing: bool = True):
    """Load ESM-2 (``size``) and wrap it with a LoRA adapter (base frozen).

    Args:
        size: key into ``ESM2_CHECKPOINTS`` ('3B' production, '35M' smoke-test).
        for_mlm: True -> EsmForMaskedLM (generation head + backbone); False ->
            bare EsmModel (encoder only, for the classifier branch).
        gradient_checkpointing: enable to fit 3B on a single GPU (bf16).

    Returns:
        ``(peft_model, tokenizer, hidden_size)``.
    """
    import torch
    from transformers import AutoTokenizer, EsmForMaskedLM, EsmModel
    from peft import LoraConfig, get_peft_model, TaskType

    ckpt = ESM2_CHECKPOINTS[size]
    tok = AutoTokenizer.from_pretrained(ckpt)
    if for_mlm:
        base = EsmForMaskedLM.from_pretrained(ckpt)
        task = TaskType.FEATURE_EXTRACTION  # we drive MLM loss manually
    else:
        base = EsmModel.from_pretrained(ckpt, add_pooling_layer=False)
        task = TaskType.FEATURE_EXTRACTION
    hidden = base.config.hidden_size
    if gradient_checkpointing and hasattr(base, "gradient_checkpointing_enable"):
        base.gradient_checkpointing_enable()

    lconf = LoraConfig(
        task_type=task, r=lora_rank, lora_alpha=lora_alpha,
        lora_dropout=lora_dropout, target_modules=lora_target_modules(),
        bias="none",
    )
    model = get_peft_model(base, lconf)
    return model, tok, hidden


def add_classifier_head(hidden_size: int, **kwargs):
    """Convenience wrapper: build a classifier head for a given hidden size."""
    return build_classifier_head(hidden_size, **kwargs)
