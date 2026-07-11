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


def lora_target_modules(full_attention: bool = True) -> list[str]:
    """ESM-2 attention module names LoRA adapts.

    Pairwise residue interactions (disulfides, salt bridges, local structure)
    live in the attention map, so adapting the FULL attention pathway — not just
    query/value — gives the fine-tuning room to re-route which residues attend to
    which (design doc Section 15 workaround #3). ``full_attention=True`` (default)
    adds ``key`` and the attention output projection ``dense``; set False for the
    lighter q/v-only set. HF ``EsmSelfAttention`` exposes ``.query/.key/.value``
    and ``EsmSelfOutput`` exposes ``.dense``.
    """
    if full_attention:
        # "attention.output.dense" targets ONLY the attention output projection,
        # not the two feed-forward "dense" layers (intermediate.dense/output.dense).
        return ["query", "key", "value", "attention.output.dense"]
    return ["query", "value"]


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


def build_lora_backbone(size: str = DEFAULT_BACKBONE, lora_rank: int = 32,
                        lora_alpha: int = 64, lora_dropout: float = 0.05,
                        for_mlm: bool = True, gradient_checkpointing: bool = True,
                        full_attention: bool = True):
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
        lora_dropout=lora_dropout,
        target_modules=lora_target_modules(full_attention=full_attention),
        bias="none",
    )
    model = get_peft_model(base, lconf)
    return model, tok, hidden


def add_classifier_head(hidden_size: int, **kwargs):
    """Convenience wrapper: build a classifier head for a given hidden size."""
    return build_classifier_head(hidden_size, **kwargs)


def load_mlm_adapter_into_classifier(clf_peft_model, adapter_dir: str,
                                     adapter_name: str = "mlm"):
    """Load a Stage-1 MLM LoRA adapter into a Stage-2 classifier backbone.

    The MLM adapter is trained on ``EsmForMaskedLM`` (keys namespaced
    ``base_model.model.esm.encoder...``) while the classifier backbone is a bare
    ``EsmModel`` (keys ``base_model.model.encoder...`` — no ``esm.`` prefix).
    A plain ``peft`` ``load_adapter`` does NOT raise on this mismatch: it creates
    the adapter slots but silently drops every non-matching weight, so the
    classifier would train from a randomly-initialised adapter and inherit NONE
    of the coupling-aware Stage-1 adaptation.

    This loader remaps the saved keys (strips the ``esm.`` submodule prefix),
    loads them into a freshly-added adapter, and VERIFIES that a non-trivial
    fraction of LoRA tensors actually received the trained values — raising if
    the transfer was empty. Returns the number of LoRA weight tensors populated.
    """
    import glob
    import torch
    from safetensors.torch import load_file

    files = (glob.glob(f"{adapter_dir}/*.safetensors")
             or glob.glob(f"{adapter_dir}/adapter_model.bin"))
    if not files:
        raise FileNotFoundError(f"no adapter weights found under {adapter_dir}")
    saved = (load_file(files[0]) if files[0].endswith(".safetensors")
             else torch.load(files[0], map_location="cpu"))

    # add an (empty) adapter with the same config, then overwrite its weights
    clf_peft_model.load_adapter(adapter_dir, adapter_name=adapter_name)
    target = dict(clf_peft_model.named_parameters())

    def _remap(k: str) -> str:
        # base_model.model.esm.encoder...  ->  base_model.model.encoder...
        k = k.replace("base_model.model.esm.", "base_model.model.")
        # peft stores runtime weights with the active adapter name in the path
        k = k.replace(".lora_A.weight", f".lora_A.{adapter_name}.weight")
        k = k.replace(".lora_B.weight", f".lora_B.{adapter_name}.weight")
        return k

    n_ok, n_miss = 0, 0
    with torch.no_grad():
        for sk, sv in saved.items():
            tk = _remap(sk)
            if tk in target and target[tk].shape == sv.shape:
                target[tk].copy_(sv.to(target[tk].dtype))
                n_ok += 1
            else:
                n_miss += 1
    if n_ok == 0:
        raise RuntimeError(
            f"MLM->classifier adapter transfer matched 0 tensors "
            f"({n_miss} unmatched) — key remap failed; classifier would train "
            f"from a random adapter. Check the esm.-prefix convention.")
    print(f"[adapter] transferred {n_ok} LoRA tensors "
          f"({n_miss} unmatched) from {adapter_dir}")
    return n_ok
