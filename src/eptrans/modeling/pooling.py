"""Pooling heads for the psychrophile locality ablation.

Each pooling operator is an explicit, falsifiable hypothesis about how the
adaptive signal is distributed over a protein's residues:

  mean       signal is UNIFORM over residues. Near-sufficient for compositional
             adaptation (IVYWREL, charge fraction, E+K enrichment are residue
             averages) and it denoises -- expected best for thermophile.
  attention  signal location is LEARNABLE. Strictly subsumes mean (uniform
             attention weights recover it exactly), so with enough data it should
             not lose; its alpha is directly inspectable, which is what makes it
             evidence about active-site localisation rather than just a score.
  topk_mil   signal is in a FEW residues. Multiple-instance learning: score each
             residue, aggregate the top-k. Sharpest locality prior, and the most
             exposed to instance-level label noise (the label is genome-level, so
             every residue inherits it).

All three consume the SAME cached (B, K, H) top-k tensor from
09b_embed_perresidue.py, so an AUPRC delta between them is attributable to the
pooling operator and not to a different embedding pass.
"""
from __future__ import annotations


def build_pooling_head(kind: str, d_in: int, d_hidden: int = 512,
                       dropout: float = 0.1, mil_k: int = 8, attn_dim: int = 128):
    """Return an (nn.Module) head consuming (B, K, d_in) + mask (B, K).

    Args:
        kind: 'mean' | 'attention' | 'topk_mil'
        mil_k: for topk_mil, how many residue scores to average (k' <= K).
        attn_dim: bottleneck width of the gated-attention scorer.
    """
    import torch
    import torch.nn as nn

    class _Base(nn.Module):
        """Shared MLP readout; identical architecture to MeanPoolClassifierHead
        (Linear(d_in,512) -> GELU -> Dropout -> Linear(512,1)) so the ablation
        differs ONLY in how residues are combined."""

        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d_in, d_hidden), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(d_hidden, 1),
            )

        @staticmethod
        def _mask(x, mask):
            if mask is None:
                return torch.ones(x.shape[:2], dtype=x.dtype, device=x.device)
            return mask.to(x.dtype)

    class MeanOverK(_Base):
        def forward(self, x, mask=None):
            m = self._mask(x, mask).unsqueeze(-1)
            pooled = (x * m).sum(1) / m.sum(1).clamp_min(1.0)
            return self.net(pooled).squeeze(-1)

    class AttentionPool(_Base):
        """Gated attention pooling (Ilse et al. 2018 formulation):
            e_i = w^T (tanh(V h_i) * sigmoid(U h_i));  alpha = softmax(e)
            z   = sum_i alpha_i h_i
        The sigmoid gate lets the model suppress residues that merely have large
        tanh response, which matters here because ESM norms vary strongly by
        local environment."""

        def __init__(self):
            super().__init__()
            self.V = nn.Linear(d_in, attn_dim)
            self.U = nn.Linear(d_in, attn_dim)
            self.w = nn.Linear(attn_dim, 1)

        def alpha(self, x, mask=None):
            e = self.w(torch.tanh(self.V(x)) * torch.sigmoid(self.U(x))).squeeze(-1)
            m = self._mask(x, mask)
            e = e.masked_fill(m == 0, float("-inf"))
            a = torch.softmax(e, dim=1)
            return torch.nan_to_num(a)   # all-padding row -> zeros, not NaN

        def forward(self, x, mask=None):
            a = self.alpha(x, mask)
            z = (a.unsqueeze(-1) * x).sum(1)
            return self.net(z).squeeze(-1)

    class TopKMIL(_Base):
        """Per-residue scoring, then mean of the top-k' residue logits."""

        def forward(self, x, mask=None):
            s = self.net(x).squeeze(-1)                  # (B, K) per-residue logits
            m = self._mask(x, mask)
            s = s.masked_fill(m == 0, float("-inf"))
            k = min(mil_k, s.shape[1])
            top = s.topk(k, dim=1).values
            top = torch.where(torch.isfinite(top), top,
                              torch.zeros_like(top))     # short proteins
            denom = torch.isfinite(s).sum(1).clamp(1, k).to(top.dtype)
            return top.sum(1) / denom

    kinds = {"mean": MeanOverK, "attention": AttentionPool, "topk_mil": TopKMIL}
    if kind not in kinds:
        raise ValueError(f"unknown pooling kind {kind!r}; expected {sorted(kinds)}")
    return kinds[kind]()
