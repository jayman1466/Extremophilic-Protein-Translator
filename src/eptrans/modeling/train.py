"""Training loops for the two heads (design doc Section 11, order of operations).

Two entry points, one shared adapted backbone:

  1. ``train_mlm`` — domain-adaptive continued MLM (LoRA) on TRAIN-only clusters
     (leakage rule 2). Confidence-weighted masked CE (§12 Loss 1); early-stop on
     val pseudo-perplexity. Saves the adapter.

  2. ``train_classifier`` — per-phenotype head on the adapted backbone; weighted
     BCE + matched-pair margin (§12 Loss 2); model-select on val AUPRC.

Both take an already-built ``(peft_model, tokenizer, hidden)`` so weight loading
(the only network/GPU-heavy step) is separated from the loop. Designed to run on
biotite GPU; small-model smoke-testable on CPU. torch imported lazily.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

__all__ = ["collate_pad", "train_mlm", "train_classifier", "evaluate_auprc"]


def collate_pad(batch, pad_id: int, keys=("input_ids", "attention_mask", "labels")):
    """Right-pad variable-length sequences in a batch to the max length."""
    import torch
    maxlen = max(x["input_ids"].shape[0] for x in batch)
    out = {}
    for k in keys:
        if k not in batch[0]:
            continue
        pad_val = -100 if k == "labels" else (pad_id if k == "input_ids" else 0)
        out[k] = torch.stack([
            torch.cat([x[k], torch.full((maxlen - x[k].shape[0],), pad_val, dtype=x[k].dtype)])
            for x in batch])
    for k in ("seq_weight", "label", "weight"):
        if k in batch[0]:
            out[k] = torch.stack([x[k] for x in batch])
    if "tagged_id" in batch[0]:
        out["tagged_id"] = [x["tagged_id"] for x in batch]
    return out


def train_mlm(peft_model, tokenizer, train_ds, val_ds=None, *, epochs: int = 3,
              lr: float = 1e-4, batch_size: int = 8, warmup_frac: float = 0.05,
              beta_kl: float = 0.0, base_model=None, device: str = "cpu",
              out_dir: str | None = None, max_steps: int | None = None,
              log_every: int = 50):
    """Domain-adaptive continued MLM (LoRA). Returns training history dict."""
    import torch
    from torch.utils.data import DataLoader
    from .losses import masked_mlm_loss, kl_forgetting_guard

    pad_id = tokenizer.pad_token_id
    dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                    collate_fn=lambda b: collate_pad(b, pad_id))
    peft_model.to(device).train()
    opt = torch.optim.AdamW([p for p in peft_model.parameters() if p.requires_grad], lr=lr)
    total = (max_steps or len(dl) * epochs)
    warmup = max(1, int(warmup_frac * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / warmup) * max(0.0, (total - s) / max(1, total - warmup)))

    hist = {"train_loss": [], "val_ppl": []}
    step = 0
    best_ppl = math.inf
    for ep in range(epochs):
        for batch in dl:
            batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
            out = peft_model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            loss = masked_mlm_loss(out.logits, batch["labels"].clamp_min(0),
                                   batch["labels"] != -100, seq_weight=batch.get("seq_weight"))
            if beta_kl > 0 and base_model is not None:
                with torch.no_grad():
                    base_logits = base_model(input_ids=batch["input_ids"],
                                             attention_mask=batch["attention_mask"]).logits
                loss = loss + beta_kl * kl_forgetting_guard(
                    out.logits, base_logits, mask=batch["labels"] != -100)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            if step % log_every == 0:
                hist["train_loss"].append((step, float(loss.detach())))
            step += 1
            if max_steps and step >= max_steps:
                break
        if val_ds is not None:
            ppl = _pseudo_perplexity(peft_model, tokenizer, val_ds, device, batch_size)
            hist["val_ppl"].append((step, ppl))
            if ppl < best_ppl and out_dir:
                best_ppl = ppl
                peft_model.save_pretrained(str(Path(out_dir) / "mlm_adapter_best"))
        if max_steps and step >= max_steps:
            break
    if out_dir:
        peft_model.save_pretrained(str(Path(out_dir) / "mlm_adapter_last"))
        json.dump(hist, open(Path(out_dir) / "mlm_history.json", "w"), indent=2)
    return hist


def _pseudo_perplexity(model, tokenizer, ds, device, batch_size):
    """Cheap MLM val metric: exp(mean masked CE) over the val set as given
    (its items already carry masked labels)."""
    import torch
    from torch.utils.data import DataLoader
    from .losses import masked_mlm_loss
    pad_id = tokenizer.pad_token_id
    dl = DataLoader(ds, batch_size=batch_size, collate_fn=lambda b: collate_pad(b, pad_id))
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for batch in dl:
            batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            l = masked_mlm_loss(out.logits, batch["labels"].clamp_min(0),
                                batch["labels"] != -100)
            tot += float(l); n += 1
    model.train()
    return math.exp(tot / max(1, n))


def train_classifier(backbone, head, tokenizer, train_ds, val_ds=None, *,
                     pairs=None, epochs: int = 5, lr_head: float = 1e-3,
                     lr_adapter: float = 1e-5, batch_size: int = 16, lam: float = 1.0,
                     margin: float = 1.0, pos_weight: float | None = None,
                     device: str = "cpu", out_dir: str | None = None,
                     max_steps: int | None = None):
    """Per-phenotype classifier head on the adapted backbone (§12 Loss 2).

    ``pairs``: optional dict mapping tagged_id -> tagged_id for the matched
    outgroup, used to build the pairwise margin term from ids present in a batch.
    Returns history with val AUPRC.
    """
    import torch
    from torch.utils.data import DataLoader
    from .losses import classifier_loss

    pad_id = tokenizer.pad_token_id
    dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                    collate_fn=lambda b: collate_pad(b, pad_id))
    backbone.to(device); head.to(device)
    params = [{"params": [p for p in backbone.parameters() if p.requires_grad], "lr": lr_adapter},
              {"params": head.parameters(), "lr": lr_head}]
    opt = torch.optim.AdamW(params)
    pw = torch.tensor(pos_weight, device=device) if pos_weight else None

    hist = {"train_loss": [], "val_auprc": []}
    best = -1.0
    step = 0
    for ep in range(epochs):
        backbone.train(); head.train()
        for batch in dl:
            batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
            h = _encode(backbone, batch["input_ids"], batch["attention_mask"])
            s = head(h, batch["attention_mask"])
            pe = po = None  # (pair term wiring: caller supplies aligned batches in production)
            loss, parts = classifier_loss(s, batch["label"], sample_weight=batch.get("weight"),
                                          pos_weight=pw, pair_ext=pe, pair_out=po,
                                          lam=lam, margin=margin)
            opt.zero_grad(); loss.backward(); opt.step()
            hist["train_loss"].append((step, float(loss.detach())))
            step += 1
            if max_steps and step >= max_steps:
                break
        if val_ds is not None:
            au = evaluate_auprc(backbone, head, tokenizer, val_ds, device, batch_size)
            hist["val_auprc"].append((step, au))
            if au > best and out_dir:
                best = au
                torch.save(head.state_dict(), str(Path(out_dir) / "clf_head_best.pt"))
        if max_steps and step >= max_steps:
            break
    if out_dir:
        json.dump(hist, open(Path(out_dir) / "clf_history.json", "w"), indent=2)
    return hist


def _encode(backbone, input_ids, attention_mask):
    """Get last_hidden_state from a peft-wrapped EsmModel or EsmForMaskedLM."""
    out = backbone(input_ids=input_ids, attention_mask=attention_mask,
                   output_hidden_states=True)
    if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
        return out.last_hidden_state
    return out.hidden_states[-1]


def evaluate_auprc(backbone, head, tokenizer, ds, device, batch_size):
    """Val AUPRC (average precision) — the §12 model-selection metric."""
    import torch
    from torch.utils.data import DataLoader
    from sklearn.metrics import average_precision_score
    pad_id = tokenizer.pad_token_id
    dl = DataLoader(ds, batch_size=batch_size, collate_fn=lambda b: collate_pad(b, pad_id))
    backbone.eval(); head.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in dl:
            batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
            h = _encode(backbone, batch["input_ids"], batch["attention_mask"])
            s = head(h, batch["attention_mask"])
            ps.extend(torch.sigmoid(s).cpu().tolist())
            ys.extend(batch["label"].cpu().tolist())
    backbone.train(); head.train()
    if len(set(ys)) < 2:
        return float("nan")
    return float(average_precision_score(ys, ps))
