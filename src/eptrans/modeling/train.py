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
import os
from pathlib import Path

__all__ = ["collate_pad", "train_mlm", "train_classifier", "evaluate_auprc",
           "LengthBucketSampler"]


class LengthBucketSampler:
    """Batch sampler that groups similar-length items to cut padding waste.

    Random batching pads every item to the batch max; with a median-285/p95-900
    length distribution that wastes a large fraction of tokens. This sorts a
    shuffled pool into length order within chunks (``pool = mult * batch_size``),
    yields contiguous same-length batches, and shuffles batch ORDER each epoch —
    so there's still stochasticity, but each batch is length-homogeneous.
    Reduces effective tokens ~2-3x on this corpus.
    """

    def __init__(self, lengths, batch_size, shuffle=True, pool_mult=50, seed=1466):
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pool = batch_size * pool_mult
        self.rng = __import__("numpy").random.default_rng(seed)

    def __iter__(self):
        import numpy as np
        idx = np.arange(len(self.lengths))
        if self.shuffle:
            self.rng.shuffle(idx)
        batches = []
        for start in range(0, len(idx), self.pool):
            chunk = idx[start:start + self.pool]
            chunk = chunk[np.argsort([self.lengths[i] for i in chunk])]
            for b in range(0, len(chunk), self.batch_size):
                batches.append(chunk[b:b + self.batch_size].tolist())
        if self.shuffle:
            self.rng.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size


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
              log_every: int = 50, bucket_by_length: bool = True,
              amp_dtype: str = "bf16", ckpt_every: int = 0, resume: bool = True):
    """Domain-adaptive continued MLM (LoRA). Returns training history dict.

    ``amp_dtype``: "bf16" (default) wraps forward passes in
    ``torch.autocast(dtype=bfloat16)`` — on the H200 this roughly halves memory
    and doubles throughput, and needs no GradScaler (bf16 has fp32's dynamic
    range). "fp32" disables autocast (CPU smoke-tests). LoRA master weights stay
    fp32; only the compute is bf16.

    ``ckpt_every``: if >0, write a resumable step-checkpoint (trainable LoRA
    weights + optimizer/scheduler state + epoch + batch-in-epoch + best_ppl +
    history) to ``<out_dir>/mlm_ckpt.pt`` every N optimizer steps. On a spot
    preemption the replacement instance resumes mid-epoch: it restores the
    interrupted epoch and skips the batches already done, so lost work is bounded
    by the ckpt interval — at most N steps back to the last checkpoint, not a
    whole epoch. ``step`` is derived from ``(epoch, batch_index)`` rather than a
    free-running counter, so a resume can never overshoot the LR schedule's
    ``total`` (which would zero the learning-rate tail and trip the max_steps
    break early). ``resume``: if True and the checkpoint exists, load and continue.
    Only trainable params are saved, so the file is small (LoRA-sized).
    """
    import torch
    from torch.utils.data import DataLoader
    from .losses import masked_mlm_loss, kl_forgetting_guard

    def _trainable_sd():
        return {n: p.detach().cpu() for n, p in peft_model.named_parameters() if p.requires_grad}

    def _save_ckpt(path, step, ep, batch_in_epoch, best_ppl, hist):
        tmp = str(path) + ".tmp"
        torch.save({"trainable": _trainable_sd(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "step": step, "epoch": ep,
                    "batch_in_epoch": batch_in_epoch, "best_ppl": best_ppl,
                    "hist": hist}, tmp)
        os.replace(tmp, path)  # atomic: a preemption mid-write can't corrupt the ckpt

    use_amp = amp_dtype == "bf16" and str(device).startswith("cuda")
    amp_ctx = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if use_amp \
        else __import__("contextlib").nullcontext

    pad_id = tokenizer.pad_token_id
    if bucket_by_length and hasattr(train_ds, "items"):
        lengths = [len(seq) for _, _, seq in train_ds.items]
        dl = DataLoader(train_ds, batch_sampler=LengthBucketSampler(lengths, batch_size),
                        collate_fn=lambda b: collate_pad(b, pad_id))
    else:
        dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                        collate_fn=lambda b: collate_pad(b, pad_id))
    peft_model.to(device).train()
    opt = torch.optim.AdamW([p for p in peft_model.parameters() if p.requires_grad], lr=lr)
    steps_per_epoch = len(dl)
    total = (max_steps or steps_per_epoch * epochs)
    warmup = max(1, int(warmup_frac * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / warmup) * max(0.0, (total - s) / max(1, total - warmup)))

    hist = {"train_loss": [], "val_ppl": []}
    best_ppl = math.inf
    start_ep = 0
    resume_skip = 0          # batches already completed in the interrupted epoch
    ckpt_path = Path(out_dir) / "mlm_ckpt.pt" if out_dir else None
    if resume and ckpt_path and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device)
        missing = peft_model.load_state_dict(ck["trainable"], strict=False)
        opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start_ep = ck["epoch"]; resume_skip = ck.get("batch_in_epoch", 0)
        best_ppl = ck["best_ppl"]; hist = ck.get("hist", hist)
        print(f"[train_mlm] resumed at epoch {start_ep}, batch {resume_skip} "
              f"(step {start_ep * steps_per_epoch + resume_skip}); "
              f"{len(missing.unexpected_keys)} unexpected keys")

    # ``step`` is DERIVED from (epoch, batch index), never incremented on top of a
    # restored value — so a resume cannot overshoot ``total`` (which would zero the
    # LR tail and fire the max_steps break early). Mid-epoch resume skips only the
    # batches already done in the interrupted epoch; later epochs run fresh from 0.
    stop = False
    for ep in range(start_ep, epochs):
        skip = resume_skip if ep == start_ep else 0
        for bi, batch in enumerate(dl):
            if bi < skip:
                continue
            step = ep * steps_per_epoch + bi
            batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
            with amp_ctx():
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
            done = step + 1                       # optimizer steps completed
            if ckpt_every and ckpt_path and done % ckpt_every == 0:
                _save_ckpt(ckpt_path, done, ep, bi + 1, best_ppl, hist)
            if max_steps and done >= max_steps:
                stop = True
                break
        if val_ds is not None:
            ppl = _pseudo_perplexity(peft_model, tokenizer, val_ds, device, batch_size, amp_ctx)
            hist["val_ppl"].append((step, ppl))
            if ppl < best_ppl and out_dir:
                best_ppl = ppl
                peft_model.save_pretrained(str(Path(out_dir) / "mlm_adapter_best"))
        if stop:
            break
    if out_dir:
        peft_model.save_pretrained(str(Path(out_dir) / "mlm_adapter_last"))
        json.dump(hist, open(Path(out_dir) / "mlm_history.json", "w"), indent=2)
    return hist


def _pseudo_perplexity(model, tokenizer, ds, device, batch_size, amp_ctx=None):
    """Cheap MLM val metric: exp(mean masked CE) over the val set as given
    (its items already carry masked labels)."""
    import contextlib
    import torch
    from torch.utils.data import DataLoader
    from .losses import masked_mlm_loss
    if amp_ctx is None:
        amp_ctx = contextlib.nullcontext
    pad_id = tokenizer.pad_token_id
    dl = DataLoader(ds, batch_size=batch_size, collate_fn=lambda b: collate_pad(b, pad_id))
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for batch in dl:
            batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
            with amp_ctx():
                out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                l = masked_mlm_loss(out.logits, batch["labels"].clamp_min(0),
                                    batch["labels"] != -100)
            tot += float(l); n += 1
    model.train()
    return math.exp(tot / max(1, n))


def train_classifier(backbone, head, tokenizer, train_ds, val_ds=None, *,
                     pair_ds=None, epochs: int = 5, lr_head: float = 1e-3,
                     lr_adapter: float = 1e-5, batch_size: int = 16,
                     pair_batch_size: int | None = None, lam: float = 1.0,
                     margin: float = 1.0, pos_weight: float | None = None,
                     device: str = "cpu", out_dir: str | None = None,
                     max_steps: int | None = None,
                     log_every: int = 20, ckpt_every: int = 0, resume: bool = True):
    """Per-phenotype classifier head on the adapted backbone (§12 Loss 2).

    ``pair_ds``: optional matched-pair Dataset (build_pair_dataset). When given,
    a pair loader runs in lockstep with the main loader (cycling if shorter);
    each step scores both members through the same backbone+head and adds the
    margin term ``max(0, margin - (s_ext - s_out))``. When None, only the
    pointwise BCE is used. Returns history with val AUPRC.

    ``log_every``: print a flushed ``[train_clf] epoch E step S/T loss L`` line
    every N optimizer steps so a block-buffered SLURM log shows live progress
    (proof-of-life) long before the first epoch-end eval. ``ckpt_every``: if >0,
    every N steps atomically write a resumable checkpoint (trainable backbone
    LoRA weights + head + optimizer + epoch + batch-in-epoch + best + history) to
    ``<out_dir>/clf_ckpt.pt`` AND flush the running ``clf_history.json`` — so
    both a growing metrics file and an mtime-updating checkpoint prove the job is
    advancing between epochs. ``step`` is DERIVED from ``(epoch, batch_index)``;
    on ``resume`` an existing checkpoint is loaded and the interrupted epoch
    resumes mid-stream, skipping only the batches already done (there is no LR
    scheduler here, so a resume cannot overshoot). Only trainable params are
    saved, so the checkpoint is LoRA-sized.
    """
    import itertools
    import torch
    from torch.utils.data import DataLoader
    from .losses import classifier_loss
    from .data import collate_pairs

    def _trainable_backbone_sd():
        return {n: p.detach().cpu() for n, p in backbone.named_parameters() if p.requires_grad}

    def _save_ckpt(path, step, ep, batch_in_epoch, best, hist):
        tmp = str(path) + ".tmp"
        torch.save({"backbone_trainable": _trainable_backbone_sd(),
                    "head": head.state_dict(), "opt": opt.state_dict(),
                    "step": step, "epoch": ep, "batch_in_epoch": batch_in_epoch,
                    "best": best, "hist": hist}, tmp)
        os.replace(tmp, path)  # atomic: a kill mid-write can't corrupt the ckpt

    pad_id = tokenizer.pad_token_id
    dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                    collate_fn=lambda b: collate_pad(b, pad_id))
    pair_iter = None
    if pair_ds is not None and len(pair_ds) > 0:
        pdl = DataLoader(pair_ds, batch_size=(pair_batch_size or batch_size), shuffle=True,
                         collate_fn=lambda b: collate_pairs(b, pad_id))
        pair_iter = itertools.cycle(pdl)  # cycle: pairs usually fewer than singles
    backbone.to(device); head.to(device)
    params = [{"params": [p for p in backbone.parameters() if p.requires_grad], "lr": lr_adapter},
              {"params": head.parameters(), "lr": lr_head}]
    opt = torch.optim.AdamW(params)
    pw = torch.tensor(pos_weight, device=device) if pos_weight else None

    hist = {"train_loss": [], "val_auprc": []}
    best = -1.0
    steps_per_epoch = len(dl)
    total = max_steps or steps_per_epoch * epochs
    start_ep = 0
    resume_skip = 0          # batches already completed in the interrupted epoch
    ckpt_path = Path(out_dir) / "clf_ckpt.pt" if out_dir else None
    if resume and ckpt_path and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device)
        backbone.load_state_dict(ck["backbone_trainable"], strict=False)
        head.load_state_dict(ck["head"]); opt.load_state_dict(ck["opt"])
        start_ep = ck["epoch"]; resume_skip = ck.get("batch_in_epoch", 0)
        best = ck.get("best", -1.0); hist = ck.get("hist", hist)
        print(f"[train_clf] resumed at epoch {start_ep}, batch {resume_skip} "
              f"(step {start_ep * steps_per_epoch + resume_skip})", flush=True)

    step = start_ep * steps_per_epoch + resume_skip
    stop = False
    for ep in range(start_ep, epochs):
        backbone.train(); head.train()
        skip = resume_skip if ep == start_ep else 0
        for bi, batch in enumerate(dl):
            if bi < skip:
                continue
            step = ep * steps_per_epoch + bi
            batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
            h = _encode(backbone, batch["input_ids"], batch["attention_mask"])
            s = head(h, batch["attention_mask"])
            pe = po = None
            if pair_iter is not None:
                pb = next(pair_iter)
                pb = {k: v.to(device) for k, v in pb.items()}
                pe = head(_encode(backbone, pb["ext_input_ids"], pb["ext_attention_mask"]),
                          pb["ext_attention_mask"])
                po = head(_encode(backbone, pb["out_input_ids"], pb["out_attention_mask"]),
                          pb["out_attention_mask"])
            loss, parts = classifier_loss(s, batch["label"], sample_weight=batch.get("weight"),
                                          pos_weight=pw, pair_ext=pe, pair_out=po,
                                          lam=lam, margin=margin)
            opt.zero_grad(); loss.backward(); opt.step()
            hist["train_loss"].append((step, float(loss.detach())))
            if step % log_every == 0:
                print(f"[train_clf] epoch {ep} step {step}/{total} "
                      f"loss {float(loss.detach()):.4f}", flush=True)
            done = step + 1
            if ckpt_every and ckpt_path and done % ckpt_every == 0:
                _save_ckpt(ckpt_path, done, ep, bi + 1, best, hist)
                json.dump(hist, open(Path(out_dir) / "clf_history.json", "w"), indent=2)
            if max_steps and done >= max_steps:
                stop = True
                break
        if val_ds is not None:
            au = evaluate_auprc(backbone, head, tokenizer, val_ds, device, batch_size)
            hist["val_auprc"].append((step, au))
            print(f"[train_clf] epoch {ep} END val_auprc {au:.4f}", flush=True)
            if au > best and out_dir:
                best = au
                torch.save(head.state_dict(), str(Path(out_dir) / "clf_head_best.pt"))
        if out_dir:
            json.dump(hist, open(Path(out_dir) / "clf_history.json", "w"), indent=2)
        if stop:
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
