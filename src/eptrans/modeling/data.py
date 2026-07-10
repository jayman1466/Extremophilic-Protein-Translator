"""Data plumbing for the modeling scaffold.

The labeled table (Stage 06) carries labels/splits/groups keyed by ``tagged_id``
(``GENOME~PROTID``) but NOT the sequences — those live in the mature-chain FASTA
(``secreted_proteins_r232.faa``, headers ``GENOME~PROTID class=... anchoring=...``).
This module joins them and produces torch datasets:

  - ``MlmSequenceDataset``: sequences from TRAIN-only clusters for domain-adaptive
    MLM (Section 11 leakage rule 2 — the adaptation corpus is train-only).
  - ``ClassifierDataset``: (sequence, binary label for one phenotype, sample
    weight) for a per-phenotype head, any split.
  - ``load_pairs`` + ``PairBatchSampler`` helper: aligned matched-pair logits for
    the margin term (Section 12 L_pair).

FASTA parsing is stdlib; torch is imported lazily inside the Dataset classes.
"""
from __future__ import annotations

import os
from typing import Iterator

import pandas as pd

__all__ = [
    "iter_fasta",
    "load_sequences",
    "attach_sequences",
    "phenotype_binary_labels",
    "build_mlm_dataset",
    "build_classifier_dataset",
    "build_pair_dataset",
    "collate_pairs",
]

# 20 standard aa tokens used for BERT random-replacement (ids resolved via the
# tokenizer at train time; kept here as the alphabet reference).
STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"


def iter_fasta(path: str | os.PathLike) -> Iterator[tuple[str, str]]:
    """Yield ``(header_id, sequence)`` where header_id is the first token."""
    hid, seq = None, []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if hid is not None:
                    yield hid, "".join(seq)
                hid = line[1:].split()[0]
                seq = []
            else:
                seq.append(line.strip())
    if hid is not None:
        yield hid, "".join(seq)


def load_sequences(fasta: str | os.PathLike, keep_ids: set | None = None) -> dict[str, str]:
    """Load ``tagged_id -> sequence`` from the mature-chain FASTA.

    ``keep_ids`` (optional) restricts to the ids actually needed (memory: the
    full secretome FASTA is ~865 MB / ~2M records).
    """
    out = {}
    for hid, seq in iter_fasta(fasta):
        if keep_ids is None or hid in keep_ids:
            out[hid] = seq
    return out


def attach_sequences(labeled: pd.DataFrame, fasta: str | os.PathLike,
                     id_col: str = "tagged_id", seq_col: str = "sequence") -> pd.DataFrame:
    """Left-join sequences onto the labeled table by ``tagged_id``.

    Rows whose id is absent from the FASTA (should be none for the mature
    secretome) get NaN and are dropped with a count reported via the returned
    frame's attrs.
    """
    ids = set(labeled[id_col].astype(str))
    seqs = load_sequences(fasta, keep_ids=ids)
    df = labeled.copy()
    df[seq_col] = df[id_col].astype(str).map(seqs)
    n_missing = int(df[seq_col].isna().sum())
    df = df[df[seq_col].notna()].reset_index(drop=True)
    df.attrs["n_missing_sequences"] = n_missing
    return df


def phenotype_binary_labels(labeled: pd.DataFrame, phenotype: str,
                            label_col: str = "label",
                            mesophile_token: str = "mesophile") -> pd.Series:
    """Binary label for one phenotype from the multi-label ``;``-joined column.

    y=1 if ``phenotype`` is among the row's labels; y=0 for mesophile rows.
    Rows that are neither (a *different* extremophile, no mesophile token) are
    returned as NaN so the caller can drop them — a per-phenotype classifier
    contrasts THIS phenotype against matched mesophiles, not against other
    extremophiles (Section 11 per-phenotype design).
    """
    lab = labeled[label_col].astype(str).str.split(";")
    is_pos = lab.apply(lambda L: phenotype in L)
    is_meso = lab.apply(lambda L: mesophile_token in L)
    import numpy as np
    y = pd.Series(np.nan, index=labeled.index, dtype="float")
    y[is_pos] = 1.0
    y[(~is_pos) & is_meso] = 0.0
    return y


def build_mlm_dataset(labeled_with_seq: pd.DataFrame, tokenizer, split: str = "train",
                      max_len: int = 1022, gamma: float = 1.0, mask_rate: float = 0.15,
                      conservation_col: str | None = None, seed: int = 1466):
    """Torch Dataset for domain-adaptive MLM over one split (default train-only).

    Each item tokenizes a sequence, samples mask positions with the
    conservation-weighted scheme (Section 13), applies BERT 80/10/10, and
    returns input_ids/labels/seq_weight. When ``conservation_col`` is absent
    (no MSA yet) conservation defaults to zeros -> uniform masking at the given
    ``mask_rate`` (gamma has no effect), which is the correct fallback.
    """
    import numpy as np
    import torch
    from torch.utils.data import Dataset
    from .masking import sample_mask_positions, bert_mask_assignment
    from .losses import confidence_to_weight

    sub = labeled_with_seq[labeled_with_seq["split"] == split].reset_index(drop=True)
    aa_ids = tokenizer.convert_tokens_to_ids(list(STANDARD_AA))
    mask_id = tokenizer.mask_token_id

    class MlmSequenceDataset(Dataset):
        def __init__(self):
            self.df = sub
            self.rng = np.random.default_rng(seed)

        def __len__(self):
            return len(self.df)

        def __getitem__(self, i):
            row = self.df.iloc[i]
            seq = row["sequence"][:max_len]
            enc = tokenizer(seq, return_tensors="pt", truncation=True, max_length=max_len + 2)
            input_ids = enc["input_ids"][0]
            attn = enc["attention_mask"][0]
            L = input_ids.shape[0]
            special = np.array(tokenizer.get_special_tokens_mask(
                input_ids.tolist(), already_has_special_tokens=True), dtype=bool)
            if conservation_col and conservation_col in row and row[conservation_col] is not None:
                cons = np.asarray(row[conservation_col], dtype=float)
                cons = np.pad(cons, (1, L - 1 - len(cons)))[:L] if len(cons) < L else cons[:L]
            else:
                cons = np.zeros(L)
            masked = sample_mask_positions(cons, mask_rate=mask_rate, gamma=gamma,
                                           special=special, rng=self.rng)
            assign = bert_mask_assignment(masked, rng=self.rng)
            labels = torch.full_like(input_ids, -100)
            labels[torch.from_numpy(assign["loss"])] = input_ids[torch.from_numpy(assign["loss"])]
            ii = input_ids.clone()
            ii[torch.from_numpy(assign["replace_mask"])] = mask_id
            rnd_pos = np.where(assign["replace_random"])[0]
            for p in rnd_pos:
                ii[p] = int(self.rng.choice(aa_ids))
            return {"input_ids": ii, "attention_mask": attn, "labels": labels,
                    "seq_weight": torch.tensor(confidence_to_weight(row.get("label_confidence")),
                                               dtype=torch.float)}

    return MlmSequenceDataset()


def build_classifier_dataset(labeled_with_seq: pd.DataFrame, tokenizer, phenotype: str,
                             split: str, max_len: int = 1022):
    """Torch Dataset for the per-phenotype classifier head (one phenotype)."""
    import torch
    from torch.utils.data import Dataset
    from .losses import confidence_to_weight

    df = labeled_with_seq[labeled_with_seq["split"] == split].copy()
    y = phenotype_binary_labels(df, phenotype)
    df = df.assign(_y=y)
    df = df[df["_y"].notna()].reset_index(drop=True)

    class ClassifierDataset(Dataset):
        def __init__(self):
            self.df = df

        def __len__(self):
            return len(self.df)

        def __getitem__(self, i):
            row = self.df.iloc[i]
            enc = tokenizer(row["sequence"][:max_len], return_tensors="pt",
                            truncation=True, max_length=max_len + 2)
            return {"input_ids": enc["input_ids"][0],
                    "attention_mask": enc["attention_mask"][0],
                    "label": torch.tensor(row["_y"], dtype=torch.float),
                    "weight": torch.tensor(confidence_to_weight(row.get("label_confidence")),
                                           dtype=torch.float),
                    "tagged_id": row["tagged_id"]}

    return ClassifierDataset()


def build_pair_dataset(labeled_with_seq: pd.DataFrame, pairs: pd.DataFrame, tokenizer,
                       phenotype: str, split: str, max_len: int = 1022):
    """Torch Dataset of matched (extremophile, outgroup) protein pairs for L_pair.

    Restricts the stage-06 protein-pairs table to (a) this phenotype's `class`
    and (b) pairs whose BOTH members fall in ``split`` (they co-cluster, so this
    is essentially all of them — the ``ext_split``/``out_split`` columns make it
    explicit). Each item returns the two tokenized proteins so the training loop
    scores them and applies the margin term ``max(0, δ - (s_ext - s_out))``.

    The split is unchanged by this — pairs are a side-car index used only to
    co-load matched proteins into a batch (answering: pairing need not be
    "retained" in the split itself; the pooled split + this index suffice).
    """
    import torch
    from torch.utils.data import Dataset

    seqmap = dict(zip(labeled_with_seq["tagged_id"].astype(str),
                      labeled_with_seq["sequence"]))
    p = pairs.copy()
    if "class" in p.columns:
        p = p[p["class"] == phenotype]
    if "ext_split" in p.columns and "out_split" in p.columns:
        p = p[(p["ext_split"] == split) & (p["out_split"] == split)]
    # both members must have a sequence
    p = p[p["ext_id"].astype(str).isin(seqmap) & p["outgroup_id"].astype(str).isin(seqmap)]
    p = p.reset_index(drop=True)

    def _tok(seq):
        enc = tokenizer(seq[:max_len], return_tensors="pt", truncation=True,
                        max_length=max_len + 2)
        return enc["input_ids"][0], enc["attention_mask"][0]

    class PairDataset(Dataset):
        def __init__(self):
            self.p = p

        def __len__(self):
            return len(self.p)

        def __getitem__(self, i):
            row = self.p.iloc[i]
            ei, ea = _tok(seqmap[str(row["ext_id"])])
            oi, oa = _tok(seqmap[str(row["outgroup_id"])])
            return {"ext_input_ids": ei, "ext_attention_mask": ea,
                    "out_input_ids": oi, "out_attention_mask": oa}

    return PairDataset()


def collate_pairs(batch, pad_id: int):
    """Pad a batch of pair items on both the ext and out sides independently."""
    import torch

    def _pad(key_ids, key_attn):
        maxlen = max(x[key_ids].shape[0] for x in batch)
        ids = torch.stack([
            torch.cat([x[key_ids], torch.full((maxlen - x[key_ids].shape[0],), pad_id,
                                              dtype=x[key_ids].dtype)]) for x in batch])
        attn = torch.stack([
            torch.cat([x[key_attn], torch.zeros(maxlen - x[key_attn].shape[0],
                                               dtype=x[key_attn].dtype)]) for x in batch])
        return ids, attn

    ei, ea = _pad("ext_input_ids", "ext_attention_mask")
    oi, oa = _pad("out_input_ids", "out_attention_mask")
    return {"ext_input_ids": ei, "ext_attention_mask": ea,
            "out_input_ids": oi, "out_attention_mask": oa}
