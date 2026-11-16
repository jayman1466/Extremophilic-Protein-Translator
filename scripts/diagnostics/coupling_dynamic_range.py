"""Why does coupling_preservation return exactly 1.0 for every design?

The term counts WT top-128 contact pairs whose probability in the design's
map stays >= 0.5. If the strongest WT contacts sit near 1.0 and ESM-2's
contact head is dominated by a structural prior that substitution barely
moves, every pair clears 0.5 and the count saturates -- a constant 1.0 that
contributes nothing to ranking, the same defect as the exception path
reached a different way.

This measures the underlying probabilities rather than the thresholded
count, so a replacement statistic can be chosen on data.
"""
import importlib.util, json, os, random
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, EsmForMaskedLM

REPO = os.environ["REPO"]; PERSIST = os.environ["PERSIST"]
ADAPTER = f"{PERSIST}/runs/mhk32/mlm_adapt/mlm_adapter_best"
CKPT = "facebook/esm2_t36_3B_UR50D"
IS621 = ("ELHYIGIDTAKEKLDVDVLRPDGRHRTKKFANTTKGHDELVSWLKGHKIDHAHICIEATGTYMEPVAECLYDAGYIVSVI"
         "NPALGKAFAQSEGLRNKTDTVDARMLAEFCRQKRPAAWEAPHPLERALRALVVRHQALTDMHTQELNRTETAREVQRPSI"
         "DAHLLWLEAELKRLEKQIKDLTDDDPDMKHRRKLLESIPGIGEKTSAVLLAYIGLKDRFAHARQFAAFAGLTPRRMSKAG"
         "HVSLRRALYMPAMVATSKTEWGRAFRDRLAANGKKGKVILGAMMRKLAQVAYGVLKSGVPFDASRH")

spec = importlib.util.spec_from_file_location("genB", f"{REPO}/scripts/11_generate_pipelineB.py")
genB = importlib.util.module_from_spec(spec); spec.loader.exec_module(genB)

tok = AutoTokenizer.from_pretrained(CKPT)
base = EsmForMaskedLM.from_pretrained(CKPT, attn_implementation="eager")
from peft import PeftModel
model = PeftModel.from_pretrained(base, ADAPTER, adapter_name="mlm")
model.set_adapter("mlm"); model.eval().to("cpu")
print("[dyn] model ready", flush=True)

from eptrans.modeling.data import _predict_contacts
from eptrans.modeling import masking as MK

wt_cm = _predict_contacts(model, tok, IS621)
pairs = MK.contact_pairs_from_map(wt_cm, threshold=0.5, min_sep=6, top_k=128)
wt_vals = np.array([float(wt_cm[i, j]) for i, j in pairs])
print(f"[dyn] pairs={len(pairs)} wt prob min={wt_vals.min():.3f} "
      f"med={np.median(wt_vals):.3f} max={wt_vals.max():.3f}", flush=True)

AA = "ACDEFGHIKLMNPQRSTVWY"; rng = random.Random(1466)
def mutate(s, k):
    s = list(s)
    for p in rng.sample(range(len(s)), k):
        s[p] = rng.choice([a for a in AA if a != s[p]])
    return "".join(s)

rows = []
for k in (0, 5, 20, 60, 100, 150):
    s = IS621 if k == 0 else mutate(IS621, k)
    cm = _predict_contacts(model, tok, s)
    v = np.array([float(cm[i, j]) for i, j in pairs])
    rows.append(dict(
        n_mut=k,
        frac_ge_0_5=float((v >= 0.5).mean()),
        frac_ge_0_8=float((v >= 0.8).mean()),
        frac_ge_0_9=float((v >= 0.9).mean()),
        frac_ge_0_95=float((v >= 0.95).mean()),
        mean_prob=float(v.mean()),
        min_prob=float(v.min()),
        p10_prob=float(np.percentile(v, 10)),
        mean_ratio_to_wt=float((v / wt_vals).mean()),
        pearson_r_full=float(np.corrcoef(np.asarray(wt_cm).flatten(),
                                         np.asarray(cm).flatten())[0, 1]),
    ))
    r = rows[-1]
    print(f"[dyn] nmut={k:3d} ge.5={r['frac_ge_0_5']:.3f} ge.8={r['frac_ge_0_8']:.3f} "
          f"ge.9={r['frac_ge_0_9']:.3f} mean={r['mean_prob']:.4f} "
          f"min={r['min_prob']:.3f} r={r['pearson_r_full']:.4f}", flush=True)

out = dict(n_pairs=len(pairs),
           wt_pair_prob=dict(min=float(wt_vals.min()),
                             median=float(np.median(wt_vals)),
                             max=float(wt_vals.max())),
           rows=rows)
def rng_of(key):
    vals = [r[key] for r in rows]
    return float(max(vals) - min(vals))
out["dynamic_range"] = {k: rng_of(k) for k in
                        ("frac_ge_0_5", "frac_ge_0_8", "frac_ge_0_9", "frac_ge_0_95",
                         "mean_prob", "mean_ratio_to_wt", "pearson_r_full")}
Path("coupling_dynamic_range.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out["dynamic_range"], indent=2), flush=True)
