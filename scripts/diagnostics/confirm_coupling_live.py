"""Confirm the coupling term is live end-to-end in the production code path.

Loads the real mhk32 adapter + alkaliphile head exactly as
scripts/11_generate_pipelineB.py does (eager attention, PEFT wrapper),
builds the oracle via build_oracle(), and scores WT plus mutants of
increasing edit distance.

Pass criteria:
  wt_contact_pairs > 0        WT contact map was produced at all
  n_failures == 0             no design-side exception fell back to 1.0
  coupling(WT) == 1.0         self-comparison is exact
  coupling(mutant) varies     the term discriminates rather than pinning

CPU-only: the h200 node is drained and the remaining gpu partition's QOS
caps a job at 12 G, which cannot hold a 3B model.
"""
import importlib.util, json, os, random, sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, EsmForMaskedLM

REPO = os.environ["REPO"]
PERSIST = os.environ["PERSIST"]
ADAPTER = f"{PERSIST}/runs/mhk32/mlm_adapt/mlm_adapter_best"
HEAD = f"{PERSIST}/runs/mhk32/heads_meanpool/clf_alkaliphile/head_best.pt"
CKPT = "facebook/esm2_t36_3B_UR50D"

IS621 = ("ELHYIGIDTAKEKLDVDVLRPDGRHRTKKFANTTKGHDELVSWLKGHKIDHAHICIEATGTYMEPVAECLYDAGYIVSVI"
         "NPALGKAFAQSEGLRNKTDTVDARMLAEFCRQKRPAAWEAPHPLERALRALVVRHQALTDMHTQELNRTETAREVQRPSI"
         "DAHLLWLEAELKRLEKQIKDLTDDDPDMKHRRKLLESIPGIGEKTSAVLLAYIGLKDRFAHARQFAAFAGLTPRRMSKAG"
         "HVSLRRALYMPAMVATSKTEWGRAFRDRLAANGKKGKVILGAMMRKLAQVAYGVLKSGVPFDASRH")

spec = importlib.util.spec_from_file_location("genB", f"{REPO}/scripts/11_generate_pipelineB.py")
genB = importlib.util.module_from_spec(spec)
spec.loader.exec_module(genB)
print("[confirm] module loaded", flush=True)

tok = AutoTokenizer.from_pretrained(CKPT)
base = EsmForMaskedLM.from_pretrained(CKPT, attn_implementation="eager")
from peft import PeftModel
model = PeftModel.from_pretrained(base, ADAPTER, adapter_name="mlm")
model.set_adapter("mlm")
model.eval().to("cpu")
print("[confirm] model + adapter loaded, attn =",
      base.config._attn_implementation, flush=True)

hidden = base.config.hidden_size
head = torch.nn.Sequential(torch.nn.Linear(hidden, 512), torch.nn.GELU(),
                           torch.nn.Dropout(0.1), torch.nn.Linear(512, 1))
_ck = torch.load(HEAD, map_location="cpu")
head.load_state_dict(_ck["state_dict"] if isinstance(_ck, dict) and "state_dict" in _ck else _ck)
head.eval().to("cpu")

sc, ll, coupling, wt_pairs = genB.build_oracle(model, tok, head, "cpu", IS621)
print(f"[confirm] wt_contact_pairs={len(wt_pairs)}", flush=True)

AA = "ACDEFGHIKLMNPQRSTVWY"
rng = random.Random(1466)
def mutate(s, k):
    s = list(s)
    for p in rng.sample(range(len(s)), k):
        s[p] = rng.choice([a for a in AA if a != s[p]])
    return "".join(s)

rows = [("WT", 0, IS621)]
for k in (1, 5, 20, 60):
    rows.append((f"mut{k}", k, mutate(IS621, k)))

out = []
for name, k, s in rows:
    cp = float(coupling(s))
    out.append(dict(name=name, n_mut=k, coupling=round(cp, 6)))
    print(f"[confirm] {name:6s} nmut={k:3d} coupling={cp:.6f}", flush=True)

health = dict(
    wt_contact_pairs=len(wt_pairs),
    n_calls=int(getattr(coupling, "n_calls", -1)),
    n_failures=int(getattr(coupling, "n_failures", -1)),
)
vals = [r["coupling"] for r in out]
result = dict(
    coupling_health=health,
    scores=out,
    checks=dict(
        wt_pairs_nonzero=len(wt_pairs) > 0,
        no_failures=health["n_failures"] == 0,
        wt_self_identity=abs(out[0]["coupling"] - 1.0) < 1e-9,
        discriminates=len(set(vals)) > 1,
        monotone_decreasing=all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1)),
    ),
)
result["pass"] = all(v for kk, v in result["checks"].items() if kk != "monotone_decreasing")
Path("coupling_live_confirmation.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2), flush=True)
