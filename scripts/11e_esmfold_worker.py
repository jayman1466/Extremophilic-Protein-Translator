#!/usr/bin/env python
"""Persistent ESMFold worker for in-loop refolding during generation.

Generation (11_generate.py) runs in the `eptrans_ml` env (transformers 5.x); ESMFold
needs the `esmfold` env (transformers 4.46.3 EsmForProteinFolding). They can't share a
process, so this worker loads ESMFold ONCE in its own env and folds sequences the
generation loop hands it through a shared queue directory. Both run concurrently in the
same sbatch (worker backgrounded via `conda run -n esmfold`, generate in the foreground).

Protocol (all files under --workdir):
  READY                       written after the model loads (generator waits for it)
  requests/<id>.fasta         one sequence (bare, no header); generator writes atomically
  responses/<id>.pdb          folded structure (atomic rename); generator reads then deletes
  STOP                        sentinel: worker folds any pending requests then exits

The generator computes active-site RMSD itself (it already has the WT CA coords), so the
worker only returns the PDB — keeping the contract minimal and the worker stateless.
"""
import sys, os, time, argparse
from pathlib import Path
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--max-len", type=int, default=1000)
    ap.add_argument("--poll", type=float, default=0.5)
    ap.add_argument("--idle-timeout", type=float, default=3600.0,
                    help="exit if no request and no STOP for this long (safety net)")
    args = ap.parse_args()

    wd = Path(args.workdir)
    req = wd / "requests"; resp = wd / "responses"
    req.mkdir(parents=True, exist_ok=True); resp.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import AutoTokenizer, EsmForProteinFolding
    tok = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1",
                                                 low_cpu_mem_usage=True).cuda().eval()
    model.esm = model.esm.half()

    @torch.no_grad()
    def fold(seq):
        ids = tok([seq[:args.max_len]], return_tensors="pt",
                  add_special_tokens=False)["input_ids"].cuda()
        return model.output_to_pdb(model(ids))[0]

    (wd / "READY").write_text("1")
    print(f"[worker] ESMFold ready, watching {req}", flush=True)

    last_activity = time.time()
    while True:
        pending = sorted(req.glob("*.fasta"))
        if not pending:
            if (wd / "STOP").exists():
                print("[worker] STOP seen, exiting", flush=True)
                break
            if time.time() - last_activity > args.idle_timeout:
                print("[worker] idle timeout, exiting", flush=True)
                break
            time.sleep(args.poll)
            continue
        for f in pending:
            rid = f.stem
            try:
                seq = f.read_text().strip()
                pdb = fold(seq)
                tmp = resp / f"{rid}.pdb.tmp"
                tmp.write_text(pdb)
                tmp.rename(resp / f"{rid}.pdb")   # atomic publish
            except Exception as e:  # noqa: BLE001 — surface fold failure to generator
                (resp / f"{rid}.err").write_text(str(e))
                print(f"[worker] fold {rid} failed: {e}", flush=True)
            finally:
                f.unlink(missing_ok=True)
            last_activity = time.time()


if __name__ == "__main__":
    main()
