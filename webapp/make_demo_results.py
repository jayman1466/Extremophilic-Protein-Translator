"""Write a demo results.json + placeholder structures for a job, so the results
UI can be developed/tested before the real generation pipeline exists.

This ALSO documents the results-bundle contract every backend must produce:
  job_dir/
    results.json           # schema below
    structures/wt.pdb
    structures/<design_id>.pdb

results.json schema:
{
  "wt_structure": "wt.pdb",
  "wt_sequence": "...",
  "by_phenotype": {
     "<phenotype>": [
        {"design_id","sequence","highlighted_seq","classifier_score",
         "active_site_rmsd","n_mutations","structure_file",
         "metrics": {label: value, ...}},
        ...
     ]
  }
}
"""
import sys, json, random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import store

AA = "ACDEFGHIKLMNPQRSTVWY"
WT = ("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKR"
      "QTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFG")


def highlight(wt, mut):
    out = []
    for a, b in zip(wt, mut):
        out.append(f'<span class="seq-mut">{b}</span>' if a != b else b)
    return "".join(out)


def mutate(seq, n):
    s = list(seq)
    pos = random.sample(range(len(s)), n)
    for p in pos:
        s[p] = random.choice(AA)
    return "".join(s)


def fake_pdb(seq, jitter=0.0):
    # minimal CA-only PDB along a line, jittered — enough for the viewer to render.
    lines = []
    for i, aa in enumerate(seq):
        x = i * 3.8 + random.uniform(-jitter, jitter)
        y = random.uniform(-jitter, jitter)
        z = random.uniform(-jitter, jitter)
        lines.append(f"ATOM  {i+1:>5}  CA  ALA A{i+1:>4}    "
                     f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C")
    lines.append("END")
    return "\n".join(lines)


def main(jid, phenotypes, n_designs):
    jd = store.job_dir(jid)
    (jd / "structures").mkdir(parents=True, exist_ok=True)
    (jd / "structures" / "wt.pdb").write_text(fake_pdb(WT))
    by = {}
    for ph in phenotypes:
        designs = []
        for k in range(n_designs):
            n_mut = random.randint(3, 12)
            mut = mutate(WT, n_mut)
            did = f"{ph[:4]}_{k+1}"
            (jd / "structures" / f"{did}.pdb").write_text(fake_pdb(mut, jitter=1.5))
            designs.append(dict(
                design_id=did, sequence=mut, highlighted_seq=highlight(WT, mut),
                classifier_score=round(random.uniform(0.55, 0.98), 3),
                active_site_rmsd=round(random.uniform(0.1, 1.4), 2),
                n_mutations=n_mut, structure_file=f"{did}.pdb",
                metrics={"pLDDT": round(random.uniform(70, 92), 1),
                         "TM-score": round(random.uniform(0.85, 0.99), 3),
                         "MPNN score": round(random.uniform(0.8, 1.5), 2),
                         "\u0394 charge": random.randint(-4, 4)}))
        designs.sort(key=lambda d: -d["classifier_score"])
        by[ph] = designs
    (jd / "results.json").write_text(json.dumps(
        dict(wt_structure="wt.pdb", wt_sequence=WT, by_phenotype=by), indent=2))
    store.set_status(jid, "done")
    print(f"wrote demo results for {jid}: {list(by)} x {n_designs}")


if __name__ == "__main__":
    store.init()
    jid = sys.argv[1]
    job = store.get_job(jid)
    main(jid, job["phenotypes"], job["n_designs"])
