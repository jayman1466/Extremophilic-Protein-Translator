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
        {"design_id","sequence","classifier_score",
         "active_site_rmsd","n_mutations","structure_file",
         "metrics": {label: value, ...},
         # sequence-track payload (3 orthogonal display channels):
         "track": {
            "seq": "<design seq>", "wt": "<wt seq>",
            "conservation": [float in [0,1] | null, ...],   # per-residue MSA conservation
            "active_site": [1-based int, ...],               # putative catalytic residues
            "active_site_assigned": bool,                    # false -> UI shows warning
            "mutations": [{"pos":1-based,"wt":a,"mut":b}, ...]
         }},
        ...
     ]
  }
}
"""
import sys, json, random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import store
import aggressiveness as agg

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
        s[p] = random.choice([a for a in AA if a != s[p]])  # guarantee a real change
    return "".join(s)


def fake_conservation(L, active_site, seed):
    # smooth-ish per-position conservation in [0,1]; active-site positions forced high.
    rng = random.Random(seed)
    base = [rng.random() for _ in range(L)]
    # smooth with neighbors so it reads like a real profile, not white noise
    cons = []
    for i in range(L):
        w = base[max(0, i-1):min(L, i+2)]
        cons.append(round(sum(w) / len(w), 3))
    for p in active_site:  # 1-based
        if 1 <= p <= L:
            cons[p-1] = round(min(1.0, 0.85 + rng.random() * 0.15), 3)
    return cons


def build_track(wt, mut, conservation, active_site, active_site_assigned):
    muts = [dict(pos=i+1, wt=a, mut=b) for i, (a, b) in enumerate(zip(wt, mut)) if a != b]
    return dict(seq=mut, wt=wt, conservation=conservation,
                active_site=active_site, active_site_assigned=active_site_assigned,
                mutations=muts)


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


def main(jid, phenotypes, n_designs, override=None):
    jd = store.job_dir(jid)
    (jd / "structures").mkdir(parents=True, exist_ok=True)
    (jd / "structures" / "wt.pdb").write_text(fake_pdb(WT))
    L = len(WT)
    # per-design aggressiveness schedule — the N designs span conservative->aggressive
    scheds = agg.resolve(n_designs, override=override)
    # a fixed putative active-site set for the demo enzyme (1-based, catalytic triad-like)
    DEMO_ACTIVE_SITE = [77, 105, 133, 40]
    by = {}
    for ph in phenotypes:
        designs = []
        for k, s in enumerate(scheds):
            # realized mutation count tracks the design's target_mut_frac (of the
            # mutable surface, ~70% of residues after freezing conserved/active-site),
            # with a little noise — so the demo shows the real conservative->aggressive spread.
            mutable = int(L * 0.70)
            n_mut = max(1, int(round(mutable * s["target_mut_frac"] * random.uniform(0.85, 1.15))))
            mut = mutate(WT, min(n_mut, L))
            did = f"{ph[:4]}_{k+1}"
            (jd / "structures" / f"{did}.pdb").write_text(fake_pdb(mut, jitter=1.5))
            # demo: one design per phenotype (the last) shows the "active site not
            # assigned" fallback so the warning path is exercised in the UI.
            as_assigned = not (k == len(scheds) - 1 and len(scheds) > 1)
            active_site = DEMO_ACTIVE_SITE if as_assigned else []
            cons = fake_conservation(L, active_site, seed=hash((ph, k)) & 0xffff)
            track = build_track(WT, mut, cons, active_site, as_assigned)
            # bolder designs: higher phenotype score, but higher active-site RMSD (risk)
            designs.append(dict(
                design_id=did, sequence=mut, highlighted_seq=highlight(WT, mut),
                track=track,
                classifier_score=round(min(0.99, 0.55 + 0.09 * s["level"] + random.uniform(-0.04, 0.04)), 3),
                active_site_rmsd=round(0.15 + 0.13 * s["level"] + random.uniform(-0.05, 0.05), 2),
                n_mutations=n_mut, structure_file=f"{did}.pdb",
                aggressiveness=s["label"], aggressiveness_level=s["level"],
                metrics={"aggressiveness": s["label"],
                         "mut %": f"{round(100*n_mut/L,1)}",
                         "pLDDT": round(random.uniform(70, 92), 1),
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
