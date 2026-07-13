#!/usr/bin/env python
"""Stage 3 (annotation transfer): foldseek + M-CSA/Swiss-Prot -> active-site freeze.

Transfers catalytic / functional-site annotations from annotated homologs onto the
WT query, in UNIPROT COORDINATE SPACE (the trick that makes this clean):

  * foldseek easy-search wt.pdb vs the local AlphaFold DB returns hits whose target
    IDs embed a UniProt accession (AF-<ACC>-F1-...). AlphaFold models are 1:1 with
    the canonical UniProt sequence, so the foldseek target residue index == UniProt
    sequence position.
  * Swiss-Prot site features (ACT_SITE / BINDING / SITE) are given as UniProt
    positions. M-CSA catalytic residues are mapped to UniProt positions too.
  * So for each AF hit accession we look up its annotated UniProt positions, walk the
    foldseek alignment (qaln/taln anchored at qstart/tstart) to map each annotated
    target position -> the aligned query position, and freeze it.

Optionally also uses the Stage-1 mmseqs MSA (uniprot_kb hits) as a SEQUENCE channel:
same UniProt-coordinate transfer, no structure needed. Structural (foldseek) hits are
higher-confidence; sequence hits add recall.

Writes active_site_transfer.json:
  {transferred: [1-based query positions],
   by_source: {foldseek_swissprot:[...], foldseek_mcsa:[...], mmseqs_swissprot:[...]},
   n_foldseek_hits, n_mmseqs_hits, detail:[{pos, source, homolog, feature}...]}

Runs offline (foldseek DBs + annotation tables are local) in any env with foldseek on
PATH (e.g. eptrans_ml). Annotation tables are pre-staged once on the login node.
"""
import sys, os, json, argparse, subprocess, re, gzip
from pathlib import Path


def otsu_threshold(values):
    """Parameter-free split of a 1-D integer distribution (Otsu 1979).

    Given per-position homolog vote counts, find the integer cut t that MAXIMISES
    between-class variance (equivalently minimises within-class variance) of the two
    groups {v < t} and {v >= t}. This adapts per enzyme to however many homologs were
    found and however deeply they're annotated: a deeply-annotated protein with a high
    catalytic-cluster peak gets a high cut; a sparsely-annotated one gets a low cut.
    Returns the smallest vote count to KEEP (the >= side). If the distribution has no
    spread (all equal) returns that single value.
    """
    if not values:
        return 1
    vmin, vmax = min(values), max(values)
    if vmin == vmax:
        return vmin
    # histogram over integer vote counts
    n = len(values)
    hist = {}
    for v in values:
        hist[v] = hist.get(v, 0) + 1
    total_sum = sum(values)
    best_t, best_var = vmin + 1, -1.0
    w0 = 0        # count below t
    sum0 = 0      # weighted sum below t
    # candidate thresholds t split into [<t] and [>=t]; iterate t over vmin+1..vmax
    for t in range(vmin, vmax + 1):
        # move class boundary: everything == (t-1) joins the "below" class
        c = hist.get(t - 1, 0)
        w0 += c
        sum0 += c * (t - 1)
        w1 = n - w0
        if w0 == 0 or w1 == 0:
            continue
        m0 = sum0 / w0
        m1 = (total_sum - sum0) / w1
        var_between = w0 * w1 * (m0 - m1) ** 2   # /n^2 const, irrelevant to argmax
        if var_between > best_var:
            best_var, best_t = var_between, t
    return best_t


def parse_swissprot_sites(tsv_gz):
    """{accession: set(uniprot_pos)} from a UniProt TSV with ft_act_site/ft_binding/
    ft_site columns. Feature strings look like 'ACT_SITE 195; /note=...; BINDING 41..43;'.
    We take every integer position (and both ends of ranges) as a functional site."""
    sites = {}
    if not Path(tsv_gz).exists():
        return sites
    op = gzip.open if str(tsv_gz).endswith(".gz") else open
    with op(tsv_gz, "rt") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        # feature columns are everything after 'Entry'
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if not parts or not parts[0]:
                continue
            acc = parts[0]
            pos = set()
            for cell in parts[1:]:
                # positions: 'ACT_SITE 195', 'BINDING 41..43'
                for m in re.finditer(r"(?:ACT_SITE|BINDING|SITE|METAL)\s+(\d+)(?:\.\.(\d+))?", cell):
                    a = int(m.group(1)); b = int(m.group(2)) if m.group(2) else a
                    pos.update(range(a, b + 1))
            if pos:
                sites[acc] = pos
    return sites


def parse_mcsa(json_path):
    """{accession: set(uniprot_pos)} of catalytic residues, tolerant to schema.
    M-CSA /residues/ entries carry residue_sequences with UniProt id + resid."""
    mcsa = {}
    if not Path(json_path).exists():
        return mcsa
    try:
        data = json.loads(Path(json_path).read_text())
    except Exception:
        return mcsa
    records = data.get("results", data) if isinstance(data, dict) else data
    for r in (records or []):
        if not isinstance(r, dict):
            continue
        for rs in (r.get("residue_sequences") or r.get("residue_chains") or []):
            acc = rs.get("uniprot_id") or rs.get("uniprot_accession") or rs.get("uniprot")
            resid = rs.get("resid") or rs.get("uniprot_position") or rs.get("auth_resid")
            if acc and resid:
                try:
                    mcsa.setdefault(acc, set()).add(int(resid))
                except (TypeError, ValueError):
                    pass
    return mcsa


def transfer_from_alignment(qstart, qaln, tstart, taln, annotated_tpos):
    """Walk a pairwise alignment; for each annotated TARGET UniProt position present
    in annotated_tpos, return the aligned 1-based QUERY position."""
    out = []
    qp, tp = qstart, tstart  # both 1-based starts
    for qc, tc in zip(qaln, taln):
        q_here = (qc != "-")
        t_here = (tc != "-")
        if t_here and q_here and tp in annotated_tpos:
            out.append(qp)
        if q_here:
            qp += 1
        if t_here:
            tp += 1
    return out


def foldseek_search(wt_pdb, db, workdir, max_seqs=300):
    """foldseek easy-search vs the AlphaFold DB -> list of
    (acc, qstart, qaln, tstart, taln, bits).

    BOUNDED for a single query against the 645 GB AF DB (the search stalled unbounded):
      --split-memory-limit  pages the huge target index through RAM in chunks (the OOM
                            guardrail, same as the mmseqs MSA fix).
      --prefilter-mode 1    UNGAPPED prefilter — much cheaper first pass; only structurally
                            similar targets reach the expensive gapped alignment.
      -s (sensitivity)      lowered from the 9.5 default to 7.0: for active-site transfer
                            we want clear structural homologs, not the deep twilight zone,
                            so a faster, less exhaustive search suffices.
      --max-seqs            cap on aligned targets (default 300 homologs is ample).
    Runtime knobs overridable via env (FOLDSEEK_MEM_LIMIT / FOLDSEEK_SENS).
    """
    m8 = Path(workdir) / "fs_hits.m8"
    tmp = Path(workdir) / "fs_tmp"
    mem_cap = os.environ.get("FOLDSEEK_MEM_LIMIT", "80G")
    sens = os.environ.get("FOLDSEEK_SENS", "7.0")
    cmd = ["foldseek", "easy-search", str(wt_pdb), db, str(m8), str(tmp),
           "--format-output", "query,target,qstart,qend,tstart,tend,qaln,taln,bits",
           "--max-seqs", str(max_seqs), "-e", "1e-3", "-s", sens,
           "--prefilter-mode", "1", "--split-memory-limit", mem_cap,
           "--threads", str(os.cpu_count() or 8)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=3600)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"[11a] foldseek failed ({type(e).__name__}): "
              f"{getattr(e,'stderr','')[:200] if hasattr(e,'stderr') else ''}", flush=True)
        return []
    hits = []
    for ln in m8.read_text().splitlines():
        p = ln.split("\t")
        if len(p) < 9:
            continue
        target = p[1]
        # AF DB target: AF-<ACC>-F1-model_v6  ->  <ACC>
        m = re.match(r"AF-([A-Z0-9]+)-F\d+", target)
        acc = m.group(1) if m else target.split("-")[0]
        hits.append((acc, int(p[2]), p[6], int(p[4]), p[7], float(p[8])))
    return hits


def mmseqs_hits_from_m8(m8_path):
    """Reuse the Stage-1 mmseqs m8 (query,target,pident,qstart,qend,qaln,taln).
    Target is a uniprot_kb id like 'sp|P00445|SODC_HUMAN' or 'UniRef...'. Extract acc."""
    hits = []
    if not m8_path or not Path(m8_path).exists():
        return hits
    for ln in Path(m8_path).read_text().splitlines():
        p = ln.split("\t")
        if len(p) < 7:
            continue
        tgt = p[1]
        m = re.search(r"[A-Z0-9]{6,10}", tgt.split("|")[1]) if "|" in tgt else re.search(r"([A-Z0-9]{6,10})", tgt)
        acc = (tgt.split("|")[1] if "|" in tgt else (m.group(1) if m else tgt))
        # mmseqs m8 here has no tstart/taln for target coords; approximate target pos
        # by counting from 1 (uniprot_kb targets are full sequences) using taln.
        hits.append((acc, int(p[3]), p[5], 1, p[6]))  # (acc,qstart,qaln,tstart~1,taln)
    return hits


def swissprot_search(seq, sprot_db, workdir, max_seqs=500):
    """mmseqs easy-search of the WT SEQUENCE vs a reviewed Swiss-Prot mmseqs DB ->
    list of (acc, qstart, qaln, tstart, taln). This is the PRIMARY annotation channel:
    the structural foldseek-vs-AF search returns mostly UNREVIEWED TrEMBL neighbours
    with no site annotations, whereas Swiss-Prot homologs carry ACT_SITE/BINDING/SITE.
    Swiss-Prot DB targets are '<db>|<ACC>|<name>' or bare ACC; both are parsed to ACC."""
    qf = Path(workdir) / "sp_query.fasta"
    qf.write_text(f">query\n{seq}\n")
    m8 = Path(workdir) / "sp_hits.m8"
    tmp = Path(workdir) / "sp_tmp"
    mem_cap = os.environ.get("MMSEQS_MEM_LIMIT", "80G")
    cmd = ["mmseqs", "easy-search", str(qf), sprot_db, str(m8), str(tmp),
           "--format-output", "query,target,pident,qstart,qend,qaln,taln",
           "-s", "5.7", "--max-seqs", str(max_seqs), "-e", "1e-3",
           "--split-memory-limit", mem_cap, "--threads", str(os.cpu_count() or 8)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=1800)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"[11a] swissprot mmseqs failed ({type(e).__name__}): "
              f"{getattr(e,'stderr','')[:200] if hasattr(e,'stderr') else ''}", flush=True)
        return []
    hits = []
    for ln in m8.read_text().splitlines():
        p = ln.split("\t")
        if len(p) < 7:
            continue
        target = p[1]
        # '<db>|<ACC>|<entry>' (UniProt FASTA header) or bare ACC
        acc = target.split("|")[1] if "|" in target else target
        hits.append((acc, int(p[3]), p[5], 1, p[6]))
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wt-pdb", required=True)
    ap.add_argument("--wt-seq", default="", help="WT sequence for the Swiss-Prot seq channel")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ann-dir", required=True, help="dir with swissprot_sites.tsv.gz + mcsa json")
    ap.add_argument("--foldseek-af-db", default="/shared/db/foldseek/latest/db/alphafold_uniprot")
    ap.add_argument("--swissprot-db", default="",
                    help="reviewed Swiss-Prot mmseqs DB stem (PRIMARY seq channel)")
    ap.add_argument("--no-foldseek", action="store_true",
                    help="skip the slow foldseek-vs-AF structural channel")
    ap.add_argument("--consensus-frac", type=float, default=0.0,
                    help="manual override: keep positions with vote count >= this fraction "
                         "of MAX support. Default 0 -> use adaptive Otsu threshold instead.")
    ap.add_argument("--consensus-min-votes", type=int, default=2,
                    help="absolute floor on supporting homologs to keep a position (default 2)")
    ap.add_argument("--mmseqs-m8", default="", help="optional Stage-1 mmseqs hits.m8 for seq channel")
    ap.add_argument("--workdir", default=".")
    ap.add_argument("--seq-len", type=int, required=True)
    args = ap.parse_args()

    Path(args.workdir).mkdir(parents=True, exist_ok=True)
    swissprot = parse_swissprot_sites(Path(args.ann_dir) / "swissprot_sites.tsv.gz")
    mcsa = parse_mcsa(Path(args.ann_dir) / "mcsa_catalytic_residues.json")
    print(f"[11a] annotation tables: swissprot accs={len(swissprot)}, mcsa accs={len(mcsa)}", flush=True)

    by_source = {"swissprot_seq": set(), "foldseek_swissprot": set(),
                 "foldseek_mcsa": set(), "mmseqs_swissprot": set()}
    detail = []

    support = {}                 # qpos -> set(homolog accs) supporting it
    contributing = set()         # homologs (accs) that transferred >=1 annotated position

    def _transfer(hit_iter, src, table):
        n = 0
        for acc, qstart, qaln, tstart, taln in hit_iter:
            tpos = table.get(acc)
            if not tpos:
                continue
            got = False
            for qp in transfer_from_alignment(qstart, qaln, tstart, taln, tpos):
                if 1 <= qp <= args.seq_len:
                    by_source[src].add(qp)
                    support.setdefault(qp, set()).add(acc)
                    detail.append(dict(pos=qp, source=src, homolog=acc))
                    n += 1
                    got = True
            if got:
                contributing.add(acc)
        return n

    # --- PRIMARY channel: mmseqs WT-sequence vs reviewed Swiss-Prot DB ---
    sp = []
    if args.swissprot_db and args.wt_seq:
        sp = swissprot_search(args.wt_seq, args.swissprot_db, args.workdir)
        print(f"[11a] Swiss-Prot seq hits: {len(sp)}", flush=True)
        _transfer(((a, qs, qa, ts, ta) for a, qs, qa, ts, ta in sp), "swissprot_seq", swissprot)

    # --- structural channel (optional, recall booster): foldseek vs AF DB ---
    fs = []
    if not args.no_foldseek:
        fs = foldseek_search(args.wt_pdb, args.foldseek_af_db, args.workdir)
        print(f"[11a] foldseek AF hits: {len(fs)}", flush=True)
        _transfer(((a, qs, qa, ts, ta) for a, qs, qa, ts, ta, _b in fs), "foldseek_swissprot", swissprot)
        _transfer(((a, qs, qa, ts, ta) for a, qs, qa, ts, ta, _b in fs), "foldseek_mcsa", mcsa)

    # --- legacy: reuse Stage-1 UniRef50 m8 (rarely joins; kept for completeness) ---
    mh = mmseqs_hits_from_m8(args.mmseqs_m8)
    if mh:
        print(f"[11a] Stage-1 m8 seq-channel hits: {len(mh)}", flush=True)
        _transfer(iter(mh), "mmseqs_swissprot", swissprot)

    raw_union = sorted(set().union(*by_source.values())) if any(by_source.values()) else []
    # Consensus filter. Each transferred position carries a vote count = number of distinct
    # annotated homologs whose (aligned) annotation landed there. Real catalytic/coordination
    # residues are recurrently annotated across the homolog set and rise to the TOP of the
    # support distribution; incidental single-homolog BINDING ranges sit at the bottom (a
    # long tail of 1-2 vote positions). We threshold on a fraction of the MAX observed
    # support (self-calibrating to how deep/annotated the homolog set is) with an absolute
    # floor of 2 votes, rather than on the count of contributing homologs (each homolog only
    # annotates its own few sites, so no position is supported by a large fraction of them).
    votes = {p: len(support.get(p, ())) for p in raw_union}
    max_sup = max(votes.values()) if votes else 0
    # Adaptive threshold: Otsu split of the vote distribution (parameter-free, per-enzyme).
    # A manual --consensus-frac override (fraction of max support) is honoured if given >0;
    # otherwise Otsu picks the natural break between the incidental-range tail and the
    # recurrently-annotated catalytic cluster. A hard floor of --consensus-min-votes always
    # applies (a single-homolog position is never trusted).
    if args.consensus_frac and args.consensus_frac > 0:
        keep_votes = max(args.consensus_min_votes, int(round(args.consensus_frac * max_sup)))
        method = f"frac={args.consensus_frac}"
    else:
        keep_votes = max(args.consensus_min_votes, otsu_threshold(list(votes.values())))
        method = "otsu"
    transferred = sorted(p for p in raw_union if votes[p] >= keep_votes)
    print(f"[11a] consensus filter ({method}): {len(raw_union)} raw -> {len(transferred)} kept "
          f"(>= {keep_votes} votes; max_support={max_sup}, floor={args.consensus_min_votes})",
          flush=True)
    out = dict(transferred=transferred, transferred_raw_union=raw_union,
               consensus_method=method, keep_votes=keep_votes, max_support=max_sup,
               n_contributing_homologs=len(contributing),
               vote_histogram={str(v): sum(1 for x in votes.values() if x == v)
                               for v in sorted(set(votes.values()))},
               position_support={str(p): votes[p] for p in transferred},
               by_source={k: sorted(v) for k, v in by_source.items()},
               n_swissprot_hits=len(sp), n_foldseek_hits=len(fs), n_mmseqs_hits=len(mh),
               detail=detail[:500])
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[11a] transferred {len(transferred)} active-site positions "
          f"(sp_seq={len(by_source['swissprot_seq'])}, "
          f"fs_sp={len(by_source['foldseek_swissprot'])}, fs_mcsa={len(by_source['foldseek_mcsa'])}, "
          f"ms_sp={len(by_source['mmseqs_swissprot'])}) -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
