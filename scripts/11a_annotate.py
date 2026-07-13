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
    """foldseek easy-search -> list of (acc, qstart, qaln, tstart, taln, bits)."""
    m8 = Path(workdir) / "fs_hits.m8"
    tmp = Path(workdir) / "fs_tmp"
    cmd = ["foldseek", "easy-search", str(wt_pdb), db, str(m8), str(tmp),
           "--format-output", "query,target,qstart,qend,tstart,tend,qaln,taln,bits",
           "--max-seqs", str(max_seqs), "-e", "1e-3", "--threads", str(os.cpu_count() or 8)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=2400)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wt-pdb", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ann-dir", required=True, help="dir with swissprot_sites.tsv.gz + mcsa json")
    ap.add_argument("--foldseek-af-db", default="/shared/db/foldseek/latest/db/alphafold_uniprot")
    ap.add_argument("--mmseqs-m8", default="", help="optional Stage-1 mmseqs hits.m8 for seq channel")
    ap.add_argument("--workdir", default=".")
    ap.add_argument("--seq-len", type=int, required=True)
    args = ap.parse_args()

    Path(args.workdir).mkdir(parents=True, exist_ok=True)
    swissprot = parse_swissprot_sites(Path(args.ann_dir) / "swissprot_sites.tsv.gz")
    mcsa = parse_mcsa(Path(args.ann_dir) / "mcsa_catalytic_residues.json")
    print(f"[11a] annotation tables: swissprot accs={len(swissprot)}, mcsa accs={len(mcsa)}", flush=True)

    by_source = {"foldseek_swissprot": set(), "foldseek_mcsa": set(), "mmseqs_swissprot": set()}
    detail = []

    # --- structural channel: foldseek vs AF DB ---
    fs = foldseek_search(args.wt_pdb, args.foldseek_af_db, args.workdir)
    print(f"[11a] foldseek AF hits: {len(fs)}", flush=True)
    for acc, qstart, qaln, tstart, taln, bits in fs:
        for src, table in (("foldseek_swissprot", swissprot), ("foldseek_mcsa", mcsa)):
            tpos = table.get(acc)
            if not tpos:
                continue
            qpos = transfer_from_alignment(qstart, qaln, tstart, taln, tpos)
            for qp in qpos:
                if 1 <= qp <= args.seq_len:
                    by_source[src].add(qp)
                    detail.append(dict(pos=qp, source=src, homolog=acc))

    # --- sequence channel: reuse Stage-1 mmseqs m8 (Swiss-Prot only) ---
    mh = mmseqs_hits_from_m8(args.mmseqs_m8)
    print(f"[11a] mmseqs seq-channel hits: {len(mh)}", flush=True)
    for acc, qstart, qaln, tstart, taln in mh:
        tpos = swissprot.get(acc)
        if not tpos:
            continue
        qpos = transfer_from_alignment(qstart, qaln, tstart, taln, tpos)
        for qp in qpos:
            if 1 <= qp <= args.seq_len:
                by_source["mmseqs_swissprot"].add(qp)
                detail.append(dict(pos=qp, source="mmseqs_swissprot", homolog=acc))

    transferred = sorted(set().union(*by_source.values())) if any(by_source.values()) else []
    out = dict(transferred=transferred,
               by_source={k: sorted(v) for k, v in by_source.items()},
               n_foldseek_hits=len(fs), n_mmseqs_hits=len(mh),
               detail=detail[:500])
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[11a] transferred {len(transferred)} active-site positions "
          f"(fs_sp={len(by_source['foldseek_swissprot'])}, fs_mcsa={len(by_source['foldseek_mcsa'])}, "
          f"ms_sp={len(by_source['mmseqs_swissprot'])}) -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
