#!/bin/bash
# Stage 14 - submit the whole assembly chain as ONE dependency-linked series.
#
# Rebuilds every input stage 06 needs, then assembles the pair table at the
# measured per-scope identity thresholds (whole_proteome 40%, secreted 50%;
# job 1164334).
#
#   A 01b/03/03b/03c  combined labels: r232 metadata flags + GenomeSPOT +
#                     measured OGT + the 330 ingested MAGs
#   B 04          genome-pair tables, per class, with the locked confidence tiers
#   C 05agg       aggregate 6.64M SignalP predictions -> secreted table
#   D 07a         cluster the SECRETOME at 50%   (secreted-scope classes)
#   E 07b         cluster the WHOLE PROTEOME at 40% (temperature classes)
#   F 06          assemble: per-scope pair derivation + merged leakage-safe split
#
# Each step is afterok-dependent on the previous, so a failure halts the chain
# rather than feeding garbage forward. Every step writes a completion marker and
# re-runs are idempotent (03b/03c drop their own prior output columns/rows).
#
# WHY 07a AND 07b ARE SEPARATE JOBS: clustering at 40% is NOT a coarsening of
# clustering at 50%. Measured on the probe proteome at production --cov-mode 0,
# 1,421 of 34,752 multi-member 50% clusters (4.09%) are split across two or more
# 40% clusters, because mmseqs picks different representatives per threshold. Both
# maps are therefore needed as independent inputs, and stage 06 groups the split on
# their union-find merge (merge_cluster_maps) so no co-clustering relation in
# either map can straddle a fold boundary.
set -uo pipefail

REPO=/groups/cress/projects/jaymin/eptrans_scratch/repo
S=/groups/cress/projects/jaymin/eptrans_scratch
P=/groups/cress/projects/jaymin/IS1111/eptrans
W=$S/assemble
LOGS=$S/logs/assemble
OGT_DIR=$S/data/ogt
mkdir -p "$W" "$LOGS" "$OGT_DIR"

# Absolute interpreter: `conda` is NOT on the non-login PATH on biotite and bare
# `python` does not exist there either (only /usr/bin/python3, which lacks pandas).
# Verified by smoke test: this interpreter has pandas 3.0.3 / pyarrow 25.0.0 / matplotlib 3.11.1 (four stages need mpl).
PY=/home/jayminp/miniconda3/envs/eptrans_ml/bin/python


# ---- preflight, run on the LOGIN node before any job is submitted ----
# The OGT sources are fetched over the network, and compute nodes have NO egress
# (login node does). 03a is therefore run here rather than inside the chain.
$PY "$REPO/scripts/03a_fetch_ogt_sources.py" --ogt-dir "$OGT_DIR" || {
  echo "FATAL: could not fetch OGT sources (needed by 03b)"; exit 1; }

# 03c needs the merged deep-sea MAG table (taxonomy + GenomeSPOT + isolation).
if [ ! -s "$W/deepsea_mags_merged.tsv" ]; then
  echo "FATAL: $W/deepsea_mags_merged.tsv is missing."
  echo "  Stage it first: it is built locally and is the input 03c joins the 330"
  echo "  ingested MAGs through (--id-map $S/mag_ingest/map.tsv)."
  exit 1
fi

# Partition: overridable, defaults to `memory`. Chosen 2026-08-04 on queue depth,
# not idle capacity -- both partitions showed 0 idle CPUs, but the pending backlog
# was 11 jobs on `memory` against 139 on `standard` (12.6x shorter), and memory's
# nodes carry 677 GB against standard's 258 GB, which suits the 40% clustering step.
# `memory` is also time-unlimited on this cluster.
PART="${PART:-gpu_h200}"

# Early stages (A0/A/B/C) are CPU-only, modest-memory python steps. When every CPU
# partition is fully allocated -- measured 2026-08-04: standard 3680/0/0, memory
# 3008/0/0, high-memory 792/0/0, all zero idle -- the GPU partitions still had large
# idle CPU counts (gpu 262 idle of 352, gpu_h200 186 of 224) because they are gated on
# GPUs, not cores. Running the CPU-only stages there costs no GPU and starts hours
# sooner. The clustering and assembly stages stay on $PART for the RAM.
LIGHT_PART="${LIGHT_PART:-gpu_h200}"

# HEAVY stages: gpu_h200's single node carries 2,063,701 MB (2 TB) with 186 of 224
# CPUs idle, which comfortably exceeds the largest request here (320 G / 48 CPU) and
# beats the fully-allocated `memory` partition on start time. No GPU is requested --
# verified these jobs allocate cpu-only (AllocTRES=cpu=N, no gres), MaxTime=UNLIMITED
# and AllowAccounts=ALL on that partition.
#
# NOTE ON --mem HERE: biotite runs SelectTypeParameters=CR_CPU, so memory is NOT a
# consumable resource -- `--mem` does not reserve RAM and does not gate scheduling.
# It is kept as documentation of the expected footprint, but it means a heavy job can
# be co-scheduled with others onto the same node and OOM. Preferring the 2 TB node is
# therefore a real safety margin, not just a queue-time optimisation.
HEAVY_PART="${HEAVY_PART:-gpu_h200}"
HEAVY="--partition=$HEAVY_PART --output=$LOGS/%x_%j.out"
LIGHT="--partition=$LIGHT_PART --output=$LOGS/%x_%j.out"

# ROUTED TO gpu_h200 (2026-08-05, user directive). Measured at submit time: standard
# 53/0/0/53, memory 39/0/0/39, high-memory 3/0/0/3 and every min_* partition were
# 100%% allocated with ZERO idle nodes, while gpu_h200's node-224-2t-8gpu-1 held 196
# idle CPUs and 121 GB free. All three tiers therefore point at gpu_h200: it is the
# only partition with capacity, it carries 2 TB, and these stages request no GPU.
# Override per-run with PART=/LIGHT_PART=/HEAVY_PART= if capacity shifts.
SB="sbatch --parsable"
COMMON="--partition=$PART --output=$LOGS/%x_%j.out"

# ---------------------------------------------------------------- A0: GTDB metadata
# 01b needs a compact TSV (domain, accession, isolation, organism, taxonomy). The
# stage-01 metadata parquet is no longer on disk, so rebuild the compact form
# straight from the two GTDB metadata dumps.
G=/groups/cress/projects/jaymin/IS1111/gtdb
A0=$($SB $LIGHT --job-name=asm_meta --cpus-per-task=4 --mem=32G --time=01:00:00 \
  --wrap "set -u; cd $W && $PY - <<'PYEOF'
import csv, gzip, sys
G='/groups/cress/projects/jaymin/IS1111/gtdb'
W='/groups/cress/projects/jaymin/eptrans_scratch/assemble'
WANT=['accession','gtdb_taxonomy','ncbi_isolation_source','ncbi_organism_name',
      'gtdb_representative','checkm_completeness','checkm_contamination',
      'checkm2_completeness','checkm2_contamination']
n=0
with open(f'{W}/gtdb_meta.tsv','w',newline='') as out:
    wr=csv.writer(out,delimiter='\t')
    wr.writerow(['domain']+WANT)
    for dom,f in (('archaea','ar53_metadata_r232.tsv.gz'),
                  ('bacteria','bac120_metadata_r232.tsv.gz')):
        with gzip.open(f'{G}/{f}','rt',newline='') as fh:
            rd=csv.DictReader(fh,delimiter='\t')
            for row in rd:
                wr.writerow([dom]+[row.get(k,'') for k in WANT]); n+=1
print('META_ROWS',n)
PYEOF
    wc -l < $W/gtdb_meta.tsv && touch $W/.A0_done")
echo "A0 gtdb meta    $A0"

# ---------------------------------------------------------------- A: labels
A=$($SB $LIGHT --job-name=asm_labels --cpus-per-task=8 --mem=64G --time=02:00:00 \
  --dependency=afterok:$A0 \
  --wrap "set -u; cd $REPO && export PYTHONPATH=$REPO/src && \
    $PY scripts/01b_flag_metadata.py \
      --tsv $W/gtdb_meta.tsv --out $W/metadata_flags.parquet \
      --fig $W/metadata_flags_counts.png && \
    $PY scripts/03_combine_bins.py \
      --flags $W/metadata_flags.parquet \
      --predictions $P/genomespot_predictions_r232.tsv \
      --bare-join --out $W/labels_a.parquet && \
    $PY scripts/03b_merge_measured_ogt.py \
      --ogt-dir $OGT_DIR \
      --labels $W/labels_a.parquet --out-labels $W/labels_b.parquet \
      --out-pooled $W/pooled_measured_ogt.tsv \
      --out-stats $W/measured_ogt_merge.stats.json && \
    $PY scripts/03c_merge_deepsea_mags.py \
      --labels $W/labels_b.parquet \
      --mags $W/deepsea_mags_merged.tsv \
      --id-map $S/mag_ingest/map.tsv \
      --out $W/combined_labels.parquet --stats $W/labels.stats.json && \
    touch $W/.A_done")
echo "A labels        $A"

# ---------------------------------------------------------------- B: genome pairs
# Locked tiers (2026-08-04): thermophile HIGH ONLY, everything else high+medium.
# max_total_per_class uncapped -- the config default of 100 is a pilot setting.
B=$($SB $LIGHT --job-name=asm_pairs --cpus-per-task=8 --mem=64G --time=02:00:00 \
  --dependency=afterok:$A \
  --wrap "set -u; cd $REPO && export PYTHONPATH=$REPO/src && \
    $PY scripts/04_select_genomes.py --labels $W/combined_labels.parquet \
      --classes thermophile --confidence high --require-col has_proteome \
      --max-total-per-class 1000000000 --max-per-sample 5 --reuse-outgroups \
      --out-prefix $W/sel_thermophile && \
    for CLS in halophile acidophile alkaliphile hyperthermophile psychrophile; do \
      $PY scripts/04_select_genomes.py --labels $W/combined_labels.parquet \
        --classes \$CLS --confidence high,medium --require-col has_proteome \
        --max-total-per-class 1000000000 --max-per-sample 5 --reuse-outgroups \
        --out-prefix $W/sel_\$CLS || exit 1; done && \
    head -1 $W/sel_thermophile.pairs.tsv > $W/all_pairs.tsv && \
    for f in $W/sel_*.pairs.tsv; do tail -n +2 \$f >> $W/all_pairs.tsv; done && \
    wc -l < $W/all_pairs.tsv && touch $W/.B_done")
echo "B genome pairs  $B"

# ---------------------------------------------------------------- C: secreted table
C=$($SB $LIGHT --job-name=asm_secreted --cpus-per-task=16 --mem=96G --time=04:00:00 \
  --dependency=afterok:$B \
  --wrap "set -u; cd $REPO && export PYTHONPATH=$REPO/src && \
    $PY -c \"
import pandas as pd, yaml, sys
cfg=yaml.safe_load(open('config/config.yaml'))
scope=cfg['dataset']['protein_scope']
# The per-class map is nested under 'by_phenotype'; the top level holds only
# 'default' and 'by_phenotype'. Iterating the top level silently yields [] and
# produces an EMPTY whole-proteome FASTA, which stage E then hard-fails on.
# This matches how 06_assemble_dataset.py reads it (ps.get('by_phenotype', {})).
by_ph=scope.get('by_phenotype') or {}
if not by_ph:
    sys.exit('FATAL: dataset.protein_scope.by_phenotype is empty')
whole=[k for k,v in by_ph.items() if v=='whole_proteome']
if not whole:
    sys.exit('FATAL: no class is whole_proteome scope; stage E would get an empty FASTA')
pr=pd.read_parquet('$W/combined_labels.parquet')
# Confidence tiers admitted to the whole-proteome MLM corpus, per class.
#
# WHY psychrophile EXCLUDES low (decided 2026-08-05, measured):
#   * habitat-keyword-only cold evidence has a 0.14-0.63%% hit rate across four
#     independent populations (n=1452/639/443/1416), so low-tier psychrophile is
#     ~99%% label noise -- the weakest-signal class carrying the worst evidence.
#   * it dominated the corpus: 4,690 genomes / 6,262,304 seqs = 57.6%% of the
#     10.87M-seq whole FASTA.
#   * dropping it costs ZERO protein pairs: 1,702,137 of its clusters contain no
#     non-low member, and a cluster with no outgroup member can never form a
#     matched pair. 82.1%% of those clusters are singletons (mean size 1.52).
#   * 58.8%% of its sequences were redundant anyway (in clusters shared with
#     retained genomes).
# hyperthermophile keeps low: its cold-end evidence problem does not apply, and
# it is the class most starved of data (293 genome pairs).
WHOLE_CONF={'psychrophile':('high','medium'),'hyperthermophile':('high','medium','low')}
m=pd.Series(False,index=pr.index)
conf=pr['final_confidence'].astype(str) if 'final_confidence' in pr.columns else None
for cl in whole:
    c='final_'+cl
    if c not in pr.columns: continue
    cm=pr[c].fillna(False).astype(bool)
    tiers=WHOLE_CONF.get(cl)
    if tiers is not None and conf is not None:
        cm &= conf.isin(tiers)
    m |= cm
    print('WHOLE_SCOPE_TIER',cl,tiers or 'all',int(cm.sum()))
# outgroups of whole-scope classes need full proteomes too
p=pd.read_csv('$W/all_pairs.tsv',sep='\\t')
acc=set(pr.loc[m,'accession'].astype(str))
if 'class' in p.columns:
    # Pair members are added UNCONDITIONALLY: stage B already applied the
    # confidence gate when selecting them, and omitting one here would silently
    # void a matched pair (the exact failure the has_proteome gate fixed).
    sel=p[p['class'].isin(whole)]
    n_before=len(acc)
    acc|=set(sel['extremophile_acc'].astype(str))|set(sel['outgroup_acc'].dropna().astype(str))
    print('WHOLE_SCOPE_PAIR_MEMBERS_ADDED',len(acc)-n_before)
open('$W/whole_scope_accessions.txt','w').write('\\n'.join(sorted(a for a in acc if a and a!='nan'))+'\\n')
print('WHOLE_SCOPE_GENOMES',len(acc),'classes',whole)
\" && \
    $PY scripts/05_aggregate_signalp.py \
      --pred-dirs '$S/signalp_targeted/chunk_*' '$S/signalp_gtdb_fill/chunk_*' \
      --legacy $P/secreted_proteins_r232.tsv \
      --proteome-root $P/../gtdb/protein_faa_reps \
      --extra-proteome-root $P/../custom_genomes/protein_faa_reps \
      --whole-scope-accessions $W/whole_scope_accessions.txt \
      --faa-secreted $W/secretome.faa --faa-whole $W/wholeproteome.faa \
      --out $W/secreted_all.tsv --stats $W/secreted.stats.json && \
    touch $W/.C_done")
echo "C secreted      $C"

# ---------------------------------------------------------------- D/E: clustering
MM=/shared/software/bin/mmseqs
D=$($SB $HEAVY --job-name=asm_clu50 --cpus-per-task=48 --mem=200G --time=12:00:00 \
  --dependency=afterok:$C \
  --wrap "set -u; cd $W && \
    test -s secretome.faa || { echo \"FATAL: secretome.faa missing or empty -- stage C did not emit it\"; exit 1; }; \\
    $MM easy-cluster secretome.faa clu50 tmp50 \
      --min-seq-id 0.5 -c 0.8 --cov-mode 0 --threads 48 && \
    rm -rf tmp50 && wc -l < clu50_cluster.tsv && touch $W/.D_done")
echo "D cluster 50%   $D"

E=$($SB $HEAVY --job-name=asm_clu40 --cpus-per-task=48 --mem=320G --time=16:00:00 \
  --dependency=afterok:$C \
  --wrap "set -u; cd $W && \
    test -s wholeproteome.faa || { echo \"FATAL: wholeproteome.faa missing or empty -- stage C did not emit it\"; exit 1; }; \\
    $MM easy-cluster wholeproteome.faa clu40 tmp40 \
      --min-seq-id 0.4 -c 0.8 --cov-mode 0 --threads 48 && \
    rm -rf tmp40 && wc -l < clu40_cluster.tsv && touch $W/.E_done")
echo "E cluster 40%   $E"

# ---------------------------------------------------------------- F: assemble
F=$($SB $HEAVY --job-name=asm_final --cpus-per-task=16 --mem=256G --time=06:00:00 \
  --dependency=afterok:$D:$E \
  --wrap "set -u; cd $REPO && export PYTHONPATH=$REPO/src && \
    $PY scripts/06_assemble_dataset.py \
      --secreted $W/secreted_all.tsv \
      --labels $W/combined_labels.parquet \
      --pairs $W/all_pairs.tsv \
      --cluster-map-named id50=$W/clu50_cluster.tsv \
      --cluster-map-named id40=$W/clu40_cluster.tsv \
      --protein-scope \
      --out $W/labeled_dataset.parquet \
      --fig $W/dataset_splits.png && \
    touch $W/.F_done")
echo "F assemble      $F"

echo
echo "CHAIN: $A0 -> $A -> $B -> $C -> {$D,$E} -> $F"
echo "$A0 $A $B $C $D $E $F" > "$W/chain_jobids.txt"
