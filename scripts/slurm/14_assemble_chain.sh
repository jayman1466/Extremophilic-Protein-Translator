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
PART="${PART:-memory}"

SB="sbatch --parsable"
COMMON="--partition=$PART --output=$LOGS/%x_%j.out"

# ---------------------------------------------------------------- A0: GTDB metadata
# 01b needs a compact TSV (domain, accession, isolation, organism, taxonomy). The
# stage-01 metadata parquet is no longer on disk, so rebuild the compact form
# straight from the two GTDB metadata dumps.
G=/groups/cress/projects/jaymin/IS1111/gtdb
A0=$($SB $COMMON --job-name=asm_meta --cpus-per-task=4 --mem=32G --time=01:00:00 \
  --wrap "set -uo pipefail; cd $W && $PY - <<'PYEOF'
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
A=$($SB $COMMON --job-name=asm_labels --cpus-per-task=8 --mem=64G --time=02:00:00 \
  --dependency=afterok:$A0 \
  --wrap "set -uo pipefail; cd $REPO && export PYTHONPATH=$REPO/src && \
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
B=$($SB $COMMON --job-name=asm_pairs --cpus-per-task=8 --mem=64G --time=02:00:00 \
  --dependency=afterok:$A \
  --wrap "set -uo pipefail; cd $REPO && export PYTHONPATH=$REPO/src && \
    $PY scripts/04_select_genomes.py --labels $W/combined_labels.parquet \
      --classes thermophile --confidence high \
      --max-total-per-class 1000000000 --reuse-outgroups \
      --out-prefix $W/sel_thermophile && \
    for CLS in halophile acidophile alkaliphile hyperthermophile psychrophile; do \
      $PY scripts/04_select_genomes.py --labels $W/combined_labels.parquet \
        --classes \$CLS --confidence high,medium \
        --max-total-per-class 1000000000 --reuse-outgroups \
        --out-prefix $W/sel_\$CLS || exit 1; done && \
    head -1 $W/sel_thermophile.pairs.tsv > $W/all_pairs.tsv && \
    for f in $W/sel_*.pairs.tsv; do tail -n +2 \$f >> $W/all_pairs.tsv; done && \
    wc -l < $W/all_pairs.tsv && touch $W/.B_done")
echo "B genome pairs  $B"

# ---------------------------------------------------------------- C: secreted table
C=$($SB $COMMON --job-name=asm_secreted --cpus-per-task=16 --mem=96G --time=04:00:00 \
  --dependency=afterok:$B \
  --wrap "set -uo pipefail; cd $REPO && export PYTHONPATH=$REPO/src && \
    $PY -c \"
import pandas as pd, yaml, sys
cfg=yaml.safe_load(open('config/config.yaml'))
scope=cfg['dataset']['protein_scope']
whole=[k for k,v in scope.items() if k!='default' and v=='whole_proteome']
pr=pd.read_parquet('$W/combined_labels.parquet')
m=pd.Series(False,index=pr.index)
for cl in whole:
    c='final_'+cl
    if c in pr.columns: m |= pr[c].fillna(False).astype(bool)
# outgroups of whole-scope classes need full proteomes too
p=pd.read_csv('$W/all_pairs.tsv',sep='\\t')
acc=set(pr.loc[m,'accession'].astype(str))
if 'class' in p.columns:
    sel=p[p['class'].isin(whole)]
    acc|=set(sel['extremophile_acc'].astype(str))|set(sel['outgroup_acc'].dropna().astype(str))
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
D=$($SB $COMMON --job-name=asm_clu50 --cpus-per-task=48 --mem=200G --time=12:00:00 \
  --dependency=afterok:$C \
  --wrap "set -uo pipefail; cd $W && \
    test -s secretome.faa || { echo \"FATAL: secretome.faa missing or empty -- stage C did not emit it\"; exit 1; }; \\
    $MM easy-cluster secretome.faa clu50 tmp50 \
      --min-seq-id 0.5 -c 0.8 --cov-mode 0 --threads 48 && \
    rm -rf tmp50 && wc -l < clu50_cluster.tsv && touch $W/.D_done")
echo "D cluster 50%   $D"

E=$($SB $COMMON --job-name=asm_clu40 --cpus-per-task=48 --mem=320G --time=16:00:00 \
  --dependency=afterok:$C \
  --wrap "set -uo pipefail; cd $W && \
    test -s wholeproteome.faa || { echo \"FATAL: wholeproteome.faa missing or empty -- stage C did not emit it\"; exit 1; }; \\
    $MM easy-cluster wholeproteome.faa clu40 tmp40 \
      --min-seq-id 0.4 -c 0.8 --cov-mode 0 --threads 48 && \
    rm -rf tmp40 && wc -l < clu40_cluster.tsv && touch $W/.E_done")
echo "E cluster 40%   $E"

# ---------------------------------------------------------------- F: assemble
F=$($SB $COMMON --job-name=asm_final --cpus-per-task=16 --mem=256G --time=06:00:00 \
  --dependency=afterok:$D:$E \
  --wrap "set -uo pipefail; cd $REPO && export PYTHONPATH=$REPO/src && \
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
