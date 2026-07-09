#!/usr/bin/env bash
# Download the reference databases needed for the function-retention oracle
# (Oracle 2) that are NOT already staged system-wide on biotite.
#
# Already present on biotite (do NOT re-download; recorded for provenance):
#   uniref50   (2025-11-13) : /shared/db/uniref/uniref50/latest/uniref50.fasta   (23G)
#   uniprot db (2025-01)    : /shared/db/uniprot/latest/mmseqs/                   (137G, UniProtKB seqs)
#   pfam hmms  (r37)        : /shared/db/pfam/latest/Pfam-A.hmm                   (3.4G)
#   foldseek pdb (2026-02-04): /shared/db/foldseek/latest/db/pdb                  (71M)
#   foldseek af  (2026-02-04): /shared/db/foldseek/latest/db/alphafold_uniprot   (75G)
#
# This script stages the three that are missing, into $DB_ROOT.
# Run on biotite:  bash scripts/download_dbs.sh
set -euo pipefail

# Persistent reference-DB home (moved out of scratch 2026-07-09 — these are
# durable, repeatedly-invoked reference data, not disposable pipeline outputs).
DB_ROOT="${DB_ROOT:-/groups/cress/projects/jaymin/IS1111/eptrans/db}"
mkdir -p "$DB_ROOT"
STAMP="$(date +%Y_%m_%d)"
echo "[dbs] DB_ROOT=$DB_ROOT  pull-date=$STAMP"

# --------------------------------------------------------------------------
# 1. Swiss-Prot flat file (reviewed UniProtKB, WITH feature tables).
#    The system mmseqs DB holds sequences only — this text/XML copy carries the
#    ACT_SITE / BINDING / METAL / SITE feature annotations we key the active-site
#    ladder off. Compressed .dat is ~950 MB (~6 GB uncompressed).
# --------------------------------------------------------------------------
SP_DIR="$DB_ROOT/swissprot_$STAMP"
mkdir -p "$SP_DIR"
echo "[dbs] Swiss-Prot flat file -> $SP_DIR"
curl -L --fail -o "$SP_DIR/uniprot_sprot.dat.gz" \
  "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.dat.gz"
# release metadata (version + date)
curl -L --fail -o "$SP_DIR/reldate.txt" \
  "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/reldate.txt" || true

# --------------------------------------------------------------------------
# 2. InterPro (entry list + protein2ipr feature mapping).
#    entry.list = human-readable accession->name; protein2ipr.dat.gz maps every
#    UniProt accession to its InterPro entries WITH match positions (the
#    active-site/site positions we transfer). protein2ipr is large (~50 GB gz).
# --------------------------------------------------------------------------
IP_DIR="$DB_ROOT/interpro_$STAMP"
mkdir -p "$IP_DIR"
echo "[dbs] InterPro -> $IP_DIR"
curl -L --fail -o "$IP_DIR/entry.list" \
  "https://ftp.ebi.ac.uk/pub/databases/interpro/current_release/entry.list"
curl -L --fail -o "$IP_DIR/interpro.xml.gz" \
  "https://ftp.ebi.ac.uk/pub/databases/interpro/current_release/interpro.xml.gz"
# protein2ipr is optional + huge; uncomment if you want the full position mapping
# locally rather than querying the protein-annotation MCP connector per-enzyme:
# curl -L --fail -o "$IP_DIR/protein2ipr.dat.gz" \
#   "https://ftp.ebi.ac.uk/pub/databases/interpro/current_release/protein2ipr.dat.gz"

# --------------------------------------------------------------------------
# 3. M-CSA catalytic residues. Flat files are FROZEN (EBI: "not being updated
#    anymore, use the API"). We snapshot via the API for a dated, reproducible
#    copy — pulled by the companion python script so it can page + parse JSON.
# --------------------------------------------------------------------------
MCSA_DIR="$DB_ROOT/mcsa_$STAMP"
mkdir -p "$MCSA_DIR"
echo "[dbs] M-CSA -> $MCSA_DIR (run scripts/download_mcsa.py to snapshot the API)"
# also grab the frozen flat file as a secondary reference
curl -L --fail -o "$MCSA_DIR/curated_data.csv" \
  "https://www.ebi.ac.uk/thornton-srv/m-csa/media/flat_files/curated_data.csv" || \
  echo "[dbs] flat-file curated_data.csv unavailable (expected; use the API snapshot)"

echo "[dbs] done. Sizes:"
du -sh "$SP_DIR" "$IP_DIR" "$MCSA_DIR" 2>/dev/null || true
