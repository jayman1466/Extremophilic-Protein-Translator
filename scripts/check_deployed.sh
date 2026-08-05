#!/usr/bin/env bash
# Verify the cluster's copy of every script/module the assemble chain invokes matches
# this working tree, BEFORE submitting. Two failures this session came from a partial
# deploy that this check would have caught in one command:
#
#   * stage B died in 1 s -- 'unrecognized arguments: --max-per-sample' -- because
#     04_select_genomes.py had not been re-transferred after the flag was added.
#   * stage F ran blind because 06_assemble_dataset.py's phase markers existed only
#     locally, so `python -u` had nothing to stream.
#
# Usage:  bash scripts/check_deployed.sh [remote_repo_root]
# Exits non-zero and lists mismatches if any file differs or is missing.
set -uo pipefail
REMOTE_ROOT="${1:-/groups/cress/projects/jaymin/eptrans_scratch/repo}"
HOST="${EPTRANS_HOST:-biotite}"

FILES=(
  scripts/01b_flag_metadata.py
  scripts/03_combine_bins.py
  scripts/03a_fetch_ogt_sources.py
  scripts/03b_merge_measured_ogt.py
  scripts/03c_merge_deepsea_mags.py
  scripts/04_select_genomes.py
  scripts/05_aggregate_signalp.py
  scripts/06_assemble_dataset.py
  scripts/slurm/14_assemble_chain.sh
  src/eptrans/selection.py
  src/eptrans/dataset.py
  src/eptrans/binning.py
  src/eptrans/gtdb.py
)

printf 'checking %d files against %s:%s\n' "${#FILES[@]}" "$HOST" "$REMOTE_ROOT"
remote_sums=$(ssh "$HOST" "cd '$REMOTE_ROOT' 2>/dev/null && md5sum ${FILES[*]} 2>&1" || true)

bad=0
for f in "${FILES[@]}"; do
  [ -f "$f" ] || { printf '  LOCAL-MISSING  %s\n' "$f"; bad=$((bad+1)); continue; }
  lsum=$(md5sum "$f" | cut -d' ' -f1)
  rsum=$(printf '%s\n' "$remote_sums" | awk -v p="$f" '$2==p {print $1}')
  if [ -z "$rsum" ]; then
    printf '  REMOTE-MISSING %s\n' "$f"; bad=$((bad+1))
  elif [ "$lsum" != "$rsum" ]; then
    printf '  STALE          %s  local=%s remote=%s\n' "$f" "${lsum:0:8}" "${rsum:0:8}"; bad=$((bad+1))
  fi
done

if [ "$bad" -eq 0 ]; then
  echo "ALL IN SYNC -- safe to submit"
else
  printf 'MISMATCHES: %d -- deploy before submitting\n' "$bad"
  exit 1
fi
