#!/usr/bin/env bash
# Push sweep results to the current branch on GitHub, force-added past .gitignore.
#
#   JOBS="sfl_accel:1 ..." bash scripts/push_results.sh
#
# Every PUSH_EVERY seconds (default 1800): a finished run (DONE) is committed whole -
# runs/<run>/<seed> and checkpoints/<run>/<seed> - and a run still in progress
# contributes its metrics.csv and train.log, so a server that dies mid-sweep still
# leaves the curves behind. wandb/ and run.pid never go in. Exits once every job
# is done and pushed.
#
# Results only travel by git: nothing is copied off the server any other way.
# Push credentials are the server's own; this script never sees a token.
#
# Sourced by launch_sweep.sh for `commit_and_push`; running it directly starts the loop.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ -z "${PY-}" ]; then
  if [ -d /workspace ]; then PY=/workspace/venvs/jaxued/bin/python; else PY="$REPO_ROOT/venvs/jaxued/bin/python"; fi
fi
OUT_DIR="${OUT_DIR:-$REPO_ROOT}"
PUSH_EVERY="${PUSH_EVERY:-1800}"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

# commit_and_push MESSAGE PATH... - stage PATHs with -f, commit if anything changed, push.
commit_and_push() {
  local message="$1"
  shift
  local existing=()
  local path
  for path in "$@"; do
    [ -e "$path" ] && existing+=("$path")
  done
  [ "${#existing[@]}" -eq 0 ] && return 0

  git add -f -- "${existing[@]}" ':(exclude,glob)**/wandb/**' ':(exclude,glob)**/run.pid'
  if git diff --cached --quiet; then
    return 0
  fi
  git commit -q -m "$message

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>" || return 1

  local attempt
  for attempt in 1 2 3 4 5; do
    # --autostash: running trainers keep rewriting metrics.csv files that an
    # earlier progress commit already tracks.
    if git pull --rebase --autostash -q origin "$BRANCH" && git push -q origin "HEAD:$BRANCH"; then
      log "pushed: ${message%%$'\n'*}"
      return 0
    fi
    log "push attempt $attempt failed, retrying in $((attempt * 30))s"
    sleep $((attempt * 30))
  done
  return 1
}

# Prints "run_name<TAB>seed<TAB>state" per job, straight from the sweep runner's
# own view of the filesystem.
job_states() {
  JOBS="$JOBS" OUT_DIR="$OUT_DIR" MPLBACKEND=Agg "$PY" - <<'PYSTATES'
import os
from tlab_ued.sweep import Job, job_status

for spec in os.environ["JOBS"].split():
    preset, seed = spec.rsplit(":", 1)
    status = job_status(Job(preset=preset, seed=int(seed)).config(out_dir=os.environ["OUT_DIR"]))
    print(f"{status['run_name']}\t{status['seed']}\t{status['state']}")
PYSTATES
}

# One pass over the jobs. Returns 0 when every job is done and everything pushed.
push_pass() {
  local states
  states="$(job_states)" || { log "could not read job states"; return 1; }

  local ok=0 all_done=1 progress=()
  local run seed state
  while IFS=$'\t' read -r run seed state; do
    [ -z "$run" ] && continue
    if [ "$state" = "done" ]; then
      commit_and_push "Add $run seed $seed results" "runs/$run/$seed" "checkpoints/$run/$seed" || ok=1
    else
      all_done=0
      progress+=("runs/$run/$seed/metrics.csv" "runs/$run/$seed/train.log")
    fi
  done <<< "$states"

  if [ "${#progress[@]}" -gt 0 ]; then
    commit_and_push "Progress: sweep metrics at $(date -u +%Y-%m-%dT%H:%MZ)" "${progress[@]}" || ok=1
  fi
  [ "$all_done" = 1 ] && [ "$ok" = 0 ]
}

main() {
  if [ -z "${JOBS-}" ]; then
    echo "JOBS is required (e.g. JOBS=\"sfl_accel:1 sfl_accel:2\")" >&2
    exit 2
  fi
  if [ "$(cd "$OUT_DIR" && pwd)" != "$REPO_ROOT" ]; then
    echo "OUT_DIR must be the repo root for results to be committable (got $OUT_DIR)" >&2
    exit 2
  fi
  log "pushing results of: $JOBS -> origin/$BRANCH every ${PUSH_EVERY}s"
  while true; do
    if push_pass; then
      log "all results pushed"
      exit 0
    fi
    sleep "$PUSH_EVERY"
  done
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  main
fi
