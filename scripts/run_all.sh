#!/usr/bin/env bash
# The whole experiment, start to finish, on one freshly created pod.
#
#   git clone https://github.com/Leamich/TLabUED && cd TLabUED \
#     && GH_TOKEN=ghp_... RUNPOD_API_KEY=rpa_... bash scripts/run_all.sh
#
# Nothing is ever copied to or from the pod: code arrives by `git clone`, results
# leave by `git push`. The pod is disposable and holds no state worth rescuing,
# which is exactly why every stage commits before the next one starts.
#
# Designed for the ssh session dying halfway through. The script re-executes
# itself detached (setsid + nohup) and records each finished stage under
# .run_all/, so re-running the same command after a dropped connection resumes
# instead of restarting. Watch it with:
#
#   tail -f run_all.log
#
# Environment:
#   GH_TOKEN         required. A GitHub token that can push to the remote.
#   RUNPOD_API_KEY   required only when TEARDOWN=1.
#   TEARDOWN         1 (default) terminates the pod after a verified push.
#   DRY_RUN          1 prints the commands instead of running them. No GPU
#                    needed - this is how the stage logic is checked before a
#                    pod exists.
#   ONLY             run a single stage by name and stop (for debugging).
#   STOP_AFTER       run up to this stage, then tear down. `STOP_AFTER=pick`
#                    does the offline half only (~2h) and is the right first pod:
#                    the bench decides which arm the sweep should run, so paying
#                    for the sweep before seeing it buys a guess.
#   BENCH_LEVELS     levels per checkpoint in the bench (default 8192).
#   SEEDS            seeds for the confirming sweep (default "0 1 2").
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

STATE_DIR="$REPO_ROOT/.run_all"
LOG="$REPO_ROOT/run_all.log"
mkdir -p "$STATE_DIR"

TEARDOWN="${TEARDOWN:-1}"
DRY_RUN="${DRY_RUN:-0}"
ONLY="${ONLY:-}"
STOP_AFTER="${STOP_AFTER:-}"
STOPPED=0
BENCH_LEVELS="${BENCH_LEVELS:-8192}"
BENCH_ATTEMPTS="${BENCH_ATTEMPTS:-8}"
SEEDS="${SEEDS:-0 1 2}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT}"

# The runs whose checkpoints the bench reads. These are already in the
# repository, so stage `fetch` is a git operation and not a download.
BENCH_RUNS="${BENCH_RUNS:-sfl_oracle_learnability_level_bfs:0 sfl_oracle_learnability_level_bfs:1 sfl_oracle_learnability_level_bfs:2 accel_maxmc:0}"
STALENESS_RUN="${STALENESS_RUN:-sfl_oracle_learnability_level_bfs}"

PY="${PY:-}"  # resolved by the bootstrap stage

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

# A headless pod has no other way to say what went wrong: no ssh key, no volume,
# and the disk dies with it. So the log itself is an artefact - pushed on the way
# out, success or failure, with every secret removed first.
#
# The scrub is not belt-and-braces: the repository is public, and a token that
# reaches it is a token that has to be revoked.
publish_log() {
  local status="$1"
  [ "$DRY_RUN" = "1" ] && return 0
  [ -f "$LOG" ] || return 0
  mkdir -p results/logs
  # Defaulted, not bare: `set -u` makes a bare ${GH_TOKEN} abort this function
  # when the token is missing - and "the token is missing" is precisely the
  # failure this function exists to report. The first pod died exactly here and
  # billed an hour of A100 without saying a word.
  sed -e "s#${GH_TOKEN:-__no_token__}#***GH_TOKEN***#g" \
      -e "s#${RUNPOD_API_KEY:-__none__}#***RUNPOD_API_KEY***#g" \
      -e 's#https://[^@/]*@github#https://***@github#g' \
      "$LOG" > "results/logs/run_all_${status}.log"
  push_results "Add the pod's own run log ($status)" results/logs/ || true
}

die() {
  log "FATAL: $*"
  publish_log failed
  # Still let the watchdog own the pod's lifetime: a failed run that leaves the
  # GPU billing is worse than a failed run.
  exit 1
}

run() {
  if [ "$DRY_RUN" = "1" ]; then
    echo "    would run: $*"
    return 0
  fi
  "$@"
}

# A stage runs once. Its marker is written only if it succeeded, so a failed
# stage is retried on the next invocation rather than silently skipped.
stage() {
  local name="$1"; shift
  if [ -n "$ONLY" ] && [ "$ONLY" != "$name" ]; then return 0; fi
  # Past STOP_AFTER everything is skipped except the teardown, so the pod still
  # shuts itself down rather than idling at the price of a GPU per hour.
  if [ "$STOPPED" = "1" ] && [ "$name" != "teardown" ]; then return 0; fi
  if [ -f "$STATE_DIR/$name.done" ]; then
    log "stage $name: already done"
    [ "$name" = "$STOP_AFTER" ] && STOPPED=1
    return 0
  fi
  log "stage $name: start"
  if "$@"; then
    touch "$STATE_DIR/$name.done"
    log "stage $name: done"
    if [ "$name" = "$STOP_AFTER" ]; then
      log "STOP_AFTER=$name reached"
      STOPPED=1
    fi
  else
    die "stage $name failed (exit $?); fix it and re-run the same command to resume"
  fi
}

# --- 0. preflight ----------------------------------------------------------
# Everything that can be known to be wrong before four hours of GPU time is
# checked here. A missing token discovered at the push stage costs the whole run.
preflight() {
  [ -n "${GH_TOKEN:-}" ] || die "GH_TOKEN is not set: results would have nowhere to go"
  if [ "$TEARDOWN" = "1" ] && [ -z "${RUNPOD_API_KEY:-}" ]; then
    die "TEARDOWN=1 but RUNPOD_API_KEY is not set (use TEARDOWN=0 to keep the pod)"
  fi

  git rev-parse --abbrev-ref HEAD >/dev/null || die "not a git repository"

  # A dry run must not touch the repository it is being rehearsed in - it is
  # normally a developer's own clone, and rewriting its remote to carry a token
  # would be both surprising and a way to leak one.
  if [ "$DRY_RUN" = "1" ]; then
    echo "    would set the git identity and a token credential helper"
    return 0
  fi

  git config user.name  "${GIT_NAME:-Mikhail Leontyev}"
  git config user.email "${GIT_EMAIL:-michlea.tlt@gmail.com}"

  # Authenticate with a credential helper rather than by putting the token in
  # the remote URL. The difference matters because this repository is public and
  # this script pushes its own log: git prints the remote URL verbatim in error
  # messages ("unable to access 'https://TOKEN@github.com/...'"), so a token in
  # the URL is one failed push away from being published. The helper keeps the
  # URL clean and reads the token from the environment at each use.
  git config --local credential.helper \
    '!f() { echo username=x-access-token; echo "password=$GH_TOKEN"; }; f'
  git config --local --unset-all remote.origin.url 2>/dev/null
  git config --local remote.origin.url "https://github.com/Leamich/TLabUED"

  git ls-remote --exit-code origin >/dev/null 2>&1 \
    || die "cannot reach origin with this GH_TOKEN"
  log "preflight ok (teardown=$TEARDOWN, dry_run=$DRY_RUN)"
}

# A pod that finishes cleanly kills itself in `teardown`. A pod that does not -
# a stage that hangs, a teardown that correctly refuses because results are
# unpushed - would otherwise bill by the hour forever. This is the backstop: an
# independent process that terminates the pod after MAX_HOURS no matter what the
# run is doing. It outlives this script on purpose.
#
# The key is read from the environment inside the child (single quotes), never
# baked into its argv where `ps` would show it.
watchdog() {
  local hours="${MAX_HOURS:-4}"
  if [ "$TEARDOWN" != "1" ] || [ -z "${RUNPOD_POD_ID:-}" ]; then
    log "  no watchdog (teardown=$TEARDOWN, pod=${RUNPOD_POD_ID:-none})"
    return 0
  fi
  if [ "$DRY_RUN" = "1" ]; then
    echo "    would arm a ${hours}h watchdog on pod $RUNPOD_POD_ID"
    return 0
  fi
  export RUNPOD_API_KEY RUNPOD_POD_ID
  setsid nohup bash -c '
    sleep '"$((hours * 3600))"'
    curl -s -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" \
      "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID" >/dev/null
  ' >/dev/null 2>&1 &
  log "  watchdog armed: pod dies after ${hours}h whatever happens"
}

# --- 1. bootstrap ----------------------------------------------------------
bootstrap() {
  run bash scripts/bootstrap.sh "${WORKSPACE:-$REPO_ROOT}" || return 1
  resolve_py
}

resolve_py() {
  if [ -n "$PY" ]; then return 0; fi
  for candidate in "/workspace/venvs/jaxued/bin/python" "$REPO_ROOT/venvs/jaxued/bin/python"; do
    if [ -x "$candidate" ]; then PY="$candidate"; break; fi
  done
  [ "$DRY_RUN" = "1" ] && PY="${PY:-python}"
  [ -n "$PY" ] || die "no venv python found; did bootstrap run?"
  export MPLBACKEND=Agg
  unset LD_LIBRARY_PATH
  log "python: $PY"
}

# --- 2. checks -------------------------------------------------------------
# Parity is not a formality. Every stage below trains a student, and if the
# baselines no longer reproduce upstream bit for bit then nothing measured after
# this point can be compared with the numbers in the README.
checks() {
  resolve_py
  run "$PY" -m pytest tests -q || return 1
  run "$PY" -m tlab_ued.parity --presets dr plr accel --num_updates 500 || return 1
}

# --- 3. collect ------------------------------------------------------------
# The only stage that costs env steps. ~16M per checkpoint, eight checkpoints per
# run, on both the training generator and the held-out-shaped validation one.
collect() {
  resolve_py
  for spec in $BENCH_RUNS; do
    local run_name="${spec%%:*}" seed="${spec##*:}"
    if [ ! -d "checkpoints/$run_name/$seed/models" ]; then
      log "  no checkpoints for $run_name/$seed, skipping"
      continue
    fi
    # Both generators on every seed. A 384-level local probe found the trained
    # policy at p = 0.990 (learnability 0.0011) on `minigrid_walls` but p = 0.810
    # (learnability 0.0296) on `perfect_maze` - so the validation distribution
    # carries ~27x the late headroom, and it is where the feature ladder has
    # anything to separate. That makes it worth replicating across seeds rather
    # than sampling once, and the same probe measured ~90k steps/s on a laptop
    # CPU, so the extra collection is minutes on an A100.
    for generator in minigrid_walls perfect_maze; do
      log "  collect $run_name seed $seed on $generator"
      run "$PY" -m tlab_ued.oracle_bench collect \
        --run "$run_name" --seed "$seed" --generator "$generator" \
        --num_levels "$BENCH_LEVELS" --attempts "$BENCH_ATTEMPTS" \
        --out_dir "$OUT_DIR" || return 1
    done
  done
}

# --- 4. bench + staleness --------------------------------------------------
bench() {
  resolve_py
  run "$PY" -m tlab_ued.oracle_bench bench --out results/oracle_bench.json || return 1
  for seed in 0 1 2; do
    [ -d "checkpoints/$STALENESS_RUN/$seed/models" ] || continue
    run "$PY" -m tlab_ued.oracle_bench staleness \
      --run "$STALENESS_RUN" --seed "$seed" --out_dir "$OUT_DIR" \
      --out "results/oracle_staleness_$seed.json" || return 1
  done
}

# --- git -------------------------------------------------------------------
# Push with retries and a rebase: another session (or an earlier stage of this
# one) may have moved master, and losing four hours of results to a
# non-fast-forward would be absurd.
push_results() {
  local message="$1"; shift
  if [ "$DRY_RUN" = "1" ]; then
    echo "    would commit and push: $message ($*)"
    return 0
  fi
  git add -f "$@" 2>/dev/null || true
  if git diff --cached --quiet; then
    log "  nothing new to commit"
    return 0
  fi
  git commit -q -m "$message

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>" || return 1

  local attempt
  for attempt in 1 2 3 4 5; do
    git pull --rebase --quiet origin master && git push --quiet origin HEAD:master && {
      log "  pushed: $message"
      return 0
    }
    log "  push attempt $attempt failed, retrying in $((attempt * 20))s"
    sleep $((attempt * 20))
  done
  return 1
}

push_bench() { push_results "Add the offline oracle bench: population ceiling, feature ladder, staleness" results/; }

# --- 5. pick ---------------------------------------------------------------
# The arm is chosen by the rule in `oracle_bench.pick_arm`, which was written
# before the data existed. No human in the loop is the point: the script has to
# run unattended, and a rule fixed in advance is a stronger claim than a choice
# made after seeing the answer.
pick() {
  resolve_py
  local arm
  if [ "$DRY_RUN" = "1" ]; then
    arm="prop"
  else
    arm="$("$PY" -m tlab_ued.oracle_bench pick --report results/oracle_bench.json 2>/dev/null | tail -1)"
  fi
  case "$arm" in
    level|wide|prop) ;;
    # A failed or degenerate bench must not idle the GPU: fall back to the arm
    # with the strongest prior reason to work (README §7.5).
    *) log "  bench gave no usable arm ('$arm'), falling back to prop"; arm="prop" ;;
  esac
  echo "$arm" > "$STATE_DIR/arm"
  log "  arm: $arm"
}

# --- 6. sweep --------------------------------------------------------------
# Eight jobs in the nine slots that fit on one A100 under MPS: the chosen arm and
# the runner-up on three seeds each, plus the two `sfl_accel_cheap` seeds that
# README §8.4 names as the report's weakest support. The ninth slot is left free
# on purpose - nine concurrent trainers was measured at 2.5h, and a spare slot is
# what lets a crashed run be restarted without waiting for the batch.
sweep() {
  resolve_py
  local arm second
  arm="$(cat "$STATE_DIR/arm" 2>/dev/null || echo prop)"
  case "$arm" in
    prop) second="wide" ;;
    wide) second="prop" ;;
    *)    second="wide" ;;
  esac

  local jobs=""
  for seed in $SEEDS; do jobs="$jobs sfl_oracle_$arm:$seed"; done
  for seed in $SEEDS; do jobs="$jobs sfl_oracle_$second:$seed"; done
  jobs="$jobs sfl_accel_cheap:1 sfl_accel_cheap:2"

  log "  jobs:$jobs"
  run "$PY" -m tlab_ued.sweep --jobs $jobs \
    --out_dir "$OUT_DIR" --max_parallel 9 --mem_fraction 0.1 --wait || return 1
}

# --- 7. report -------------------------------------------------------------
report() {
  resolve_py
  run "$PY" scripts/report.py --out_dir "$OUT_DIR" || return 1
  push_results "Add the confirming sweep: honest oracle arms on three seeds" \
    results/ runs/ checkpoints/
}

# --- 8. teardown -----------------------------------------------------------
# Only after a push that returned success. There is no network volume on this
# account, so terminating destroys the disk: if the results are not in git they
# are gone, and an idle pod is much cheaper than a repeated experiment.
teardown() {
  if [ "$TEARDOWN" != "1" ]; then
    log "  TEARDOWN=0, leaving the pod running"
    return 0
  fi
  local pod="${RUNPOD_POD_ID:-}"
  [ -n "$pod" ] || { log "  RUNPOD_POD_ID is unset (not on a pod?), skipping"; return 0; }

  if [ "$(git rev-parse HEAD)" != "$(git rev-parse origin/master 2>/dev/null || echo none)" ]; then
    log "  local HEAD is not on origin/master - refusing to terminate with unpushed work"
    return 0
  fi

  log "  terminating pod $pod"
  run curl -s -X DELETE \
    -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
    "https://rest.runpod.io/v1/pods/$pod" >/dev/null
}

# --- detach ----------------------------------------------------------------
# Re-exec detached the first time round, so closing the terminal (or losing the
# connection) does not take the run with it.
if [ "${RUN_ALL_DETACHED:-0}" != "1" ] && [ "$DRY_RUN" != "1" ]; then
  export RUN_ALL_DETACHED=1
  log "detaching; follow with: tail -f $LOG"
  setsid nohup bash "$0" "$@" >>"$LOG" 2>&1 < /dev/null &
  echo "pid $!"
  exit 0
fi

log "=== run_all starting (repo $(git rev-parse --short HEAD)) ==="
stage preflight preflight
watchdog
stage bootstrap bootstrap
stage checks    checks
stage collect   collect
stage bench     bench
stage push_bench push_bench
stage pick      pick
stage sweep     sweep
stage report    report
log "=== run_all finished ==="
publish_log ok
stage teardown  teardown
