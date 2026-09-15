#!/usr/bin/env bash
# Bring up a GPU server and run a sweep on it, in stages.
#
#   STAGE=setup     bash scripts/launch_sweep.sh   # bootstrap, CPU pytest, parity
#   STAGE=calibrate bash scripts/launch_sweep.sh   # measure memory and throughput, recommend parallelism
#   STAGE=sweep     bash scripts/launch_sweep.sh   # launch the sweep + result pusher, detached
#   STAGE=status    bash scripts/launch_sweep.sh   # progress, ETA, GPU state
#
# Environment:
#   JOBS          preset:seed list (default: the 12 runs of this experiment)
#   MAX_PARALLEL  trainers at once       } default for `sweep`: the recommendation
#   MEM_FRACTION  XLA memory cap each    } written by `calibrate`
#   OUT_DIR       defaults to the repo root, so runs/ and checkpoints/ land where
#                 push_results.sh can commit them
#
# Why MPS: without it, concurrent trainers time-slice the GPU. On an A100, nine of
# them gave 81k env-steps/s in total without MPS and 243k with it. The tell that
# it is not working is nvidia-smi showing ~100% GPU utilisation at ~1% memory
# utilisation.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

STAGE="${STAGE:-}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT}"
# Main comparisons first, ablations after: if the sweep has to be cut short, the
# seeds the README is missing are the ones that finish.
JOBS="${JOBS:-sfl_oracle_level:1 sfl_accel:1 sfl_accel_cheap:1 sfl_oracle_level:2 sfl_accel:2 sfl_accel_cheap:2 \
sfl_oracle_level_nomut:0 sfl_oracle_level_noverify:0 sfl_oracle_level_nomut:1 sfl_oracle_level_noverify:1 \
sfl_oracle_level_nomut:2 sfl_oracle_level_noverify:2}"
CALIBRATION_PRESETS="sfl_accel sfl_accel_cheap sfl_oracle_level sfl_oracle_level_nomut sfl_oracle_level_noverify"

if [ -d /workspace ]; then WORKSPACE=/workspace; else WORKSPACE="$REPO_ROOT"; fi
PY="${PY:-$WORKSPACE/venvs/jaxued/bin/python}"
export PY OUT_DIR JOBS

# See bootstrap.sh: a Jupyter MPLBACKEND breaks gymnax, and system CUDA libraries
# on LD_LIBRARY_PATH shadow the pip wheels jax was built against.
export MPLBACKEND=Agg
unset LD_LIBRARY_PATH

export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

# shellcheck source=push_results.sh
source "$REPO_ROOT/scripts/push_results.sh"

gpu_slug() {
  nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -cs 'A-Za-z0-9' '_' | sed 's/_$//'
}

start_mps() {
  if ! command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
    log "WARNING: nvidia-cuda-mps-control not found - trainers will time-slice the GPU (~3x slower in total)"
    return 0
  fi
  mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
  if pgrep -x nvidia-cuda-mps-control >/dev/null 2>&1; then
    log "MPS daemon already running"
    return 0
  fi
  if nvidia-cuda-mps-control -d; then
    log "MPS daemon started (pipe dir $CUDA_MPS_PIPE_DIRECTORY)"
  else
    log "WARNING: could not start MPS - trainers will time-slice the GPU (~3x slower in total)"
  fi
}

sweep_running() {
  pgrep -f "tlab_ued.sweep .*--wait" >/dev/null 2>&1
}

# --- setup -------------------------------------------------------------------
stage_setup() {
  if [ -z "$(git config user.email || true)" ]; then
    echo "git user.email is not set; results could not be committed. Set it first." >&2
    exit 1
  fi
  bash scripts/bootstrap.sh
  mkdir -p results
  # On CPU: a GPU pytest maps tens of GB of device memory for nothing. Both
  # variables: hiding the device alone makes jax 0.4.30 fail in its CUDA plugin
  # instead of falling back, and JAX_PLATFORMS alone still maps device memory.
  CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu "$PY" -m pytest tests -q
  # On the GPU, but before any trainer exists: next to a full card it fails in cuSolver.
  "$PY" -m tlab_ued.parity --presets dr plr accel --num_updates 500 2>&1 | tee results/parity.log
  local identical
  identical="$(grep -c ": IDENTICAL" results/parity.log || true)"
  if [ "$identical" -lt 3 ]; then
    echo "parity is not IDENTICAL for all three baselines - stop here" >&2
    exit 1
  fi
  log "setup ok: tests pass, parity IDENTICAL"
}

# --- calibrate ---------------------------------------------------------------
stage_calibrate() {
  if sweep_running; then
    echo "a sweep is running; calibrating next to it would measure nothing useful" >&2
    exit 1
  fi
  start_mps
  local slug dir n
  slug="$(gpu_slug)"
  dir="results/calibration_$slug"
  mkdir -p "$dir"
  n=$(echo $CALIBRATION_PRESETS | wc -w)

  local cal_jobs="" preset
  for preset in $CALIBRATION_PRESETS; do cal_jobs="$cal_jobs $preset:0"; done

  # Fresh smoke directories, or the sweep runner reports them done and measures nothing.
  "$PY" - $cal_jobs <<'PYCLEAN'
import os, shutil, sys
from tlab_ued.logging_utils import checkpoint_dir, run_dir
from tlab_ued.sweep import Job
for spec in sys.argv[1:]:
    preset, seed = spec.rsplit(":", 1)
    config = Job(preset=preset, seed=int(seed)).config(out_dir=os.environ["OUT_DIR"], smoke=True)
    for path in (run_dir(config), checkpoint_dir(config)):
        shutil.rmtree(path, ignore_errors=True)
PYCLEAN

  nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,utilization.memory \
    --format=csv,noheader,nounits -l 5 > "$dir/gpu.csv" &
  local gpu_sampler=$!
  nvidia-smi --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader,nounits -l 5 > "$dir/apps.csv" &
  local apps_sampler=$!

  # A generous cap: this stage measures what a trainer takes, it must not constrain it.
  local cal_fraction
  cal_fraction="$(awk -v n="$n" 'BEGIN { printf "%.3f", 0.9 / n }')"
  log "calibrating on $slug: $n smoke runs at once, mem_fraction $cal_fraction"
  # A trainer that crashes before its first eval step looks "pending" to the
  # sweep runner and is relaunched forever; the timeout ends that.
  timeout 3600 "$PY" -m tlab_ued.sweep --smoke --jobs $cal_jobs --out_dir "$OUT_DIR" \
    --max_parallel "$n" --mem_fraction "$cal_fraction" --wait > "$dir/sweep.log" 2>&1 || true
  kill "$gpu_sampler" "$apps_sampler" 2>/dev/null || true

  local status=0
  "$PY" - "$dir" $cal_jobs <<'PYREPORT' || status=$?
import csv, math, os, sys
from tlab_ued.config import make_config
from tlab_ued.logging_utils import run_dir
from tlab_ued.sweep import Job

out, specs = sys.argv[1], sys.argv[2:]
lines, failed, per_run_sps = [], [], []
for spec in specs:
    preset, seed = spec.rsplit(":", 1)
    config = Job(preset=preset, seed=int(seed)).config(out_dir=os.environ["OUT_DIR"], smoke=True)
    directory = run_dir(config)
    path = os.path.join(directory, "metrics.csv")
    rows = list(csv.DictReader(open(path))) if os.path.exists(path) else []
    if not os.path.exists(os.path.join(directory, "DONE")) or len(rows) < 2:
        failed.append((spec, os.path.join(directory, "train.log")))
        continue
    # The last eval step: compilation is behind it, and all trainers were still running.
    last, prev = rows[-1], rows[-2]
    sps = (float(last["num_env_steps"]) - float(prev["num_env_steps"])) / float(last["time_delta"])
    per_run_sps.append(sps)
    lines.append(f"  {spec:<32} {sps:>9.0f} env-steps/s")

def column(path, index):
    values = []
    for row in open(path):
        parts = [p.strip() for p in row.split(",")]
        try:
            values.append(float(parts[index]))
        except (IndexError, ValueError):
            pass
    return values

gpu = os.path.join(out, "gpu.csv")
used, total = column(gpu, 0), column(gpu, 1)
util, mem_util = column(gpu, 2), column(gpu, 3)
apps = column(os.path.join(out, "apps.csv"), 2)
n = len(specs)
total_mib = max(total) if total else float("nan")
# Under MPS some drivers list only the MPS server, so also take the card total / n.
per_trainer = max([max(apps) if apps else 0.0, (max(used) / n) if used else 0.0])

target_runs = 12
full_run_steps = make_config(preset="sfl_accel")["num_updates"] * 32 * 256
report = [
    f"gpu: {os.path.basename(out).removeprefix('calibration_')}",
    f"concurrent smoke runs: {n}, finished: {len(per_run_sps)}",
    *lines,
]
if failed:
    report.append("FAILED (see train.log):")
    report += [f"  {spec}: {log}" for spec, log in failed]
if per_run_sps and used:
    aggregate = sum(per_run_sps)
    parallel = max(1, min(target_runs, math.floor(0.85 * total_mib / (per_trainer * 1.15))))
    fraction = min(0.95 / parallel, max(per_trainer * 1.3 / total_mib, 0.85 / parallel))
    work = target_runs * full_run_steps
    # Total throughput at `parallel` trainers is somewhere between what `n` gave and
    # linear scaling from it; both ends are reported rather than a guess in between.
    slow = work / aggregate / 3600
    fast = work / (aggregate * max(parallel, n) / n) / 3600
    report += [
        f"aggregate throughput at {n} concurrent: {aggregate:.0f} env-steps/s",
        f"GPU memory: total {total_mib:.0f} MiB, peak used {max(used):.0f} MiB",
        f"peak per trainer: {per_trainer:.0f} MiB",
        f"GPU util median {sorted(util)[len(util)//2]:.0f}%, memory util median {sorted(mem_util)[len(mem_util)//2]:.0f}%"
        if util and mem_util else "GPU util: no samples",
        f"recommended: MAX_PARALLEL={parallel} MEM_FRACTION={fraction:.3f}",
        f"ETA for {target_runs} full runs: {fast:.1f}-{slow:.1f} h "
        f"(linear scaling from {n} to {parallel} trainers .. no gain beyond {n}); "
        "smoke evals are lighter than full ones, refine from the first eval steps of the sweep",
    ]
    with open(os.path.join(out, "recommended.env"), "w") as f:
        f.write(f"MAX_PARALLEL={parallel}\nMEM_FRACTION={fraction:.3f}\n")
text = "\n".join(report)
print(text)
open(os.path.join(out, "calibration.txt"), "w").write(text + "\n")
sys.exit(1 if failed or not per_run_sps else 0)
PYREPORT

  commit_and_push "Add $slug calibration: memory and throughput of concurrent trainers" "$dir" \
    || { echo "push failed - fix git credentials before launching the sweep" >&2; exit 1; }
  return $status
}

# --- sweep -------------------------------------------------------------------
stage_sweep() {
  if sweep_running; then
    log "a sweep is already running"
    return 0
  fi
  local slug
  slug="$(gpu_slug)"
  if [ -z "${MAX_PARALLEL-}" ] || [ -z "${MEM_FRACTION-}" ]; then
    local recommended="results/calibration_$slug/recommended.env"
    if [ ! -f "$recommended" ]; then
      echo "set MAX_PARALLEL and MEM_FRACTION, or run STAGE=calibrate first" >&2
      exit 1
    fi
    # shellcheck disable=SC1090
    source "$recommended"
  fi
  start_mps
  log "sweep: max_parallel $MAX_PARALLEL, mem_fraction $MEM_FRACTION"
  log "jobs: $JOBS"

  setsid nohup "$PY" -m tlab_ued.sweep --jobs $JOBS --out_dir "$OUT_DIR" \
    --max_parallel "$MAX_PARALLEL" --mem_fraction "$MEM_FRACTION" --wait \
    > sweep.log 2>&1 < /dev/null &
  setsid nohup bash scripts/push_results.sh > push.log 2>&1 < /dev/null &
  sleep 5
  log "launched. Watch with:"
  echo "  STAGE=status bash scripts/launch_sweep.sh"
  echo "  tail -f sweep.log push.log"
}

# --- status ------------------------------------------------------------------
stage_status() {
  "$PY" - <<'PYSTATUS'
import csv, os, time
from tlab_ued.sweep import Job, job_status

rows = []
for spec in os.environ["JOBS"].split():
    preset, seed = spec.rsplit(":", 1)
    config = Job(preset=preset, seed=int(seed)).config(out_dir=os.environ["OUT_DIR"])
    s = job_status(config)
    path = os.path.join(s["dir"], "metrics.csv")
    deltas = [float(r["time_delta"]) for r in csv.DictReader(open(path))] if os.path.exists(path) else []
    recent = deltas[-5:]
    left = (s["total_eval_steps"] - s["eval_steps"]) * (sum(recent) / len(recent)) if recent else None
    rows.append((spec, s["state"], f"{s['eval_steps']}/{s['total_eval_steps']}",
                 f"{left / 3600:.1f} h" if left is not None else "-"))
print(f"{'job':<32}{'state':<13}{'progress':<10}remaining (at current pace)")
for row in rows:
    print(f"{row[0]:<32}{row[1]:<13}{row[2]:<10}{row[3]}")
PYSTATUS
  nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,utilization.memory --format=csv
  pgrep -x nvidia-cuda-mps-control >/dev/null && echo "MPS: running" || echo "MPS: NOT running"
  tail -3 push.log 2>/dev/null || true
}

case "$STAGE" in
  setup) stage_setup ;;
  calibrate) stage_calibrate ;;
  sweep) stage_sweep ;;
  status) stage_status ;;
  *) echo "STAGE must be one of: setup calibrate sweep status" >&2; exit 2 ;;
esac
