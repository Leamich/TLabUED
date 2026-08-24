"""Resumable sweep runner.

A full run is hours long, so training is launched as a *detached* subprocess
(`start_new_session=True`): it survives the notebook kernel being restarted and
the browser tab being closed, which on a rented pod is the difference between
losing an afternoon and not.

State lives in the filesystem, not in memory, so the notebook can be re-run at
any time and simply picks the sweep back up:
    runs/<run_name>/<seed>/run.pid      pid of a live trainer
    runs/<run_name>/<seed>/train.log    stdout/stderr
    runs/<run_name>/<seed>/metrics.csv  one row per eval step -> progress
    runs/<run_name>/<seed>/DONE         written when the trainer exits cleanly

    python -m tlab_ued.sweep --presets dr plr accel --seeds 0 1 2 --wait
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import signal
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

from tlab_ued.config import BASELINE_PRESETS, make_config, to_argv
from tlab_ued.logging_utils import run_dir


@dataclasses.dataclass
class Job:
    preset: str
    seed: int
    overrides: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def config(self, **common: Any) -> Dict[str, Any]:
        return make_config(preset=self.preset, seed=self.seed, **{**common, **self.overrides})


def child_env(**overrides: str) -> Dict[str, str]:
    """Environment for a subprocess running the *venv* interpreter.

    The parent is often a Jupyter kernel from a different Python installation,
    and a few of its variables are actively hostile to the child:

    - `MPLBACKEND` is set by the kernel to `module://matplotlib_inline...`, a
      package the venv does not have. gymnax imports `matplotlib.pyplot` at
      import time, so inheriting it turns every `import jaxued.environments`
      into a `ValueError: Key backend`. Headless children want Agg.
    - `PYTHONPATH` / `PYTHONHOME` would splice the kernel's site-packages
      (numpy 2, a different jax) into the venv. Nothing here needs them:
      `tlab_ued` and `jaxued` are installed into the venv.
    - `LD_LIBRARY_PATH` is set to `/usr/local/cuda/lib64` by the image's profile
      on RunPod (interactive shells and tmux get it, plain `ssh cmd` does not).
      That directory holds the *system* CUDA libraries, which shadow the pip
      CUDA wheels jax was installed with; a cuBLAS 12.8 loaded into a jaxlib
      built for 12.4 fails with `INTERNAL: the library was not initialized` on
      the first matmul. Our jax gets its CUDA from pip wheels only.
    """
    dropped = ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH")
    env = {k: v for k, v in os.environ.items() if k not in dropped}
    env["MPLBACKEND"] = "Agg"
    env.setdefault("WANDB_MODE", "offline")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.update(overrides)
    return env


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def _read_pid(directory: str) -> Optional[int]:
    path = os.path.join(directory, "run.pid")
    if not os.path.exists(path):
        return None
    try:
        pid = int(open(path).read().strip())
    except (ValueError, OSError):
        return None
    return pid if _pid_alive(pid) else None


def _completed_eval_steps(directory: str) -> int:
    path = os.path.join(directory, "metrics.csv")
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        return max(sum(1 for _ in f) - 1, 0)  # minus the header


def job_status(config: Dict[str, Any]) -> Dict[str, Any]:
    """Where one job stands, read entirely from disk."""
    directory = run_dir(config)
    total = config["num_updates"] // config["eval_freq"]
    done_marker = os.path.exists(os.path.join(directory, "DONE"))
    steps = _completed_eval_steps(directory)
    pid = _read_pid(directory)
    if done_marker or steps >= total:
        state = "done"
    elif pid is not None:
        state = "running"
    elif steps > 0:
        state = "interrupted"
    else:
        state = "pending"
    return {
        "run_name": config["run_name"],
        "seed": config["seed"],
        "state": state,
        "eval_steps": steps,
        "total_eval_steps": total,
        "progress": steps / total if total else 0.0,
        "pid": pid,
        "dir": directory,
        "log": os.path.join(directory, "train.log"),
    }


def launch(
    config: Dict[str, Any],
    python: str = sys.executable,
    mem_fraction: Optional[float] = None,
    resume: bool = True,
) -> Dict[str, Any]:
    """Start one training run detached, and return its status."""
    directory = run_dir(config)
    os.makedirs(directory, exist_ok=True)

    status = job_status(config)
    if status["state"] in ("done", "running"):
        return status

    if resume and status["state"] == "interrupted":
        config = {**config, "resume": True}

    env = child_env()
    if mem_fraction is not None:
        # Several trainers share one A100: cap each one's XLA arena instead of
        # letting the first process preallocate 75% of the card.
        env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(mem_fraction)
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    cmd = [python, "-m", "tlab_ued.train", *to_argv(config)]
    log_path = os.path.join(directory, "train.log")
    log_file = open(log_path, "a" if config.get("resume") else "w")
    log_file.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} launching: {' '.join(cmd)}\n")
    log_file.flush()

    process = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        start_new_session=True,  # outlive the notebook kernel
    )
    with open(os.path.join(directory, "run.pid"), "w") as f:
        f.write(str(process.pid))
    # A trainer that exits cleanly leaves a DONE marker; a stale one must go.
    done_path = os.path.join(directory, "DONE")
    if os.path.exists(done_path):
        os.remove(done_path)
    return job_status(config)


def sweep_step(
    jobs: Sequence[Job],
    max_parallel: int = 1,
    python: str = sys.executable,
    mem_fraction: Optional[float] = None,
    **common: Any,
) -> List[Dict[str, Any]]:
    """Start as many pending jobs as the parallelism budget allows; return statuses.

    Call it repeatedly (from a notebook cell, or from the `--wait` loop below);
    it is idempotent and safe to run while jobs are in flight.
    """
    configs = [job.config(**common) for job in jobs]
    statuses = [job_status(c) for c in configs]
    running = sum(1 for s in statuses if s["state"] == "running")

    for i, (config, status) in enumerate(zip(configs, statuses)):
        if running >= max_parallel:
            break
        if status["state"] in ("pending", "interrupted"):
            statuses[i] = launch(config, python=python, mem_fraction=mem_fraction)
            running += 1
    return statuses


def format_status(statuses: Sequence[Dict[str, Any]]) -> str:
    lines = [f"{'run':<24}{'seed':<6}{'state':<13}{'progress':<12}{'pid'}"]
    for s in statuses:
        progress = f"{s['eval_steps']}/{s['total_eval_steps']}"
        lines.append(
            f"{s['run_name']:<24}{s['seed']:<6}{s['state']:<13}{progress:<12}{s['pid'] or '-'}"
        )
    return "\n".join(lines)


def run_sweep(
    jobs: Sequence[Job],
    max_parallel: int = 1,
    poll_seconds: int = 60,
    python: str = sys.executable,
    mem_fraction: Optional[float] = None,
    **common: Any,
) -> List[Dict[str, Any]]:
    """Blocking variant: keep the queue full until every job is done."""
    while True:
        statuses = sweep_step(
            jobs, max_parallel=max_parallel, python=python, mem_fraction=mem_fraction, **common
        )
        print(format_status(statuses), flush=True)
        if all(s["state"] == "done" for s in statuses):
            return statuses
        if not any(s["state"] in ("running", "pending", "interrupted") for s in statuses):
            return statuses
        time.sleep(poll_seconds)


def stop(config: Dict[str, Any]) -> bool:
    """Stop a running job (its checkpoints stay; relaunch resumes from them)."""
    pid = _read_pid(run_dir(config))
    if pid is None:
        return False
    os.killpg(os.getpgid(pid), signal.SIGTERM)
    return True


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run a queue of training jobs")
    parser.add_argument("--presets", nargs="+", default=list(BASELINE_PRESETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument(
        "--jobs",
        nargs="+",
        default=None,
        metavar="PRESET:SEED",
        help=(
            "explicit preset/seed pairs, instead of the presets x seeds product "
            "(e.g. --jobs accel:1 sfl_accel:0)"
        ),
    )
    parser.add_argument("--out_dir", type=str, default=".")
    parser.add_argument("--max_parallel", type=int, default=1)
    parser.add_argument("--mem_fraction", type=float, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--wait", action="store_true", help="block until the sweep finishes")
    parser.add_argument("--status", action="store_true", help="print status and exit")
    args = parser.parse_args(argv)

    if args.jobs:
        jobs = [Job(preset=spec.rsplit(":", 1)[0], seed=int(spec.rsplit(":", 1)[1]))
                for spec in args.jobs]
    else:
        jobs = [Job(preset=p, seed=s) for s in args.seeds for p in args.presets]
    common = {"out_dir": args.out_dir, "smoke": args.smoke}

    if args.status:
        print(format_status([job_status(job.config(**common)) for job in jobs]))
        return
    if args.wait:
        run_sweep(
            jobs, max_parallel=args.max_parallel, mem_fraction=args.mem_fraction, **common
        )
    else:
        print(
            format_status(
                sweep_step(
                    jobs,
                    max_parallel=args.max_parallel,
                    mem_fraction=args.mem_fraction,
                    **common,
                )
            )
        )


if __name__ == "__main__":
    main()
