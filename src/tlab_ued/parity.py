"""Parity check: our trainer vs the upstream examples, same config, same seed.

The baselines are only worth reporting if our refactor reproduces upstream
exactly. This runs both implementations in-process on a short budget, captures
what each one passes to `wandb.log`, and diffs the scalars.

Because both sides share the frozen student and keep upstream's `jax.random.split`
pattern, the expected result is *bit-identical* metrics, not merely similar
curves. A non-zero diff means a seam was cut wrong - investigate before spending
GPU hours.

    from tlab_ued.parity import check_parity
    report = check_parity("accel", seed=0, num_updates=500)
    report["max_abs_diff"]   # -> {"solve_rate/mean": 0.0, ...}
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np

from tlab_ued.config import make_config, upstream_config

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
UPSTREAM_EXAMPLES = os.path.join(REPO_ROOT, "third_party", "jaxued", "examples")

# Which upstream script each of our teachers is a port of.
UPSTREAM_SCRIPT = {"dr": "maze_dr.py", "plr": "maze_plr.py", "accel": "maze_plr.py"}


def _scalars(log: Dict[str, Any]) -> Dict[str, float]:
    """Numeric entries only - wandb Images/Videos are dropped."""
    out = {}
    for key, value in log.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float, np.integer, np.floating)):
            out[key] = float(value)
        elif hasattr(value, "shape") and getattr(value, "ndim", 1) == 0:
            out[key] = float(value)
    return out


@contextlib.contextmanager
def capture_wandb_logs(sink: List[Dict[str, float]]):
    """Intercept `wandb.log` while leaving `wandb.init` intact.

    `wandb.init` has to run for real: upstream reads `wandb.config` and calls
    `.as_dict()` on it. Offline mode keeps that working without an API key.
    """
    import wandb

    os.environ["WANDB_MODE"] = "offline"
    original_log, original_image, original_video = wandb.log, wandb.Image, wandb.Video

    def spy(data, *args, **kwargs):
        sink.append(_scalars(data))

    def no_media(*args, **kwargs):
        return None

    wandb.log = spy
    # Only scalars are compared, and encoding upstream's eval animations would
    # otherwise drag moviepy into the check.
    wandb.Image = no_media
    wandb.Video = no_media
    try:
        yield sink
    finally:
        wandb.log, wandb.Image, wandb.Video = original_log, original_image, original_video


def load_upstream(teacher: str):
    """Import the upstream example module straight from third_party/."""
    script = UPSTREAM_SCRIPT[teacher]
    path = os.path.join(UPSTREAM_EXAMPLES, script)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found - clone jaxued into third_party/ first (see notebooks/baseline.ipynb)"
        )
    module_name = f"upstream_{os.path.splitext(script)[0]}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def run_upstream(config: Dict[str, Any]) -> List[Dict[str, float]]:
    module = load_upstream(config["teacher"])
    logs: List[Dict[str, float]] = []
    with capture_wandb_logs(logs):
        module.main(upstream_config(config), project=config["project"])
    return logs


def run_ours(config: Dict[str, Any]) -> List[Dict[str, float]]:
    from tlab_ued.train import main

    logs: List[Dict[str, float]] = []
    with capture_wandb_logs(logs):
        main({**config, "log_media": "none"})
    return logs


def compare(ours: List[Dict[str, float]], theirs: List[Dict[str, float]]) -> Dict[str, Any]:
    """Max absolute difference per metric, over the eval steps both produced."""
    n = min(len(ours), len(theirs))
    keys = sorted(set(ours[0]) & set(theirs[0])) if n else []
    # `sps` and `time_delta` are wall-clock, not results.
    keys = [k for k in keys if k not in ("sps", "time_delta")]
    diffs = {
        key: max(abs(ours[i].get(key, np.nan) - theirs[i].get(key, np.nan)) for i in range(n))
        for key in keys
    }
    only_ours = sorted(set(ours[0]) - set(theirs[0])) if n else []
    only_theirs = sorted(set(theirs[0]) - set(ours[0])) if n else []
    return {
        "num_eval_steps": n,
        "max_abs_diff": diffs,
        "worst": max(diffs.items(), key=lambda kv: kv[1]) if diffs else None,
        "identical": all(d == 0.0 for d in diffs.values()) if diffs else False,
        "metrics_only_in_ours": only_ours,
        "metrics_only_in_upstream": only_theirs,
    }


def check_parity(
    preset: str = "accel",
    seed: int = 0,
    num_updates: int = 500,
    out_dir: Optional[str] = None,
    **overrides: Any,
) -> Dict[str, Any]:
    """Run both implementations and diff them. Returns the comparison report."""
    config = make_config(
        preset=preset,
        seed=seed,
        num_updates=num_updates,
        eval_freq=min(250, num_updates),
        eval_num_attempts=2,
        checkpoint_save_interval=0,  # nothing to save for a parity run
        out_dir=out_dir or os.path.join(REPO_ROOT, "runs", "_parity"),
        run_name=f"parity_{preset}",
        allow_student_changes=True,  # the short budget is the point
        **overrides,
    )
    print(f"--- upstream {UPSTREAM_SCRIPT[config['teacher']]} ({preset}, seed {seed}) ---", flush=True)
    theirs = run_upstream(config)
    print(f"--- tlab_ued.train ({preset}, seed {seed}) ---", flush=True)
    ours = run_ours(config)

    report = compare(ours, theirs)
    report["preset"] = preset
    report["seed"] = seed
    report["num_updates"] = num_updates
    verdict = "IDENTICAL" if report["identical"] else f"DIFFERS (worst: {report['worst']})"
    print(f"parity[{preset}, seed {seed}] over {report['num_eval_steps']} eval steps: {verdict}")
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Check our trainer against upstream")
    parser.add_argument("--presets", nargs="+", default=["dr", "plr", "accel"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_updates", type=int, default=500)
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    failures = []
    for preset in args.presets:
        report = check_parity(preset, seed=args.seed, num_updates=args.num_updates, out_dir=args.out_dir)
        if not report["identical"]:
            failures.append((preset, report["worst"]))
    if failures:
        print("\nPARITY FAILURES:")
        for preset, worst in failures:
            print(f"  {preset}: worst metric {worst}")
        sys.exit(1)
    print("\nAll presets match upstream exactly.")
