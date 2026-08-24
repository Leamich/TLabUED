"""Run logging: wandb (offline by default) mirrored to a plain CSV.

Two consumers, two formats. wandb gets everything including rendered levels and
eval animations, which is what you want while watching a run. The CSV gets the
scalars only and is what the notebook plots and what survives a pod being
destroyed - no API key, no network, no sync step required.

Layout per run:
    <out_dir>/runs/<run_name>/<seed>/metrics.csv
    <out_dir>/runs/<run_name>/<seed>/meta.json     provenance: git SHAs, GPU, config
    <out_dir>/runs/<run_name>/<seed>/train.log     stdout, when launched via sweep.py
"""

from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import time
from typing import Any, Dict, Optional

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def run_dir(config: Dict[str, Any]) -> str:
    return os.path.join(config["out_dir"], "runs", str(config["run_name"]), str(config["seed"]))


def checkpoint_dir(config: Dict[str, Any]) -> str:
    """Upstream's layout, which is also the layout the assignment asks us to submit."""
    return os.path.join(
        config["out_dir"], "checkpoints", str(config["run_name"]), str(config["seed"])
    )


def _git_sha(path: str) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "-C", path, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def collect_provenance(config: Dict[str, Any]) -> Dict[str, Any]:
    """Everything needed to explain a number six weeks later."""
    import jax

    devices = jax.devices()
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tlab_ued_sha": _git_sha(REPO_ROOT),
        "jaxued_sha": _git_sha(os.path.join(REPO_ROOT, "third_party", "jaxued")),
        "jax_version": jax.__version__,
        "devices": [f"{d.device_kind} ({d.platform})" for d in devices],
        "python": platform.python_version(),
        "hostname": platform.node(),
        "config": config,
    }


def _flatten_scalars(d: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
    """Keep only things that make sense in a CSV cell."""
    flat: Dict[str, float] = {}
    for key, value in d.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten_scalars(value, prefix=f"{name}/"))
        elif isinstance(value, (int, float, np.integer, np.floating)):
            flat[name] = float(value)
        elif isinstance(value, np.ndarray) and value.ndim == 0:
            flat[name] = float(value)
        elif hasattr(value, "shape") and getattr(value, "ndim", 1) == 0:
            flat[name] = float(value)
    return flat


# Metrics `log_eval` handles by name; everything else a teacher returns is
# logged generically under `train/`.
_RESERVED_STATS = frozenset(
    {
        "update_count",
        "eval_solve_rates",
        "eval_returns",
        "eval_ep_lengths",
        "eval_animation",
        "losses",
        "mean_num_blocks",
        "media",
        "time_delta",
    }
)


class Logger:
    """wandb + CSV writer for one training run."""

    def __init__(self, config: Dict[str, Any], tags=None):
        import wandb

        self.config = config
        self.wandb = wandb
        self.dir = run_dir(config)
        os.makedirs(self.dir, exist_ok=True)

        # Offline unless the caller says otherwise: long runs should never block
        # on a missing API key.
        os.environ.setdefault("WANDB_MODE", "offline")
        self.run = wandb.init(
            config=config,
            project=config["project"],
            group=config["run_name"],
            name=f"{config['run_name']}_s{config['seed']}",
            tags=list(tags or []),
            dir=self.dir,
            reinit=True,
        )
        wandb.define_metric("num_updates")
        wandb.define_metric("num_env_steps")
        for prefix in ("solve_rate", "level_sampler", "agent", "return", "eval_ep_lengths"):
            wandb.define_metric(f"{prefix}/*", step_metric="num_updates")

        with open(os.path.join(self.dir, "meta.json"), "w") as f:
            json.dump(collect_provenance(config), f, indent=2, default=str)

        self.csv_path = os.path.join(self.dir, "metrics.csv")
        self._csv_fields = None
        self._csv_file = None
        self._csv_writer = None

    # --- csv ------------------------------------------------------------------
    def _write_csv_row(self, row: Dict[str, float]) -> None:
        if self._csv_writer is None:
            # Column set is fixed by the first row; later keys that were not in
            # it are dropped rather than silently shifting columns.
            self._csv_fields = list(row)
            resume = self.config.get("resume") and os.path.exists(self.csv_path)
            self._csv_file = open(self.csv_path, "a" if resume else "w", newline="")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._csv_fields)
            if not resume:
                self._csv_writer.writeheader()
        self._csv_writer.writerow({k: row.get(k) for k in self._csv_fields})
        self._csv_file.flush()

    # --- the per-eval-step log ------------------------------------------------
    def log_eval(self, stats: Dict[str, Any], teacher_info: Dict[str, Any]) -> Dict[str, float]:
        """Mirrors upstream's `log_eval`, plus the CSV row. Returns the scalars."""
        config = self.config
        wandb = self.wandb

        update_count = int(stats["update_count"])
        env_steps = update_count * config["num_train_envs"] * config["num_steps"]
        print(f"Logging update: {update_count}", flush=True)

        # `sps` keeps upstream's formula (cumulative steps / last interval) so the
        # numbers stay comparable with theirs - but it is not a rate: it climbs
        # steadily as training proceeds. `steps_per_second` is the real one.
        interval_steps = config["eval_freq"] * config["num_train_envs"] * config["num_steps"]
        log_dict: Dict[str, Any] = {
            "num_updates": update_count,
            "num_env_steps": env_steps,
            "sps": env_steps / stats["time_delta"],
            "steps_per_second": interval_steps / stats["time_delta"],
            "time_delta": stats["time_delta"],
        }

        solve_rates = np.asarray(stats["eval_solve_rates"])
        returns = np.asarray(stats["eval_returns"])
        log_dict.update(
            {f"solve_rate/{name}": r for name, r in zip(config["eval_levels"], solve_rates)}
        )
        log_dict["solve_rate/mean"] = solve_rates.mean()
        log_dict.update({f"return/{name}": r for name, r in zip(config["eval_levels"], returns)})
        log_dict["return/mean"] = returns.mean()
        log_dict["eval_ep_lengths/mean"] = np.asarray(stats["eval_ep_lengths"]).mean()

        losses = stats.get("losses")
        if losses is not None:
            # Key names follow upstream's so the dashboards line up.
            total, (critic_loss, actor_loss, entropy) = losses
            log_dict.update(
                {
                    "agent/loss": float(np.asarray(total).mean()),
                    "agent/critic_loss": float(np.asarray(critic_loss).mean()),
                    "agent/actor_loss": float(np.asarray(actor_loss).mean()),
                    "agent/entropy": float(np.asarray(entropy).mean()),
                }
            )
        if "mean_num_blocks" in stats:
            log_dict["level/mean_num_blocks"] = float(np.asarray(stats["mean_num_blocks"]).mean())

        # Anything else a teacher's branches returned, averaged over the updates
        # in this eval block. Upstream's teachers return nothing here, so the
        # baselines' columns are unchanged; a teacher that measures its own
        # curriculum (e.g. `sfl_accel`'s success rate) gets it into the CSV
        # without a new call site.
        for key, value in stats.items():
            if key in _RESERVED_STATS:
                continue
            array = np.asarray(value)
            if array.ndim == 0:
                log_dict[f"train/{key}"] = float(array)

        log_dict.update(teacher_info.get("log", {}))

        scalars = _flatten_scalars(log_dict)
        self._write_csv_row(scalars)

        # media (wandb only)
        media = stats.get("media") or {}
        info = teacher_info.get("info", {})
        for name, image in media.items():
            if name.endswith("_levels"):
                kind = name[: -len("_levels")]
                if int(info.get(f"num_{kind}_updates", 1)) <= 0:
                    continue
                log_dict[f"images/{name}"] = [wandb.Image(np.asarray(i)) for i in image]
            else:
                log_dict[f"images/{name}"] = wandb.Image(np.asarray(image), caption=name)

        animation = stats.get("eval_animation")
        if animation is not None:
            frames_all, episode_lengths = animation
            for i, level_name in enumerate(config["eval_levels"]):
                frames = np.asarray(frames_all[:, i][: int(episode_lengths[i])])
                log_dict[f"animations/{level_name}"] = wandb.Video(frames, fps=4)

        wandb.log(log_dict)
        return scalars

    def finish(self) -> None:
        if self._csv_file is not None:
            self._csv_file.close()
        try:
            self.wandb.finish()
        except Exception:
            pass
