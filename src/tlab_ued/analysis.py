"""Reading runs back: aggregation and plots for the report.

Everything here works off the per-run `metrics.csv` files, so results can be
analysed on a laptop after copying `runs/` off the pod - no GPU, no wandb, no
JAX import.

    from tlab_ued.analysis import load_runs, plot_curves, final_table
    df = load_runs("/workspace/tlab_ued")
    plot_curves(df)
    final_table(df)
"""

from __future__ import annotations

import glob
import json
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from tlab_ued.config import EVAL_LEVELS


def load_runs(out_dir: str = ".", pattern: str = "*") -> pd.DataFrame:
    """Every eval-step row of every run under `<out_dir>/runs/`.

    Adds `run_name`, `seed` and (when meta.json is present) the teacher and
    score function, so runs can be grouped without re-deriving them from names.
    """
    frames: List[pd.DataFrame] = []
    for path in sorted(glob.glob(os.path.join(out_dir, "runs", pattern, "*", "metrics.csv"))):
        seed_dir = os.path.dirname(path)
        run_name = os.path.basename(os.path.dirname(seed_dir))
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if frame.empty:
            continue
        frame["run_name"] = run_name
        frame["seed"] = int(os.path.basename(seed_dir))
        meta_path = os.path.join(seed_dir, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                config = json.load(f).get("config", {})
            frame["teacher"] = config.get("teacher")
            frame["score_function"] = config.get("score_function")
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def curve(
    df: pd.DataFrame, metric: str = "solve_rate/mean", group: str = "run_name"
) -> pd.DataFrame:
    """Mean and standard error across seeds, per group, per update count."""
    grouped = df.groupby([group, "num_updates"])[metric]
    out = grouped.agg(["mean", "std", "count"]).reset_index()
    out["sem"] = out["std"] / np.sqrt(out["count"].clip(lower=1))
    return out


def plot_curves(
    df: pd.DataFrame,
    metric: str = "solve_rate/mean",
    group: str = "run_name",
    ax=None,
    title: Optional[str] = None,
):
    """Learning curves with a mean +/- s.e.m. band per method."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))
    stats = curve(df, metric=metric, group=group)
    for name, part in stats.groupby(group):
        seeds = int(part["count"].max())
        ax.plot(part["num_updates"], part["mean"], label=f"{name} (n={seeds})")
        ax.fill_between(
            part["num_updates"],
            part["mean"] - part["sem"].fillna(0),
            part["mean"] + part["sem"].fillna(0),
            alpha=0.2,
        )
    ax.set_xlabel("PPO updates")
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} on the held-out levels")
    ax.legend()
    ax.grid(alpha=0.3)
    return ax


def ema(values: Sequence[float], gamma: float) -> float:
    """Exponential smoothing `s <- gamma * s + (1 - gamma) * x`, seeded with the first value.

    Every evaluation of a run contributes, and roughly the last `1 / (1 - gamma)`
    of them dominate - which keeps the number on the end of training without
    resting on one noisy evaluation.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan")
    smoothed = values[0]
    for value in values[1:]:
        smoothed = gamma * smoothed + (1.0 - gamma) * value
    return float(smoothed)


def per_seed_final(
    df: pd.DataFrame,
    columns: Sequence[str],
    group: str = "run_name",
    last_k: int = 1,
    ema_gamma: Optional[float] = None,
) -> pd.DataFrame:
    """One final value per (group, seed) and column.

    The mean of the last `last_k` evaluations, or - when `ema_gamma` is given - the
    EMA over all of them in update order.
    """
    ordered = df.sort_values("num_updates")
    if ema_gamma is None:
        return ordered.groupby([group, "seed"]).tail(last_k).groupby([group, "seed"])[
            list(columns)
        ].mean()
    return ordered.groupby([group, "seed"])[list(columns)].agg(
        lambda series: ema(series.dropna(), ema_gamma)
    )


def final_table(
    df: pd.DataFrame,
    metric: str = "solve_rate/mean",
    group: str = "run_name",
    last_k: int = 1,
    ema_gamma: Optional[float] = None,
) -> pd.DataFrame:
    """Per-method final score: mean over seeds of each seed's final value.

    A seed's final value is the mean of its last `last_k` evals, or the EMA of all
    its evals when `ema_gamma` is given. The README quotes the EMA with gamma = 0.8.
    """
    per_seed = per_seed_final(df, [metric], group, last_k, ema_gamma).reset_index()
    out = per_seed.groupby(group)[metric].agg(["mean", "std", "count"]).reset_index()
    out["sem"] = out["std"] / np.sqrt(out["count"].clip(lower=1))
    return out.sort_values("mean", ascending=False).reset_index(drop=True)


def per_level_table(
    df: pd.DataFrame,
    levels: Sequence[str] = tuple(EVAL_LEVELS),
    group: str = "run_name",
    last_k: int = 1,
    ema_gamma: Optional[float] = None,
) -> pd.DataFrame:
    """Final solve rate per held-out level - which levels a method wins or loses."""
    columns = [f"solve_rate/{name}" for name in levels if f"solve_rate/{name}" in df.columns]
    out = per_seed_final(df, columns, group, last_k, ema_gamma).groupby(level=group).mean()
    out.columns = [c.replace("solve_rate/", "") for c in out.columns]
    return out


def plot_per_level(
    df: pd.DataFrame,
    group: str = "run_name",
    last_k: int = 1,
    ema_gamma: Optional[float] = None,
    ax=None,
):
    """Grouped bars: one cluster per held-out level, one bar per method."""
    import matplotlib.pyplot as plt

    table = per_level_table(df, group=group, last_k=last_k, ema_gamma=ema_gamma)
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 4.5))
    n_methods = len(table.index)
    width = 0.8 / max(n_methods, 1)
    x = np.arange(len(table.columns))
    for i, (name, row) in enumerate(table.iterrows()):
        ax.bar(x + i * width - 0.4 + width / 2, row.values, width=width, label=str(name))
    ax.set_xticks(x)
    ax.set_xticklabels(table.columns, rotation=30, ha="right")
    ax.set_ylabel("solve rate")
    ax.set_title("Final solve rate per held-out level")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    return ax


def throughput(df: pd.DataFrame, group: str = "run_name") -> pd.DataFrame:
    """Real steps/second and the implied hours for a full 30k-update run.

    Note `sps` in the CSV is upstream's formula - cumulative env steps divided
    by the last interval's wall time - which climbs throughout a run and is not
    a rate. Prefer `steps_per_second`; fall back to deriving it from
    `time_delta` for runs logged before that column existed.
    """
    agg = {"time_delta": ("time_delta", "median")}
    if "steps_per_second" in df.columns:
        agg["steps_per_second"] = ("steps_per_second", "median")
    out = df.groupby(group).agg(**agg)
    interval_steps = 250 * 32 * 256  # eval_freq * num_train_envs * num_steps
    derived = interval_steps / out["time_delta"]
    if "steps_per_second" not in out.columns:
        out["steps_per_second"] = derived
    else:
        # Mixing runs logged before and after that column existed: the old ones
        # would otherwise be a silent NaN in the middle of the table.
        out["steps_per_second"] = out["steps_per_second"].fillna(derived)
    out["hours_per_30k_updates"] = out["time_delta"] * (30000 / 250) / 3600
    return out.reset_index()


def summarize(
    out_dir: str = ".", last_k: int = 1, ema_gamma: Optional[float] = None
) -> Dict[str, Any]:
    """One call for the notebook: curves table, final table, per-level table."""
    df = load_runs(out_dir)
    if df.empty:
        return {"runs": df, "final": df, "per_level": df, "throughput": df}
    return {
        "runs": df,
        "final": final_table(df, last_k=last_k, ema_gamma=ema_gamma),
        "per_level": per_level_table(df, last_k=last_k, ema_gamma=ema_gamma),
        "throughput": throughput(df),
    }
