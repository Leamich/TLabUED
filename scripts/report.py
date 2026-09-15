"""Turn finished runs into the figures and tables that go in the report.

    python scripts/report.py --out_dir /workspace/tlab_ued

Reads only `runs/*/*/metrics.csv`, so it needs no GPU and no wandb - it works
just as well on a laptop after copying `runs/` off the pod. Writes:

    results/figs/solve_rate.png      learning curves + per-level bars
    results/figs/curriculum.png      what the teacher was feeding the student
    results/summary.md               the numbers, with the budget accounting
    results/final_table.csv          per-method final solve rate
    results/per_level_table.csv      per-method, per-level solve rate

By default only the configurations the README reports are included; `--all_runs`
adds everything else under `runs/` (the ablations of the upper bound). Final
numbers use the README's aggregation: an EMA with gamma = 0.8 over each seed's
evaluations, averaged over seeds.
"""

from __future__ import annotations

import argparse
import os
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from tlab_ued.analysis import (  # noqa: E402
    final_table,
    load_runs,
    per_level_table,
    plot_curves,
    plot_per_level,
    throughput,
)

# The configurations the README reports. Everything else under runs/ is left out
# of the figures and tables unless --all_runs is given.
REPORT_RUNS = (
    "dr",
    "plr_maxmc",
    "accel_maxmc",
    "sfl_accel_learnability",
    "sfl_oracle_learnability_level",
    "sfl_accel_learnability_n64",
    "sfl_oracle_learnability_level_bfs",
)

# Columns a teacher logs about its own curriculum, and how to read them.
CURRICULUM_PANELS = [
    (
        "train/success_rate",
        "Success rate logged during training",
        "Mean over every rollout of the last 250 updates. On replay it is the\n"
        "smoothed stored p of the replayed levels, elsewhere p measured over 4\n"
        "envs per level - not the success rate of the gradient batch alone.",
    ),
    (
        "train/learnability",
        "Learnability p(1-p) of the training batch",
        "0.25 is the ceiling (p = 0.5).",
    ),
    (
        "level_sampler/mean_p",
        "Mean success rate over the level buffer",
        "The buffer's difficulty, in units that mean the same thing at every\n"
        "point in training.",
    ),
    (
        "level/mean_num_blocks",
        "Walls per level in the training batch",
        "Structural complexity of what the curriculum produces, and the one\n"
        "curriculum column every teacher logs - so it compares across methods.",
    ),
]

# Panels that only mean something as a pair: the point is the gap between two
# columns, not either curve alone.
PAIRED_PANELS = [
    (
        "sfl/topk_learnability",
        "sfl/population_learnability",
        "kept",
        "population",
        "SFL phase: what selection buys",
        "Learnability of the top-k the phase kept, against the population it\n"
        "drew them from. If these meet, selection is doing nothing and the\n"
        "phase is wasted budget.",
    ),
    (
        "oracle/selected_learnability",
        "oracle/control_learnability",
        "picked",
        "control",
        "Oracle: picked vs uniformly drawn levels",
        "Measured learnability of the levels the oracle picked, against the uniform\n"
        "controls measured in the same rollouts. The summary's gain is the ratio\n"
        "of their means over all phases, not the mean of per-phase ratios.",
    ),
    (
        "oracle/buffer_mean_p",
        "level_sampler/mean_p",
        "oracle estimate",
        "last measurement",
        "Buffer staleness: predicted p vs stored p",
        "The oracle's current view of the buffer against the last measured\n"
        "value of each entry. These diverging is the staleness that free\n"
        "re-scoring exists to fix.",
    ),
]


def plot_curriculum(df: pd.DataFrame, out_path: str) -> List[str]:
    """Per-run curriculum diagnostics. Skips panels no run logged."""
    available = [(c, t, n) for c, t, n in CURRICULUM_PANELS if c in df.columns]
    paired = [p for p in PAIRED_PANELS if {p[0], p[1]} <= set(df.columns)]
    panels = len(available) + len(paired)
    if not panels:
        return []

    rows = (panels + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(13, 4.2 * rows), squeeze=False)
    flat = [ax for row in axes for ax in row]

    for ax, (column, title, note) in zip(flat, available):
        for name, run in df.groupby("run_name"):
            series = run[["num_updates", column]].dropna()
            if series.empty:
                continue
            ax.plot(series["num_updates"], series[column], label=str(name))
        ax.set_title(title)
        ax.set_xlabel("updates")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        if note:
            ax.text(
                0.02, 0.02, note, transform=ax.transAxes, fontsize=7, va="bottom", alpha=0.75
            )

    for ax, (col_a, col_b, label_a, label_b, title, note) in zip(flat[len(available) :], paired):
        for name, run in df.groupby("run_name"):
            first = run[["num_updates", col_a]].dropna()
            second = run[["num_updates", col_b]].dropna()
            if first.empty or second.empty:
                continue
            line = ax.plot(first["num_updates"], first[col_a], label=f"{name}: {label_a}")[0]
            ax.plot(
                second["num_updates"],
                second[col_b],
                linestyle="--",
                color=line.get_color(),
                label=f"{name}: {label_b}",
            )
        ax.set_title(title)
        ax.set_xlabel("updates")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        ax.text(
            0.02, 0.02, note, transform=ax.transAxes, fontsize=7, va="bottom", alpha=0.75
        )

    for ax in flat[panels:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [c for c, _, _ in available] + [f"{a} vs {b}" for a, b, *_ in paired]


def oracle_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-run oracle selection: picked vs control learnability over all phases.

    `gain` is the ratio of the two means, not the mean of the logged per-phase
    `oracle/selection_gain`: late in training both groups measure close to zero,
    and an average of per-phase ratios is mostly an average of that noise.
    """
    columns = ["oracle/selected_learnability", "oracle/control_learnability"]
    if not set(columns) <= set(df.columns):
        return pd.DataFrame()
    out = df.groupby(["run_name", "seed"])[columns].mean()
    out.columns = ["picked", "control"]
    # `--no-oracle_verify` never measures a shortlist, so its statistics stay at
    # their zero initialisation - the absence of a measurement, not a zero.
    out = out[(out["picked"] > 0) & (out["control"] > 0)]
    out["gain"] = out["picked"] / out["control"]
    return out.round(4)


def budget_table(df: pd.DataFrame) -> pd.DataFrame:
    """What each run actually spent, from its own last row.

    Env steps are exact for every run. The branch counts are only there for a
    teacher that logs them - upstream's PLR/ACCEL keep those counters out of the
    logged dict, so the ACCEL baseline's split stays the analytic one from
    `branch_budget`.
    """
    columns = ["num_env_steps", "num_updates"] + [
        c for c in df.columns if c.startswith("branch/")
    ]
    last = df.sort_values("num_updates").groupby(["run_name", "seed"]).tail(1)
    return last.set_index(["run_name", "seed"])[columns]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", type=str, default=".")
    parser.add_argument(
        "--ema_gamma",
        type=float,
        default=0.8,
        help="final number of a seed: EMA over all its evaluations with this gamma",
    )
    parser.add_argument(
        "--last_k",
        type=int,
        default=None,
        help="use the mean of the last k evaluations instead of the EMA",
    )
    parser.add_argument(
        "--all_runs",
        action="store_true",
        help="include every run under runs/, not only the configurations the README reports",
    )
    args = parser.parse_args()
    ema_gamma = None if args.last_k is not None else args.ema_gamma
    last_k = args.last_k or 1
    aggregation = (
        f"the mean of the last {last_k} evaluations of each seed"
        if ema_gamma is None
        else f"an EMA (gamma = {ema_gamma}) over all evaluations of each seed"
    )

    df = load_runs(args.out_dir)
    if df.empty:
        raise SystemExit(f"no runs under {args.out_dir}/runs")
    df = df[~df["run_name"].str.endswith("_smoke")]
    df = df[~df["run_name"].str.startswith("parity")]
    # Short diagnostic runs, not experiments: they share the run directory but
    # not the budget, so averaging them into any table is misleading.
    df = df[~df["run_name"].str.startswith("oracle_probe")]
    if not args.all_runs:
        df = df[df["run_name"].isin(REPORT_RUNS)]

    results = os.path.join(args.out_dir, "results")
    figs = os.path.join(results, "figs")
    os.makedirs(figs, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(11, 10))
    plot_curves(df, ax=axes[0])
    plot_per_level(df, ax=axes[1], last_k=last_k, ema_gamma=ema_gamma)
    fig.tight_layout()
    fig.savefig(os.path.join(figs, "solve_rate.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    plotted = plot_curriculum(df, os.path.join(figs, "curriculum.png"))

    final = final_table(df, last_k=last_k, ema_gamma=ema_gamma)
    per_level = per_level_table(df, last_k=last_k, ema_gamma=ema_gamma).round(3)
    speed = throughput(df).round(1)
    budget = budget_table(df)

    final.to_csv(os.path.join(results, "final_table.csv"), index=False)
    per_level.to_csv(os.path.join(results, "per_level_table.csv"))

    # Which seeds are behind each row of every table below. With one seed per
    # method this is the first thing a reader needs, not a footnote.
    inventory = (
        df.groupby(["run_name", "seed"])["num_updates"]
        .max()
        .reset_index()
        .rename(columns={"num_updates": "updates"})
        .sort_values(["run_name", "seed"])
    )

    lines = [
        "# Results",
        "",
        f"The final number of a seed is {aggregation}; a method's is the mean over seeds.",
        "",
        "## Runs",
        "",
        inventory.to_markdown(index=False),
        "",
        "## Held-out solve rate",
        "",
        final.round(4).to_markdown(index=False),
        "",
        "## Per level",
        "",
        per_level.to_markdown(),
        "",
        "## Budget actually spent",
        "",
        budget.to_markdown(),
        "",
        "## Throughput",
        "",
        speed.to_markdown(index=False),
        "",
    ]
    oracle = oracle_table(df)
    if not oracle.empty:
        lines += [
            "## Oracle selection",
            "",
            "Mean over all SFL phases of the measured learnability of the levels the oracle "
            "picked (`picked`) and of the uniformly drawn controls measured in the same rollouts "
            "(`control`); `gain` is the ratio of those means (1.0 = chance).",
            "",
            oracle.to_markdown(),
            "",
        ]
    if plotted:
        lines += [
            "## Curriculum diagnostics",
            "",
            "See `results/figs/curriculum.png`. Columns present: " + ", ".join(plotted) + ".",
            "",
        ]
    summary = os.path.join(results, "summary.md")
    with open(summary, "w") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print(f"\nwrote {summary} and {figs}/")


if __name__ == "__main__":
    main()
