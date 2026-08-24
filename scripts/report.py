"""Turn finished runs into the figures and tables that go in the report.

    python scripts/report.py --out_dir /workspace/tlab_ued

Reads only `runs/*/*/metrics.csv`, so it needs no GPU and no wandb - it works
just as well on a laptop after copying `runs/` off the pod. Writes:

    results/figs/solve_rate.png      learning curves + per-level bars
    results/figs/curriculum.png      what the teacher was feeding the student
    results/summary.md               the numbers, with the budget accounting
    results/final_table.csv          per-method final solve rate
    results/per_level_table.csv      per-method, per-level solve rate

The curriculum figure is the one worth reading first: with one seed per method
the solve-rate gap is inside seed noise, but `train/success_rate` and the SFL
selection gain are per-run measurements of whether the mechanism did what it
claims, and those do not need a second seed to be informative.
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

# Columns a teacher logs about its own curriculum, and how to read them.
CURRICULUM_PANELS = [
    (
        "train/success_rate",
        "Success rate on the training batch",
        "p on the levels the student is being trained on. SFL's claim is that\n"
        "this sits near 0.5; a curriculum drifting to 0 is showing impossible\n"
        "levels, to 1 trivial ones.",
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
    (
        "oracle/selection_gain",
        "Oracle selection gain (measured)",
        "Learnability of the levels the oracle picked over the uniformly drawn\n"
        "controls measured beside them. 1.0 is chance. Out of sample: it commits\n"
        "to a ranking over 8192 levels, and only then do the rollouts run.",
    ),
    (
        "oracle/control_rank_corr",
        "Oracle rank correlation on the controls",
        "Spearman between predicted and measured p on the uniform controls. The\n"
        "shortlist version (oracle/rank_corr) is range-restricted by the very\n"
        "selection that produced it, so this is the honest one.",
    ),
    (
        "oracle/control_brier",
        "Oracle calibration off its own distribution",
        "Brier score on uniformly drawn levels. Most of the oracle's training\n"
        "data is buffer levels, so this is the extrapolation it is asked to do\n"
        "every phase.",
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
        "oracle/predicted_learnability",
        "oracle/selected_learnability",
        "predicted",
        "realised",
        "Oracle: the winner's curse",
        "What the oracle expected of its picks, against what the rollouts then\n"
        "found. An argmax over a prediction selects partly for the prediction\n"
        "being wrong; the gap is that error, and is what verification absorbs.",
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


# The oracle's self-measurement, averaged over the phases of a run. Every one of
# these is a comparison the run performed on itself against ground truth, so
# unlike the solve rate they carry an argument at one seed.
ORACLE_COLUMNS = {
    "oracle/selection_gain": "gain",
    "oracle/selected_learnability": "picked",
    "oracle/control_learnability": "control",
    "oracle/predicted_learnability": "predicted",
    "oracle/control_rank_corr": "rank corr",
    "oracle/control_brier": "brier",
    "oracle/loss": "loss",
}


def oracle_table(df: pd.DataFrame, warmup_updates: int = 1000) -> pd.DataFrame:
    """Per-run oracle diagnostics, over the phases where the ranking was used.

    Rows before `warmup_updates` are dropped: until then the shortlist is drawn
    at random on purpose, so a selection gain from that period measures nothing.
    """
    present = {c: n for c, n in ORACLE_COLUMNS.items() if c in df.columns}
    if not present:
        return pd.DataFrame()
    warm = df[df["num_updates"] >= warmup_updates]
    out = warm.groupby(["run_name", "seed"])[list(present)].mean()
    out = out.rename(columns=present).dropna(how="all")
    # `--no-oracle_verify` never measures a shortlist, so its phase statistics
    # stay at their zero initialisation. An all-zero row is the absence of a
    # measurement, not a measurement of zero, so drop it rather than print it.
    measured = [c for c in ("gain", "picked", "predicted") if c in out.columns]
    if measured:
        out = out[out[measured].abs().sum(axis=1) > 0]
    return out.round(3)


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
        "--last_k",
        type=int,
        default=3,
        help="average the last k evaluations for the final numbers",
    )
    args = parser.parse_args()

    df = load_runs(args.out_dir)
    if df.empty:
        raise SystemExit(f"no runs under {args.out_dir}/runs")
    df = df[~df["run_name"].str.endswith("_smoke")]
    df = df[~df["run_name"].str.startswith("parity")]
    # Short diagnostic runs, not experiments: they share the run directory but
    # not the budget, so averaging them into any table is misleading.
    df = df[~df["run_name"].str.startswith("oracle_probe")]

    results = os.path.join(args.out_dir, "results")
    figs = os.path.join(results, "figs")
    os.makedirs(figs, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(11, 10))
    plot_curves(df, ax=axes[0])
    plot_per_level(df, ax=axes[1], last_k=args.last_k)
    fig.tight_layout()
    fig.savefig(os.path.join(figs, "solve_rate.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    plotted = plot_curriculum(df, os.path.join(figs, "curriculum.png"))

    final = final_table(df, last_k=args.last_k)
    per_level = per_level_table(df, last_k=args.last_k).round(3)
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
        f"Final numbers average the last {args.last_k} evaluations of each seed.",
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
            "## Oracle diagnostics",
            "",
            "Averaged over the phases after warm-up. `gain` is measured learnability of the "
            "oracle's picks over the uniform controls beside them (1.0 = chance); `predicted` "
            "against `picked` is the winner's curse; `rank corr` and `brier` are on the controls, "
            "the only sample the selection did not restrict.",
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
