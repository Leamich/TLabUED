"""BFS diagnostics for the levels a teacher actually trains on.

A curriculum can only teach what its levels admit: a batch that is 30%
unsolvable spends 30% of the student's budget on episodes with no reachable
reward, and a batch whose optimal solution is 8 steps long teaches nothing after
the first thousand updates. Both failure modes are invisible in the training
curves - they show up only as a flat eval solve rate - so this module measures
them directly, before the run starts.

The measurement is an exact shortest-path search over the *environment's* action
space, not over grid cells: a state is `(agent_pos, agent_dir)` and the moves are
`left`, `right`, `forward`, so the reported number is the true minimum number of
env steps to finish the episode. It ends the way `Maze` ends it - by stepping
*into* the goal cell from an adjacent cell facing it - so `steps` is directly
comparable to `max_steps_in_episode` and converts to the best attainable return
via the env's own time penalty.

Used two ways:

    # at launch, printed by train.py (--diagnose_levels 0 turns it off)
    python -m tlab_ued.train --preset accel

    # standalone, e.g. to compare generators before committing to a run
    python -m tlab_ued.level_diagnostics --preset accel --diagnose_levels 8192
    python -m tlab_ued.level_diagnostics --level_generator minigrid_walls --n_walls 60
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

# Sentinel for "no solution": large enough to dominate any real path length in a
# 13x13 maze (at most 4*13*13 states), small enough to stay far from int32 range.
UNREACHABLE = 1_000_000

# (dx, dy) per direction index - the same table `jaxued.environments.maze` uses.
_DIR_TO_VEC: Tuple[Tuple[int, int], ...] = ((1, 0), (0, 1), (-1, 0), (0, -1))

# Difficulty buckets, in optimal env steps. Upper bound inclusive; the last one
# catches levels that are solvable but not within the episode.
BUCKETS: Tuple[Tuple[str, int], ...] = (
    ("trivial<=10", 10),
    ("easy<=30", 30),
    ("medium<=60", 60),
    ("hard<=120", 120),
    ("brutal>120", UNREACHABLE - 1),
)


def _shift_in(plane: jnp.ndarray, dx: int, dy: int, fill) -> jnp.ndarray:
    """`out[..., y, x] = plane[..., y - dy, x - dx]`, off-grid reads give `fill`.

    This is the "where could a forward move have come from" gather: a cell's new
    distance via `forward` comes from the cell one step behind it.
    """
    h, w = plane.shape[-2:]
    pad = ((0, 0),) * (plane.ndim - 2) + ((1, 1), (1, 1))
    padded = jnp.pad(plane, pad, constant_values=fill)
    return padded[..., 1 - dy : 1 - dy + h, 1 - dx : 1 - dx + w]


@jax.jit
def solve_levels(levels) -> Dict[str, jnp.ndarray]:
    """Exact optimal-play analysis of a batch of levels.

    Args:
        levels: a `Level` pytree with a leading batch dimension.

    Returns a dict of per-level arrays:
        steps:           minimum env steps to end the episode (`UNREACHABLE` if
                         the goal cannot be reached at all)
        reachable_cells: number of cells the agent can stand on
        free_cells:      number of non-wall, non-goal cells
        num_walls:       walls in the level
        well_formatted:  `Level.is_well_formatted()`

    The search is a Bellman-Ford relaxation over `(dir, y, x)` run to a fixed
    point, which is BFS with a uniform cost of 1 per action - every action the
    agent has (`left`, `right`, `forward`) costs exactly one env step. Blocked
    cells are the walls *and the goal*: `Maze` refuses a forward move into the
    goal cell in the same way it refuses one into a wall, it just ends the
    episode while doing so, which is why the answer is `dist_to_a_facing_cell + 1`
    rather than `dist_to_the_goal_cell`.
    """
    wall = jnp.asarray(levels.wall_map, dtype=bool)
    n, h, w = wall.shape
    rows = jnp.arange(n)

    # uint32 in the Level dataclass: cast before any subtraction, or `0 - 1`
    # silently becomes 4294967295.
    gx = jnp.asarray(levels.goal_pos[:, 0], dtype=jnp.int32)
    gy = jnp.asarray(levels.goal_pos[:, 1], dtype=jnp.int32)
    ax = jnp.asarray(levels.agent_pos[:, 0], dtype=jnp.int32)
    ay = jnp.asarray(levels.agent_pos[:, 1], dtype=jnp.int32)
    ad = jnp.asarray(levels.agent_dir, dtype=jnp.int32)

    goal_cell = jnp.zeros((n, h, w), dtype=bool).at[rows, gy, gx].set(True)
    blocked = wall | goal_cell

    inf = jnp.int32(UNREACHABLE)
    dist = jnp.full((n, 4, h, w), inf, dtype=jnp.int32).at[rows, ad, ay, ax].set(0)

    def relax(carry):
        i, dist, _ = carry
        # A turn keeps the cell and moves the direction by one, either way.
        turned = jnp.minimum(jnp.roll(dist, 1, axis=1), jnp.roll(dist, -1, axis=1))
        forward = jnp.stack(
            [_shift_in(dist[:, d], dx, dy, inf) for d, (dx, dy) in enumerate(_DIR_TO_VEC)],
            axis=1,
        )
        cand = jnp.minimum(turned, forward)
        cand = jnp.where(cand >= inf, inf, cand + 1)
        cand = jnp.where(blocked[:, None], inf, cand)
        # min with the old value keeps the start cell at 0 even when a malformed
        # level puts the agent on top of a wall.
        new_dist = jnp.minimum(dist, cand)
        return i + 1, new_dist, jnp.any(new_dist != dist)

    max_iters = 4 * h * w + 2
    _, dist, _ = jax.lax.while_loop(
        lambda carry: carry[2] & (carry[0] < max_iters), relax, (0, dist, jnp.bool_(True))
    )

    # The episode ends on a forward move into the goal, so look up the distance
    # to each cell adjacent to the goal while facing it, and add that last step.
    padded = jnp.pad(dist, ((0, 0), (0, 0), (1, 1), (1, 1)), constant_values=inf)
    dxs = jnp.array([d[0] for d in _DIR_TO_VEC], dtype=jnp.int32)
    dys = jnp.array([d[1] for d in _DIR_TO_VEC], dtype=jnp.int32)
    entry = padded[
        rows[:, None],
        jnp.arange(4)[None, :],
        (gy[:, None] - dys[None, :]) + 1,
        (gx[:, None] - dxs[None, :]) + 1,
    ]
    best_entry = entry.min(axis=1)
    steps = jnp.where(best_entry >= inf, inf, best_entry + 1)

    free = ~blocked
    return {
        "steps": steps,
        "reachable_cells": (dist.min(axis=1) < inf).sum(axis=(-2, -1)),
        "free_cells": free.sum(axis=(-2, -1)),
        "num_walls": wall.sum(axis=(-2, -1)),
        "well_formatted": jax.vmap(lambda level: level.is_well_formatted())(levels),
    }


def summarize(levels, max_steps_in_episode: int = 250) -> Dict[str, Any]:
    """Batch statistics from `solve_levels`, as plain Python numbers.

    "Solvable" means solvable *within the episode*: a level whose optimal play
    needs more steps than the time limit is as unrewarding to the student as one
    with no path at all, but it is counted separately because it says something
    different about the generator.
    """
    out = jax.tree_util.tree_map(np.asarray, solve_levels(levels))
    steps = out["steps"].astype(np.int64)
    n = int(steps.shape[0])
    cells = int(np.prod(np.asarray(levels.wall_map).shape[1:]))

    reachable = steps < UNREACHABLE
    solvable = reachable & (steps <= max_steps_in_episode)
    solved_steps = steps[solvable]

    def pct(mask) -> float:
        return float(np.mean(mask) * 100.0)

    stats: Dict[str, Any] = {
        "num_levels": n,
        "max_steps_in_episode": int(max_steps_in_episode),
        "solvable_pct": pct(solvable),
        "unreachable_pct": pct(~reachable),
        "over_time_limit_pct": pct(reachable & ~solvable),
        "malformed_pct": pct(~out["well_formatted"]),
        "mean_num_walls": float(out["num_walls"].mean()),
        "wall_pct": float(out["num_walls"].mean() / max(1, cells) * 100.0),
        "reachable_free_pct": float(
            np.mean(out["reachable_cells"] / np.maximum(1, out["free_cells"])) * 100.0
        ),
    }

    if solved_steps.size:
        # The best return the student could possibly get, under the env's own
        # time penalty: reward = 1 - 0.9 * t / max_steps at the terminal step.
        best_return = 1.0 - 0.9 * solved_steps / max_steps_in_episode
        stats.update(
            {
                "steps_mean": float(solved_steps.mean()),
                "steps_median": float(np.median(solved_steps)),
                "steps_p10": float(np.percentile(solved_steps, 10)),
                "steps_p90": float(np.percentile(solved_steps, 90)),
                "steps_max": int(solved_steps.max()),
                "optimal_return_mean": float(best_return.mean()),
            }
        )
    else:
        stats.update(
            {
                "steps_mean": float("nan"),
                "steps_median": float("nan"),
                "steps_p10": float("nan"),
                "steps_p90": float("nan"),
                "steps_max": 0,
                "optimal_return_mean": float("nan"),
            }
        )

    lower = 0
    buckets: Dict[str, float] = {}
    for name, upper in BUCKETS:
        buckets[name] = pct(solvable & (steps > lower) & (steps <= upper))
        lower = upper
    buckets["unsolvable"] = pct(~solvable)
    stats["buckets_pct"] = buckets
    return stats


def format_summary(stats: Dict[str, Any], title: str) -> str:
    """A few lines meant to be read in a training log, not parsed."""
    buckets = " | ".join(f"{k} {v:.1f}%" for k, v in stats["buckets_pct"].items())
    lines = [
        f"{title}  (n={stats['num_levels']})",
        f"  solvable        {stats['solvable_pct']:6.1f}%   "
        f"(goal unreachable {stats['unreachable_pct']:.1f}%, "
        f"needs >{stats['max_steps_in_episode']} steps {stats['over_time_limit_pct']:.1f}%)",
        f"  optimal steps   median {stats['steps_median']:.0f}  mean {stats['steps_mean']:.1f}  "
        f"p10 {stats['steps_p10']:.0f}  p90 {stats['steps_p90']:.0f}  max {stats['steps_max']}",
        f"  best return     {stats['optimal_return_mean']:.3f} mean over solvable levels",
        f"  difficulty      {buckets}",
        f"  reachable area  {stats['reachable_free_pct']:.1f}% of free cells   "
        f"walls {stats['mean_num_walls']:.1f} ({stats['wall_pct']:.1f}% of the grid)",
    ]
    if stats["malformed_pct"] > 0:
        lines.append(f"  MALFORMED       {stats['malformed_pct']:.1f}% of levels")
    return "\n".join(lines)


def sample_levels(sample_random_level: Callable, rng, num_levels: int):
    return jax.vmap(sample_random_level)(jax.random.split(rng, num_levels))


def mutate_levels(mutate_level: Callable, rng, levels, num_edits: int, rounds: int = 1):
    """Apply the configured mutator `rounds` times, as ACCEL does to a lineage."""
    num_levels = int(np.asarray(levels.wall_map).shape[0])
    for _ in range(rounds):
        rng, rng_mutate = jax.random.split(rng)
        levels = jax.vmap(mutate_level, (0, 0, None))(
            jax.random.split(rng_mutate, num_levels), levels, num_edits
        )
    return levels


def diagnose(
    config: Dict[str, Any],
    sample_random_level: Callable,
    mutate_level: Optional[Callable] = None,
    max_steps_in_episode: int = 250,
    num_levels: Optional[int] = None,
    mutation_rounds: Optional[int] = None,
    seed: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """Diagnose the generator (and, for mutation teachers, its mutator).

    The PRNG key is derived from the seed but is *not* taken from the training
    stream - the diagnostic must not shift a single training draw, or every
    result in `runs/` would depend on whether it was switched on.
    """
    num_levels = int(config.get("diagnose_levels", 0) if num_levels is None else num_levels)
    if num_levels <= 0:
        return {}
    if mutation_rounds is None:
        mutation_rounds = int(config.get("diagnose_mutation_rounds", 1))

    seed = int(config.get("seed", 0) if seed is None else seed)
    rng = jax.random.fold_in(jax.random.PRNGKey(seed), 0xD1A6)
    rng_sample, rng_mutate = jax.random.split(rng)

    levels = sample_levels(sample_random_level, rng_sample, num_levels)
    reports = {"generator": summarize(levels, max_steps_in_episode)}

    # Only teachers that mutate care about the mutated distribution. Note this
    # mutates freshly generated levels, whereas ACCEL mutates replayed ones: it
    # measures the operator, not the curriculum it will end up producing.
    mutates = bool(config.get("use_accel")) or str(config.get("teacher")) == "accel"
    if mutate_level is not None and mutates and mutation_rounds > 0:
        children = mutate_levels(
            mutate_level, rng_mutate, levels, int(config["num_edits"]), mutation_rounds
        )
        reports["mutator"] = summarize(children, max_steps_in_episode)
    return reports


def report(
    config: Dict[str, Any],
    sample_random_level: Callable,
    mutate_level: Optional[Callable] = None,
    max_steps_in_episode: int = 250,
    out_path: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Dict[str, Any]]:
    """`diagnose` plus printing, and a JSON copy next to the run's metrics."""
    reports = diagnose(config, sample_random_level, mutate_level, max_steps_in_episode, **kwargs)
    if not reports:
        return reports

    rounds = kwargs.get("mutation_rounds") or config.get("diagnose_mutation_rounds", 1)
    titles = {
        "generator": (
            f"Levels from generator {config.get('level_generator')!r} "
            f"(n_walls={config.get('n_walls')})"
        ),
        "mutator": (
            f"...after {rounds}x mutator {config.get('level_mutator')!r} "
            f"({config.get('num_edits')} edits)"
        ),
    }
    print("")
    for key, stats in reports.items():
        print(format_summary(stats, titles.get(key, key)), flush=True)
    print("")

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(reports, f, indent=2)
    return reports


def main(argv: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
    """Standalone entry point: build the env and level functions, then report."""
    from jaxued.environments import Maze

    from tlab_ued import levels as level_registry
    from tlab_ued.config import from_args

    config = from_args(argv)
    env = Maze(
        max_height=13,
        max_width=13,
        agent_view_size=config["agent_view_size"],
        normalize_obs=True,
    )
    # A CLI invocation *is* the diagnostic, so an unset --diagnose_levels means
    # "use a sample big enough to trust the tails", not "skip it".
    num_levels = config["diagnose_levels"] or 8192
    return report(
        config,
        level_registry.get_generator(config, env),
        level_registry.get_mutator(config, env),
        env.default_params.max_steps_in_episode,
        num_levels=num_levels,
    )


if __name__ == "__main__":
    main()
