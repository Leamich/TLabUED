"""Level generator and mutator registries.

The second teacher-side plugin point: *how levels come into existence*. A new
generation strategy (structured mazes, curriculum over wall counts, a learned
generator) or a new mutation operator registers here and is selected with
`--level_generator` / `--level_mutator`.

Both built-ins are JaxUED's own, so the baselines stay faithful:
  - "minigrid_walls": `make_level_generator` - uniform random walls/goal/agent
  - "minimax":        `make_level_mutator_minimax` - the mutation operator ACCEL uses
"""

from __future__ import annotations

from typing import Any, Callable, Dict

import chex
import jax
import jax.numpy as jnp
from jaxued.environments.maze import Level, make_level_generator, make_level_mutator_minimax

# A generator factory takes (config, env) and returns `rng -> Level`.
GeneratorFactory = Callable[[Dict[str, Any], Any], Callable]
# A mutator factory takes (config, env) and returns `(rng, level, num_edits) -> Level`.
MutatorFactory = Callable[[Dict[str, Any], Any], Callable]

LEVEL_GENERATORS: Dict[str, GeneratorFactory] = {}
LEVEL_MUTATORS: Dict[str, MutatorFactory] = {}


def register_generator(name: str) -> Callable[[GeneratorFactory], GeneratorFactory]:
    def decorator(fn: GeneratorFactory) -> GeneratorFactory:
        if name in LEVEL_GENERATORS:
            raise ValueError(f"level generator {name!r} is already registered")
        LEVEL_GENERATORS[name] = fn
        return fn

    return decorator


def register_mutator(name: str) -> Callable[[MutatorFactory], MutatorFactory]:
    def decorator(fn: MutatorFactory) -> MutatorFactory:
        if name in LEVEL_MUTATORS:
            raise ValueError(f"level mutator {name!r} is already registered")
        LEVEL_MUTATORS[name] = fn
        return fn

    return decorator


def get_generator(config: Dict[str, Any], env) -> Callable:
    name = config.get("level_generator", "minigrid_walls")
    if name not in LEVEL_GENERATORS:
        raise ValueError(f"Unknown level generator {name!r}. Registered: {sorted(LEVEL_GENERATORS)}")
    return LEVEL_GENERATORS[name](config, env)


def get_mutator(config: Dict[str, Any], env) -> Callable:
    name = config.get("level_mutator", "minimax")
    if name not in LEVEL_MUTATORS:
        raise ValueError(f"Unknown level mutator {name!r}. Registered: {sorted(LEVEL_MUTATORS)}")
    return LEVEL_MUTATORS[name](config, env)


# === Built-ins: what the upstream examples use ===


@register_generator("minigrid_walls")
def minigrid_walls(config: Dict[str, Any], env) -> Callable:
    """Upstream: `make_level_generator(max_height, max_width, n_walls)`."""
    return make_level_generator(env.max_height, env.max_width, config["n_walls"])


# === Validation set ===


def _carve_perfect_maze(rng: chex.PRNGKey, height: int, width: int) -> chex.Array:
    """A uniform-ish perfect maze as a (height, width) wall map.

    Randomised depth-first search on the lattice of even coordinates, which is the
    convention the hand-drawn prefabs use: cells sit at `(2i, 2j)`, the cell
    between two of them is the wall that DFS knocks out, and `(odd, odd)` is
    always wall. `StandardMaze` is exactly this shape.

    Written as a fixed-length `fori_loop` rather than a `while`: every cell is
    pushed once and popped once, so `2 * num_cells` iterations are not a budget
    that might run out, they are the exact length of the search. Once the stack
    empties the remaining iterations are no-ops.
    """
    ch, cw = (height + 1) // 2, (width + 1) // 2
    n = ch * cw
    rng, rng_start = jax.random.split(rng)

    cells = jnp.arange(n)
    wall = jnp.ones((height, width), dtype=bool).at[2 * (cells // cw), 2 * (cells % cw)].set(False)

    start = jax.random.randint(rng_start, (), 0, n)
    visited = jnp.zeros(n, dtype=bool).at[start].set(True)
    stack = jnp.zeros(n, dtype=jnp.int32).at[0].set(start)

    dy = jnp.array([-1, 1, 0, 0])
    dx = jnp.array([0, 0, -1, 1])

    def step(i, carry):
        wall, visited, stack, top = carry
        active = top >= 0
        cur = stack[jnp.clip(top, 0, n - 1)]
        cy, cx = cur // cw, cur % cw

        ny, nx = cy + dy, cx + dx
        in_bounds = (ny >= 0) & (ny < ch) & (nx >= 0) & (nx < cw)
        neighbour = jnp.clip(ny, 0, ch - 1) * cw + jnp.clip(nx, 0, cw - 1)
        open_to = in_bounds & ~visited[neighbour]

        # Random tie-break among the unvisited neighbours: uniform noise on the
        # candidates, -1 elsewhere, argmax. Cheaper than a masked categorical and
        # exactly as uniform.
        noise = jax.random.uniform(jax.random.fold_in(rng, i), (4,))
        pick = jnp.argmax(jnp.where(open_to, noise, -1.0))
        advance = active & open_to[pick]
        target = neighbour[pick]

        # The wall between two cells is their midpoint in map coordinates.
        wall = wall.at[cy + ny[pick], cx + nx[pick]].set(
            jnp.where(advance, False, wall[cy + ny[pick], cx + nx[pick]])
        )
        visited = visited | (advance & (cells == target))
        stack = stack.at[jnp.clip(top + 1, 0, n - 1)].set(
            jnp.where(advance, target, stack[jnp.clip(top + 1, 0, n - 1)])
        )
        top = jnp.where(active, jnp.where(advance, top + 1, top - 1), top)
        return wall, visited, stack, top

    wall, _, _, _ = jax.lax.fori_loop(0, 2 * n, step, (wall, visited, stack, jnp.int32(0)))
    return wall


@register_generator("perfect_maze")
def perfect_maze(config: Dict[str, Any], env) -> Callable:
    """Structured mazes with long routes - the *validation* generator.

    Never used for training. `minigrid_walls` scatters 25 walls at random, which
    makes levels whose optimal route has a median of 11 steps and a maximum of 27
    (README §3.1); the held-out levels are hand-drawn mazes whose routes are an
    order of magnitude longer. That gap is why tuning against levels this
    generator produces says little about the levels we are scored on.

    A perfect maze has exactly one route between any two cells, so the route is
    forced to wind. It gives the oracle bench an out-of-distribution axis - can a
    model fit on `minigrid_walls` rank *these* - without touching the eight
    held-out levels, which docs/TASK.md puts off limits for tuning and asks us to
    replace with a validation set of our own.
    """
    height, width = env.max_height, env.max_width

    def sample(rng: chex.PRNGKey) -> Level:
        rng_maze, rng_pos, rng_dir = jax.random.split(rng, 3)
        wall_map = _carve_perfect_maze(rng_maze, height, width)

        free = (~wall_map).reshape(-1).astype(jnp.float32)
        both = jax.random.choice(rng_pos, height * width, (2,), replace=False, p=free)
        agent_idx, goal_idx = both[0], both[1]

        return Level(
            wall_map=wall_map,
            goal_pos=jnp.array([goal_idx % width, goal_idx // width], dtype=jnp.uint32),
            agent_pos=jnp.array([agent_idx % width, agent_idx // width], dtype=jnp.uint32),
            agent_dir=jax.random.randint(rng_dir, (), 0, 4).astype(jnp.uint8),
            width=width,
            height=height,
        )

    return sample


@register_mutator("minimax")
def minimax_mutator(config: Dict[str, Any], env) -> Callable:
    """Upstream ACCEL: `make_level_mutator_minimax(100)`.

    The 100 is the upstream literal (the number of candidate wall positions the
    mutator considers), not a config value - kept as-is for baseline fidelity.
    """
    return make_level_mutator_minimax(100)
