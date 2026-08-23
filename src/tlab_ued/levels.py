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

from jaxued.environments.maze import make_level_generator, make_level_mutator_minimax

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


@register_mutator("minimax")
def minimax_mutator(config: Dict[str, Any], env) -> Callable:
    """Upstream ACCEL: `make_level_mutator_minimax(100)`.

    The 100 is the upstream literal (the number of candidate wall positions the
    mutator considers), not a config value - kept as-is for baseline fidelity.
    """
    return make_level_mutator_minimax(100)
