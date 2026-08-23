"""Score-function registry.

The score function is the part of a UED teacher that decides *which levels look
worth training on*. It is the thing docs/TASK.md asks us to improve, so it is a
plugin point: add a function, decorate it with `@register_score_fn("name")`, and
it becomes available as `--score_function name`.

Every score function receives the same rich `RolloutSignals` bundle rather than a
fixed argument list, so a new idea can use signals (observations, actions,
logits, levels, per-env returns) that the two upstream scores ignore, without
touching a single call site in `train.py`.

The two built-ins wrap JaxUED's own implementations verbatim, so the baselines
stay bit-for-bit comparable with upstream:
  - "MaxMC": max Monte-Carlo regret proxy (Jiang et al., 2021)
  - "pvl":   positive value loss (Jiang et al., 2021)
Both are analysed - and criticised - in Rutherford et al., 2024 (arXiv:2408.15099),
which is the starting point for a better score.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import chex
from flax import struct

from jaxued.utils import max_mc, positive_value_loss


@struct.dataclass
class RolloutSignals:
    """Everything a score function may look at, for one batch of levels.

    Shapes: (num_steps, num_train_envs) for trajectory arrays, (num_train_envs,)
    for per-level arrays. `levels` is the batch of levels that produced the
    rollout; `extras` is a free-form dict for teacher-specific signals.
    """

    dones: chex.Array
    values: chex.Array
    rewards: chex.Array
    advantages: chex.Array
    targets: chex.Array
    log_probs: chex.Array
    actions: chex.Array
    max_returns: chex.Array
    levels: Any = None
    extras: Optional[Dict[str, Any]] = None


ScoreFn = Callable[[Dict[str, Any], RolloutSignals], chex.Array]

SCORE_FUNCTIONS: Dict[str, ScoreFn] = {}


def register_score_fn(name: str) -> Callable[[ScoreFn], ScoreFn]:
    """Decorator registering a score function under `--score_function <name>`."""

    def decorator(fn: ScoreFn) -> ScoreFn:
        if name in SCORE_FUNCTIONS:
            raise ValueError(f"score function {name!r} is already registered")
        SCORE_FUNCTIONS[name] = fn
        return fn

    return decorator


def get_score_fn(config: Dict[str, Any]) -> ScoreFn:
    name = config["score_function"]
    if name not in SCORE_FUNCTIONS:
        raise ValueError(
            f"Unknown score function {name!r}. Registered: {sorted(SCORE_FUNCTIONS)}"
        )
    return SCORE_FUNCTIONS[name]


def compute_score(config: Dict[str, Any], signals: RolloutSignals) -> chex.Array:
    """Score one batch of levels. Returns shape (num_train_envs,)."""
    return get_score_fn(config)(config, signals)


# === Built-ins: the upstream baselines ===


@register_score_fn("MaxMC")
def max_mc_score(config: Dict[str, Any], signals: RolloutSignals) -> chex.Array:
    """Maximum Monte-Carlo regret proxy - upstream default."""
    return max_mc(signals.dones, signals.values, signals.max_returns)


@register_score_fn("pvl")
def positive_value_loss_score(config: Dict[str, Any], signals: RolloutSignals) -> chex.Array:
    """Positive value loss - the other upstream option."""
    return positive_value_loss(signals.dones, signals.advantages)
