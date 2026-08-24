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
which is the starting point for a better score:
  - "learnability": p(1-p) over the level's success rate, the SFL score from that
    paper. It needs an estimate of p, which no single rollout provides, so it is
    only available with a teacher that measures one (`sfl_accel`).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import chex
import jax.numpy as jnp
from flax import struct

from jaxued.utils import max_mc, positive_value_loss


@struct.dataclass
class RolloutSignals:
    """Everything a score function may look at, for one batch of levels.

    Shapes: (num_steps, num_train_envs) for trajectory arrays, (num_train_envs,)
    for per-level arrays. `levels` is the batch of levels that produced the
    rollout; `extras` is a free-form dict for teacher-specific signals.

    Every field defaults to None: a score built on a teacher-supplied signal
    (see `learnability`) may be evaluated where no trajectory exists - the SFL
    teacher scores a level population accumulated over several rollouts, not the
    one batch in front of it.
    """

    dones: Optional[chex.Array] = None
    values: Optional[chex.Array] = None
    rewards: Optional[chex.Array] = None
    advantages: Optional[chex.Array] = None
    targets: Optional[chex.Array] = None
    log_probs: Optional[chex.Array] = None
    actions: Optional[chex.Array] = None
    max_returns: Optional[chex.Array] = None
    levels: Any = None
    extras: Optional[Dict[str, Any]] = None


ScoreFn = Callable[[Dict[str, Any], RolloutSignals], chex.Array]

SCORE_FUNCTIONS: Dict[str, ScoreFn] = {}


def register_score_fn(name: str, requires_extras: tuple = ()) -> Callable[[ScoreFn], ScoreFn]:
    """Decorator registering a score function under `--score_function <name>`.

    `requires_extras` names `RolloutSignals.extras` keys the score cannot be
    computed without. It is advisory - the function still checks - but it lets
    callers (and the tests) tell a score that works with any teacher from one
    that only works with the teacher that produces its signal.
    """

    def decorator(fn: ScoreFn) -> ScoreFn:
        if name in SCORE_FUNCTIONS:
            raise ValueError(f"score function {name!r} is already registered")
        fn.requires_extras = tuple(requires_extras)
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


# === Learnability (Rutherford et al., 2024) ===


def first_episode_success(dones: chex.Array, rewards: chex.Array) -> chex.Array:
    """Did the *first* episode of each rollout reach the goal? -> (num_envs,) in {0,1}.

    Only the first episode counts. Under `AutoReplayWrapper` an env replays the
    same level until the rollout ends, so an easy level yields several episodes
    and a hard one yields exactly one (the 250-step timeout); counting them all
    would weight easy levels by how fast they are solved. The first episode also
    starts from a zero LSTM state, which is how the held-out evaluation runs.

    In this maze the only non-zero reward is the terminal one for stepping into
    the goal, so "any positive reward" is exactly "solved".
    """
    prior_dones = jnp.cumsum(dones.astype(jnp.int32), axis=0) - dones.astype(jnp.int32)
    in_first_episode = prior_dones == 0
    return ((rewards * in_first_episode) > 0).any(axis=0).astype(jnp.float32)


def learnability(success_rate: chex.Array) -> chex.Array:
    """`p(1-p)`: the variance of a Bernoulli trial on the level.

    Maximal at p = 0.5 - the student sometimes solves the level and sometimes
    does not, which is where a gradient step has something to move. Levels it
    always solves (p=1) and levels it never solves (p=0) both score 0, which is
    the property MaxMC and positive value loss lack: an impossible level, of
    which this generator makes plenty, has a *high* value loss and a low
    learnability.
    """
    return success_rate * (1.0 - success_rate)


@register_score_fn("learnability", requires_extras=("success_rate",))
def learnability_score(config: Dict[str, Any], signals: RolloutSignals) -> chex.Array:
    """`p(1-p)` over a teacher-supplied success rate.

    `p` cannot be estimated from a single rollout - one episode per level gives
    `p` in {0,1} and a score of exactly 0 either way - so the teacher owns the
    estimate and passes it in `extras["success_rate"]`. `SFLACCELTeacher` does
    this two ways: several attempts at the same level within one update, and a
    running estimate carried per buffer level across replays.
    """
    extras = signals.extras or {}
    if "success_rate" not in extras:
        raise ValueError(
            "score_function 'learnability' needs extras['success_rate'], which the active "
            "teacher does not provide. Use --teacher sfl_accel, or pick a score function "
            "that reads the rollout directly (MaxMC, pvl)."
        )
    return learnability(extras["success_rate"])
