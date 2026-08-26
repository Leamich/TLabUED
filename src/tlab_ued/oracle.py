"""A learnability oracle: a small model that predicts p(solve | level).

The expensive thing in SFL (Rutherford et al., 2024) is not the idea, it is the
*measurement*. Learnability is `p(1-p)` where `p` is the student's success rate
on a level, and the only way `teachers/sfl_accel.py` can get `p` is to play the
level: 224 candidate levels x 4 attempts = 28 rollout-only updates per phase,
11.2% of the whole training budget spent measuring instead of learning. And 224
candidates is a narrow window on a level space of 2^169 wall maps.

This module makes `p` *predictable* instead of measurable. A small convnet reads
the level - the whole level: wall map, goal, agent pose, and optionally the exact
BFS shortest path, none of which the partially-observing student can see - and
outputs a success probability. It is trained online, supervised, on labels the
teacher already collects for free: every rollout any branch does ends in "solved"
or "not solved" for the level it was played on.

What that buys, in rising order of how much trust it needs:

1. **Selection pressure.** Ranking 8192 fresh levels costs one forward pass of a
   50k-parameter net and zero env steps, so the top-k comes from a population 30x
   larger than SFL can afford to play.
2. **A cheaper phase.** The oracle proposes, rollouts dispose: only its top
   candidates are actually played (`--sfl_num_levels`, 64 by default rather than
   224), and the updates that frees go back to gradient updates.
3. **Free re-scoring.** A level's buffer score is only as fresh as the last time
   it was replayed. The oracle can re-score all 4000 buffer entries at every
   phase for the price of one forward pass.
4. **Guided mutation.** ACCEL mutates blind. With an oracle, a parent can emit
   `--oracle_mutation_proposals` children and keep the one predicted closest to a
   coin flip - a hill-climb on learnability that costs no env steps.

The obvious objection is that a model trained on the teacher's own distribution
is being asked about fresh random levels, and that an argmax over a *predicted*
quantity selects partly for prediction error. That is exactly why the default
keeps stage 2's verification rollouts, and why a uniformly random control group
rides inside them: every phase the teacher measures how much better the oracle's
picks were than levels nobody chose (`oracle/selection_gain`). The
`sfl_oracle_noverify` preset drops the verification and is the honest test of
whether the oracle can be trusted alone.

The counter-argument in the oracle's favour is worth stating too, because it is
not obvious: SFL's own `p_hat` is 4 Bernoulli samples, so `p_hat(1-p_hat)` is a
very noisy statistic, and taking the top 32 of 224 by a noisy statistic selects
for luck as much as for learnability. A model that pools information across
thousands of levels can be the *lower*-variance estimator of the two.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax import struct

from tlab_ued.level_diagnostics import solve_levels

# A feature set is an underscore-joined set of tokens. At most one of them picks
# the trunk that reads the map, and any number of them add scalars beside it:
#
#   level   the original trunk: 3x3, then two strided 3x3, then a dense layer
#           over the flattened 3x3 result. It does reach the whole map - but only
#           after two stride-2 steps, so each surviving cell is a blur of a 4x4
#           block and the one-cell gaps that decide whether a corridor connects
#           are averaged away. It learned "how open is this level" and not "how
#           long is the route" (README §7.5).
#   wide    the same idea with the stride removed and depth added, so spatial
#           detail survives to the readout. Tests whether the original trunk
#           failed for want of resolution rather than for want of capacity.
#   prop    k iterations of a single shared 3x3 convolution (the value-iteration
#           trunk of Tamar et al., 2016). Same parameters as one layer, but the
#           computation it can express is "propagate reachability outward", which
#           is the shape of the quantity BFS computes exactly.
#   bfs     four scalars from an exact solve. Privileged: the ceiling, not a
#           method. Never in a submitted arm.
#   policy  two scalars the frozen student produces for free on the start frame.
#           Honest - the teacher may read the student, it may not change it.
#
# The combinations below are the ones with a reason to exist; `parse_features`
# accepts any of them and nothing else.
FEATURE_SETS = (
    "level",
    "level_bfs",
    "bfs",
    "wide",
    "prop",
    "level_policy",
    "wide_policy",
    "prop_policy",
    "wide_bfs",
    "prop_bfs",
)

VISION_TOKENS = ("level", "wide", "prop")

# capped optimal steps, solvable-within-the-episode, reachable fraction, wall density
BFS_SCALARS = 4
# V(s0) and the entropy of pi(s0), both at a zero LSTM carry.
POLICY_SCALARS = 2


def parse_features(features: str) -> frozenset:
    """The token set of a feature name, validated."""
    if features not in FEATURE_SETS:
        raise ValueError(f"oracle_features={features!r}; expected one of {FEATURE_SETS}")
    return frozenset(features.split("_"))


def scalar_width(features: str) -> int:
    """How wide the scalar vector is for this feature set.

    Not a constant any more: `bfs` and `policy` contribute independently, and a
    set with neither still carries a zero-width-free placeholder of 1 so that
    every `OracleState` has the same rank regardless of configuration.
    """
    tokens = parse_features(features)
    width = BFS_SCALARS * ("bfs" in tokens) + POLICY_SCALARS * ("policy" in tokens)
    return max(width, 1)


def encode_planes(levels) -> chex.Array:
    """A level as (n, H, W, 7) float planes: walls, goal, agent, agent direction.

    The four direction planes carry the agent's pose *at the agent's cell* rather
    than as four constant planes, so the whole encoding is translation
    equivariant and a 3x3 convolution sees "agent facing east" as a local
    pattern.
    """
    wall = jnp.asarray(levels.wall_map, dtype=jnp.float32)
    n, h, w = wall.shape
    rows = jnp.arange(n)
    # uint32 in the Level dataclass - cast before it meets an index expression.
    gx, gy = (jnp.asarray(levels.goal_pos[:, i], dtype=jnp.int32) for i in (0, 1))
    ax, ay = (jnp.asarray(levels.agent_pos[:, i], dtype=jnp.int32) for i in (0, 1))
    ad = jnp.asarray(levels.agent_dir, dtype=jnp.int32)

    goal = jnp.zeros((n, h, w), dtype=jnp.float32).at[rows, gy, gx].set(1.0)
    agent = jnp.zeros((n, h, w), dtype=jnp.float32).at[rows, ay, ax].set(1.0)
    facing = jnp.zeros((n, 4, h, w), dtype=jnp.float32).at[rows, ad, ay, ax].set(1.0)
    stacked = jnp.concatenate([wall[:, None], goal[:, None], agent[:, None], facing], axis=1)
    return stacked.transpose(0, 2, 3, 1)


def bfs_features(levels, max_steps_in_episode: int) -> chex.Array:
    """(n, 4) scalars from an exact solve of the level - the privileged features.

    `solve_levels` is the same Bellman-Ford over (dir, y, x) that
    `level_diagnostics` uses to describe a generator, so "optimal steps" here is
    the true minimum number of env steps, and an unsolvable level is identified
    as such rather than guessed at. That matters more than it sounds: this
    generator makes plenty of levels with no path at all, and telling "hard" from
    "impossible" is the whole reason learnability beats a value-loss proxy.
    """
    out = solve_levels(levels)
    steps = out["steps"].astype(jnp.float32)
    cap = float(max_steps_in_episode)
    cells = float(levels.wall_map.shape[-1] * levels.wall_map.shape[-2])
    return jnp.stack(
        [
            jnp.minimum(steps, cap) / cap,
            (steps <= cap).astype(jnp.float32),
            out["reachable_cells"] / jnp.maximum(out["free_cells"], 1),
            out["num_walls"] / cells,
        ],
        axis=-1,
    )


def binomial_nll(logits: chex.Array, attempts: chex.Array, successes: chex.Array) -> chex.Array:
    """Mean per-episode binomial NLL of `successes` out of `attempts`.

    -[s.log(p) + (n-s).log(1-p)] is `n.softplus(z) - s.z` for a logit z, which is
    the numerically safe form. Dividing by n weights each *level* equally however
    many times it was played; rows with n = 0 (empty ring slots online, padding
    offline) contribute nothing.

    Module-level so that `oracle_bench` fits its candidate models against exactly
    the objective the online oracle is trained on - a bench that optimised
    something else would be measuring a different model than the one that would
    run.
    """
    nll = attempts * jax.nn.softplus(logits) - successes * logits
    return (nll / jnp.maximum(attempts, 1.0)).sum() / jnp.maximum((attempts > 0).sum(), 1.0)


def readout(x: chex.Array, planes: chex.Array) -> chex.Array:
    """Pool a (n, H, W, C) feature map down to (n, 4C), keeping *where* it matters.

    Global mean and max alone are translation invariant, which is wrong here: the
    quantity being predicted is a property of the route between two particular
    cells. `encode_planes` puts a one-hot for the goal in plane 1 and one for the
    agent in plane 2, so multiplying by them and summing is a gather at those two
    cells - the same readout a value-iteration network takes at the agent's
    position, written so that `__call__` needs no extra arguments.
    """
    goal, agent = planes[..., 1:2], planes[..., 2:3]
    return jnp.concatenate(
        [
            x.mean(axis=(1, 2)),
            x.max(axis=(1, 2)),
            (x * agent).sum(axis=(1, 2)),
            (x * goal).sum(axis=(1, 2)),
        ],
        axis=-1,
    )


class OracleNet(nn.Module):
    """Level -> logit of the student's success probability.

    Deliberately small (~50k parameters). This model has to *track* a policy that
    is changing under it, so its job is to adapt within a few hundred gradient
    steps, not to be the best possible predictor of a frozen policy.
    """

    features: str = "level_bfs"
    hidden: int = 64
    # Iterations of the shared convolution under `prop`. 16 > 13, so a signal can
    # cross the whole map and come back.
    prop_iters: int = 16

    @nn.compact
    def __call__(self, planes: chex.Array, scalars: chex.Array) -> chex.Array:
        tokens = parse_features(self.features)
        parts = []
        if "level" in tokens:
            x = nn.relu(nn.Conv(16, (3, 3))(planes))
            x = nn.relu(nn.Conv(32, (3, 3), strides=(2, 2))(x))
            x = nn.relu(nn.Conv(32, (3, 3), strides=(2, 2))(x))
            parts.append(x.reshape(x.shape[0], -1))
        if "wide" in tokens:
            # No stride: every layer adds 1 cell of radius, so five of them reach
            # 11 cells in each direction and the final mean covers the rest. The
            # cost of that is width, so the channel counts stay small.
            x = planes
            for _ in range(5):
                x = nn.relu(nn.Conv(16, (3, 3))(x))
            parts.append(readout(x, planes))
        if "prop" in tokens:
            # One shared convolution applied `prop_iters` times, i.e. a value
            # iteration over the map: the parameter count of a single layer with
            # the receptive field of sixteen. The skip from `planes` keeps the
            # walls visible at every iteration rather than only at the first.
            embed = nn.Conv(16, (3, 3), name="prop_in")(planes)
            step = nn.Conv(16, (3, 3), name="prop_step")
            skip = nn.Conv(16, (1, 1), name="prop_skip")
            x = nn.relu(embed)
            for _ in range(self.prop_iters):
                x = nn.relu(step(x) + skip(embed))
            parts.append(readout(x, planes))
        if "bfs" in tokens or "policy" in tokens:
            parts.append(scalars)
        x = nn.relu(nn.Dense(self.hidden)(jnp.concatenate(parts, axis=-1)))
        # Zero-initialised head: an untrained oracle predicts p = 0.5 everywhere,
        # which is a harmless prior rather than a confident wrong one.
        return nn.Dense(1, kernel_init=nn.initializers.zeros)(x).squeeze(-1)


@struct.dataclass
class OracleState:
    """Oracle parameters plus the ring buffer of observations they are fit to.

    Rides inside `TrainState.teacher_state`, never inside `TrainState.params`:
    the graders load a checkpoint and read `loaded["params"]`, which must stay the
    untouched `ActorCritic` tree.

    The ring is the only mechanism keeping the oracle current. `p` is a property
    of the *present* policy, so an observation from 10k updates ago is not stale
    data, it is wrong data; holding only the last `--oracle_buffer_capacity`
    level-observations is what bounds how wrong.
    """

    params: Any
    opt_state: Any
    levels: Any  # (capacity, ...) Level pytree
    scalars: chex.Array  # (capacity, scalar_width(features)), computed at observation time
    attempts: chex.Array  # (capacity,) episodes played on that level
    successes: chex.Array  # (capacity,) of which this many reached the goal
    ptr: chex.Array  # next write position
    count: chex.Array  # observations ever written, saturating at capacity
    loss: chex.Array  # last training loss, for the log


class LearnabilityOracle:
    """Static half of the oracle - the net, the optimiser and the hyperparameters.

    Follows the shape of `jaxued.level_sampler.LevelSampler`: this object holds no
    data, every method takes an `OracleState` and returns a new one.
    """

    def __init__(self, config: Dict[str, Any], max_steps_in_episode: int):
        validate(config)
        self.features = str(config["oracle_features"])
        self.tokens = parse_features(self.features)
        self.use_bfs = "bfs" in self.tokens
        self.scalar_width = scalar_width(self.features)
        self.capacity = int(config["oracle_buffer_capacity"])
        self.batch_size = int(config["oracle_batch_size"])
        self.train_steps = int(config["oracle_train_steps"])
        self.max_steps_in_episode = int(max_steps_in_episode)
        self.net = OracleNet(features=self.features, hidden=int(config["oracle_hidden"]))
        self.tx = optax.adam(float(config["oracle_lr"]))

    # --- features -------------------------------------------------------------
    def features_of(self, levels) -> Tuple[chex.Array, chex.Array]:
        """(planes, scalars) for a batch of levels; BFS runs only when it is used."""
        planes = encode_planes(levels)
        if self.use_bfs:
            scalars = bfs_features(levels, self.max_steps_in_episode)
        else:
            scalars = jnp.zeros((planes.shape[0], self.scalar_width), dtype=jnp.float32)
        return planes, scalars

    # --- state ----------------------------------------------------------------
    def initialize(self, rng: chex.PRNGKey, pholder_level) -> OracleState:
        batch = jax.tree_util.tree_map(lambda x: jnp.asarray(x)[None], pholder_level)
        planes, scalars = self.features_of(batch)
        params = self.net.init(rng, planes, scalars)
        levels = jax.tree_util.tree_map(
            lambda x: jnp.asarray(x)[None].repeat(self.capacity, axis=0), pholder_level
        )
        zeros = jnp.zeros(self.capacity, dtype=jnp.float32)
        return OracleState(
            params=params,
            opt_state=self.tx.init(params),
            levels=levels,
            scalars=jnp.zeros((self.capacity, self.scalar_width), dtype=jnp.float32),
            # attempts = 0 makes an unwritten slot weightless in the loss, so the
            # ring needs no separate validity mask.
            attempts=zeros,
            successes=zeros,
            ptr=jnp.asarray(0, dtype=jnp.int32),
            count=jnp.asarray(0, dtype=jnp.int32),
            loss=jnp.zeros((), dtype=jnp.float32),
        )

    # --- prediction -----------------------------------------------------------
    def predict(self, state: OracleState, levels) -> chex.Array:
        """Predicted success probability for a batch of levels. No env steps."""
        planes, scalars = self.features_of(levels)
        return jax.nn.sigmoid(self.net.apply(state.params, planes, scalars))

    # --- fitting --------------------------------------------------------------
    def observe(
        self, state: OracleState, levels, attempts: chex.Array, successes: chex.Array
    ) -> OracleState:
        """Write a batch of (level, successes out of attempts) into the ring.

        `attempts` is 1 for the replay branch - one episode per level - and
        `--sfl_num_attempts` for the scoring branches, which is why the loss below
        is binomial rather than Bernoulli: a level played four times is four times
        the evidence of one played once.
        """
        size = attempts.shape[0]
        idx = (state.ptr + jnp.arange(size)) % self.capacity
        _, scalars = self.features_of(levels)
        return state.replace(
            levels=jax.tree_util.tree_map(
                lambda buf, x: buf.at[idx].set(x), state.levels, levels
            ),
            scalars=state.scalars.at[idx].set(scalars),
            attempts=state.attempts.at[idx].set(attempts.astype(jnp.float32)),
            successes=state.successes.at[idx].set(successes.astype(jnp.float32)),
            ptr=(state.ptr + size) % self.capacity,
            count=jnp.minimum(state.count + size, self.capacity),
        )

    def loss_fn(self, params, planes, scalars, attempts, successes) -> chex.Array:
        """Mean per-episode binomial NLL over the levels in a minibatch."""
        return binomial_nll(self.net.apply(params, planes, scalars), attempts, successes)

    def train(self, state: OracleState, rng: chex.PRNGKey) -> OracleState:
        """`--oracle_train_steps` Adam steps on minibatches drawn from the ring.

        Runs on every update, in every branch, so the oracle is never more than
        one update behind the policy it is predicting. The cost is a 50k-parameter
        forward/backward over 256 levels against the student's PPO update over
        5 epochs of 32 x 256 LSTM steps - three orders of magnitude apart.
        """
        if self.train_steps == 0:
            return state

        def step(carry, _):
            state, rng = carry
            rng, rng_batch = jax.random.split(rng)
            # Sampling across the whole ring is safe before it fills: unwritten
            # slots carry attempts = 0 and drop out of the loss.
            idx = jax.random.randint(rng_batch, (self.batch_size,), 0, self.capacity)
            levels = jax.tree_util.tree_map(lambda x: x[idx], state.levels)
            loss, grads = jax.value_and_grad(self.loss_fn)(
                state.params,
                encode_planes(levels),
                state.scalars[idx],
                state.attempts[idx],
                state.successes[idx],
            )
            updates, opt_state = self.tx.update(grads, state.opt_state, state.params)
            state = state.replace(
                params=optax.apply_updates(state.params, updates),
                opt_state=opt_state,
                loss=loss,
            )
            return (state, rng), None

        (state, _), _ = jax.lax.scan(step, (state, rng), None, self.train_steps)
        return state


# === measuring the oracle ===


def rank(values: chex.Array) -> chex.Array:
    """0-based ranks. Ties break arbitrarily, which is tolerable at n >= 64."""
    return jnp.argsort(jnp.argsort(values)).astype(jnp.float32)


def correlation(a: chex.Array, b: chex.Array) -> chex.Array:
    a, b = a - a.mean(), b - b.mean()
    return (a * b).mean() / jnp.maximum(a.std() * b.std(), 1e-8)


def calibration(predicted: chex.Array, measured: chex.Array) -> Dict[str, chex.Array]:
    """How well `predicted` matched what the rollouts actually found.

    Computed on the levels a phase verifies, which include a uniformly random
    control group - without those these numbers would only ever describe levels
    the oracle already liked, which is the one population they cannot be trusted
    on.
    """
    return {
        "brier": ((predicted - measured) ** 2).mean(),
        "bias": (predicted - measured).mean(),
        "rank_corr": correlation(rank(predicted), rank(measured)),
    }


def validate(config: Dict[str, Any]) -> None:
    """Fail before the first rollout rather than inside a jitted branch."""
    tokens = parse_features(str(config["oracle_features"]))
    if "policy" in tokens:
        # The policy scalars need the student's parameters, which `features_of`
        # does not have: only `oracle_bench` assembles them, from a checkpoint.
        # Screening a `policy` arm offline is free; putting it on the training
        # path is a separate change, and this stops a run starting as if it had
        # already been made.
        raise ValueError(
            f"oracle_features={config['oracle_features']!r} is a bench-only feature set: "
            "the policy scalars are not plumbed into the teacher. Screen it with "
            "`python -m tlab_ued.oracle_bench bench` first."
        )
    if not tokens & set(VISION_TOKENS) and "bfs" not in tokens:
        raise ValueError(f"oracle_features={config['oracle_features']!r} would see nothing")
    for key in ("oracle_buffer_capacity", "oracle_batch_size", "oracle_hidden"):
        if int(config[key]) <= 0:
            raise ValueError(f"{key} must be positive, got {config[key]}")
    if int(config["oracle_train_steps"]) < 0:
        raise ValueError("oracle_train_steps cannot be negative")
