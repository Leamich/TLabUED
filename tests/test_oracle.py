"""Tests for the learnability oracle itself, with no teacher around it.

The teacher's tests (tests/test_sfl_oracle.py) can only check that the wiring
holds together. These check what the whole method rests on: that the encoding
says what it claims about a level, that the ring forgets in the right order, and
that the model can fit a level property from the kind of labels the teacher
feeds it.
"""

from __future__ import annotations

import pytest

pytest.importorskip("jax", reason="needs the full JAX stack")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from jaxued.environments.maze import Level, make_level_generator  # noqa: E402

from tlab_ued.config import make_config  # noqa: E402
from tlab_ued.oracle import (  # noqa: E402
    LearnabilityOracle,
    bfs_features,
    calibration,
    encode_planes,
    validate,
)

# Plane indices of `encode_planes`: wall, goal, agent, then four facing planes.
WALL, GOAL, AGENT = 0, 1, 2

SOLVABLE = """
#####
#>..#
#.#.#
#..G#
#####
"""

# The goal's only approach cell is itself cut off, so the goal is unreachable and
# one free cell is stranded - both of which the BFS features are meant to see.
SEALED = """
#####
#>#G#
#.#.#
#..##
#####
"""


def stack(*levels):
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *levels)


def random_levels(n: int, seed: int = 0):
    generate = make_level_generator(13, 13, 25)
    return jax.vmap(generate)(jax.random.split(jax.random.PRNGKey(seed), n))


def first(levels):
    return jax.tree_util.tree_map(lambda x: x[0], levels)


def oracle_config(**overrides):
    """A config small enough, and a learning rate fast enough, for a unit test.

    `overrides` replace these defaults rather than colliding with them.
    """
    defaults = dict(
        oracle_buffer_capacity=64,
        oracle_batch_size=32,
        oracle_hidden=32,
        oracle_lr=1e-2,
    )
    return make_config(preset="sfl_oracle", **{**defaults, **overrides})


# === the encoding ===


def test_encode_planes_marks_the_agent_the_goal_and_the_facing():
    planes = encode_planes(stack(Level.from_str(SOLVABLE)))[0]

    assert planes.shape == (5, 5, 7)
    # `>` puts the agent at (x=1, y=1) facing direction 0; G is at (x=3, y=3).
    assert planes[1, 1, AGENT] == 1.0 and planes[3, 3, GOAL] == 1.0
    assert planes[..., AGENT].sum() == 1.0 and planes[..., GOAL].sum() == 1.0
    # The facing is carried at the agent's cell rather than as a constant plane,
    # which is what keeps the encoding translation equivariant.
    facing = planes[..., 3:]
    assert facing[1, 1, 0] == 1.0 and facing.sum() == 1.0
    assert planes[0, :, WALL].sum() == 5.0  # the border


def test_bfs_features_tell_impossible_from_merely_hard():
    levels = stack(Level.from_str(SOLVABLE), Level.from_str(SEALED))
    steps, solvable, reachable, walls = bfs_features(levels, 250).T

    assert solvable.tolist() == [1.0, 0.0]
    # An unreachable goal saturates the capped step count rather than blowing up.
    assert float(steps[0]) < float(steps[1]) == pytest.approx(1.0)
    # And it strands a free cell, which the reachable fraction picks up.
    assert float(reachable[0]) == pytest.approx(1.0)
    assert float(reachable[1]) < 1.0
    assert float(walls[0]) < float(walls[1])


# === the ring ===


def test_an_untrained_oracle_predicts_a_coin_flip():
    """The head is zero-initialised, so an unfit oracle is uninformative rather
    than confidently wrong. That is what makes the warm-up period harmless."""
    oracle = LearnabilityOracle(oracle_config(), 250)
    levels = random_levels(8)
    state = oracle.initialize(jax.random.PRNGKey(0), first(levels))

    assert jnp.allclose(oracle.predict(state, levels), 0.5)


def test_the_ring_forgets_the_oldest_observation_first():
    oracle = LearnabilityOracle(oracle_config(oracle_buffer_capacity=4), 250)
    levels = random_levels(6)
    state = oracle.initialize(jax.random.PRNGKey(0), first(levels))

    for i in range(6):
        one = jax.tree_util.tree_map(lambda x, i=i: x[i : i + 1], levels)
        state = oracle.observe(state, one, jnp.ones(1), jnp.full(1, float(i)))

    assert int(state.count) == 4  # saturates at capacity
    assert int(state.ptr) == 2  # and wrapped twice
    # Observations 0 and 1 were overwritten by 4 and 5; 2 and 3 survive.
    assert sorted(state.successes.tolist()) == [2.0, 3.0, 4.0, 5.0]


def test_empty_ring_slots_do_not_drag_the_prediction_down():
    """Unwritten slots carry attempts = 0, which is why the ring needs no mask.

    Four solved levels in a 64-slot ring: if the 60 empty slots counted as
    failures the prediction could never leave 0.5.
    """
    oracle = LearnabilityOracle(oracle_config(oracle_train_steps=200), 250)
    levels = random_levels(4)
    state = oracle.initialize(jax.random.PRNGKey(0), first(levels))
    state = oracle.observe(state, levels, jnp.full(4, 4.0), jnp.full(4, 4.0))
    state = oracle.train(state, jax.random.PRNGKey(1))

    assert float(oracle.predict(state, levels).mean()) > 0.7


# === fitting ===


def test_the_oracle_learns_a_property_it_can_see():
    """Can it fit at all? Label the half of a batch whose optimal path is longer.

    This is not a claim about predicting a real policy - `oracle/rank_corr` in a
    run measures that. It is the weaker precondition: given labels that *are* a
    function of the level, this net and this optimiser find the function.
    """
    oracle = LearnabilityOracle(oracle_config(oracle_train_steps=300), 250)
    levels = random_levels(64, seed=3)
    # An exact 32/32 split, whatever the generator's difficulty mix happens to be.
    order = jnp.argsort(bfs_features(levels, 250)[:, 0])
    hard = jnp.zeros(64, dtype=jnp.float32).at[order[32:]].set(1.0)

    state = oracle.initialize(jax.random.PRNGKey(0), first(levels))
    state = oracle.observe(state, levels, jnp.ones(64), hard)
    state = oracle.train(state, jax.random.PRNGKey(1))

    predicted = oracle.predict(state, levels)
    assert float(predicted[hard == 1].mean()) > float(predicted[hard == 0].mean()) + 0.2


def test_a_level_always_failed_is_predicted_lower_than_one_always_solved():
    oracle = LearnabilityOracle(
        oracle_config(oracle_buffer_capacity=2, oracle_batch_size=2, oracle_train_steps=200), 250
    )
    levels = random_levels(2, seed=5)
    state = oracle.initialize(jax.random.PRNGKey(0), first(levels))
    state = oracle.observe(state, levels, jnp.array([8.0, 8.0]), jnp.array([0.0, 8.0]))
    state = oracle.train(state, jax.random.PRNGKey(1))

    predicted = oracle.predict(state, levels)
    assert float(predicted[0]) < 0.3 < 0.7 < float(predicted[1])


# === measuring the oracle ===


def test_calibration_scores_a_perfect_and_a_reversed_prediction():
    measured = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])
    perfect = calibration(measured, measured)
    assert float(perfect["brier"]) == pytest.approx(0.0)
    assert float(perfect["rank_corr"]) == pytest.approx(1.0, abs=1e-5)

    backwards = calibration(measured[::-1], measured)
    assert float(backwards["rank_corr"]) == pytest.approx(-1.0, abs=1e-5)
    # Right on average, wrong on every level: no bias at all, and a bad Brier
    # score. Which is why the run logs both.
    assert float(backwards["bias"]) == pytest.approx(0.0, abs=1e-6)
    assert float(backwards["brier"]) > 0.1


@pytest.mark.parametrize(
    "overrides",
    [
        {"oracle_features": "nonsense"},
        {"oracle_buffer_capacity": 0},
        {"oracle_batch_size": -1},
        {"oracle_train_steps": -1},
    ],
)
def test_bad_oracle_settings_are_refused(overrides):
    with pytest.raises(ValueError):
        validate({**make_config(preset="sfl_oracle"), **overrides})
