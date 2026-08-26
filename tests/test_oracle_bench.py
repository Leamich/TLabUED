"""Tests for the offline oracle bench and the honest feature sets.

The bench exists to make a decision - which `--oracle_features` arm is worth a
2.5-hour run - so what has to be tested is that its numbers mean what the
decision assumes. Three things: the population statistics are computed from
measurements and not from a model, the selection score rewards ranking rather
than calibration, and the preregistered pick rule cannot quietly choose an arm
the teacher is unable to run.

The GPU-shaped half (`collect`, `staleness`) is exercised end-to-end on a tiny
synthetic checkpoint elsewhere; here the concern is the arithmetic.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax", reason="needs the full JAX stack")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from jaxued.environments import Maze  # noqa: E402

from tlab_ued.levels import get_generator  # noqa: E402
from tlab_ued.level_diagnostics import UNREACHABLE, solve_levels  # noqa: E402
from tlab_ued.oracle import (  # noqa: E402
    FEATURE_SETS,
    OracleNet,
    parse_features,
    scalar_width,
    validate,
)
from tlab_ued.oracle_bench import (  # noqa: E402
    BENCH_FEATURES,
    auc_learnable,
    pick_arm,
    population_stats,
    rank_corr,
    scalars_for,
    score_predictions,
)


def maze_env():
    return Maze(max_height=13, max_width=13, agent_view_size=5, normalize_obs=True)


# === the feature sets ===


@pytest.mark.parametrize("features", FEATURE_SETS)
def test_every_feature_set_builds_and_takes_the_width_it_declares(features):
    """`scalar_width` is the contract between the model and whoever feeds it."""
    net = OracleNet(features=features, hidden=16)
    planes = jnp.zeros((3, 13, 13, 7))
    scalars = jnp.zeros((3, scalar_width(features)))
    params = net.init(jax.random.PRNGKey(0), planes, scalars)
    assert net.apply(params, planes, scalars).shape == (3,)


def test_every_trunk_can_at_least_see_the_whole_map():
    """A wall in the far corner must be able to change the prediction.

    A precondition, not the hypothesis: `level` passes this too, because its
    dense layer spans the whole downsampled map. What `level` loses is
    *resolution* - after two stride-2 convolutions a one-cell gap that decides
    whether a corridor connects has been averaged into its neighbours - and that
    is what README §7.5 is about. A trunk that failed even this test would be
    broken rather than merely blurry.
    """
    for features in ("level", "wide", "prop"):
        net = OracleNet(features=features, hidden=16)
        planes = jnp.zeros((1, 13, 13, 7)).at[0, 6, 6, 2].set(1.0).at[0, 0, 0, 1].set(1.0)
        scalars = jnp.zeros((1, scalar_width(features)))
        params = net.init(jax.random.PRNGKey(0), planes, scalars)
        # The head is zero-initialised on purpose (an untrained oracle predicts
        # 0.5 everywhere), so every net answers 0 until it is trained. Reading
        # the trunk means giving the head a non-zero kernel first.
        params = jax.tree_util.tree_map(lambda x: x, params)
        params["params"]["Dense_1"]["kernel"] = jnp.ones_like(
            params["params"]["Dense_1"]["kernel"]
        )

        far_corner = planes.at[0, 12, 12, 0].set(1.0)
        base = float(net.apply(params, planes, scalars)[0])
        moved = float(net.apply(params, far_corner, scalars)[0])
        assert abs(moved - base) > 1e-6, f"{features} cannot see the far corner"


def test_policy_features_are_rejected_on_the_training_path():
    """They have no provider inside the teacher, so a run must not start.

    Screening `policy` offline is free; putting it on the training path is a
    change nobody has made yet. Failing loudly here is what stops a preset from
    silently training against zeros.
    """
    from tlab_ued.config import make_config

    # It has to fail while the config is being built, not at the first update:
    # by then a sweep slot has already been spent.
    with pytest.raises(ValueError, match="bench-only"):
        make_config(teacher="sfl_oracle", oracle_features="wide_policy")


# === population statistics ===


def test_population_stats_read_the_headroom_off_the_measurements():
    """Ceiling over floor is the whole selection problem, and it needs no model."""
    # 96 hopeless levels and 4 coin flips: a perfect ranker gets 0.25, chance
    # gets 0.01, so the headroom is exactly 25x.
    p = np.concatenate([np.zeros(96), np.full(4, 0.5)])
    stats = population_stats(p, top_k=4)
    assert stats["ceiling"] == pytest.approx(0.25)
    assert stats["floor"] == pytest.approx(0.01)
    assert stats["headroom"] == pytest.approx(25.0)
    assert stats["frac_learnable"] == pytest.approx(0.04)


def test_a_degenerate_population_has_no_headroom():
    """When every level is solved, selection cannot help - and must not look like it.

    This is the late-training regime of README §7.2, and the reason the bench
    reports the population before it reports any model: a `share_of_ceiling` of
    1.0 on a population with headroom 1.0 is worth nothing.
    """
    stats = population_stats(np.ones(1000), top_k=48)
    assert stats["ceiling"] == pytest.approx(0.0)
    assert stats["headroom"] == pytest.approx(1.0, abs=0.01)


# === scoring a prediction ===


def test_selection_score_rewards_the_ranking_not_the_calibration():
    """A shifted prediction ranks identically, so it must select identically.

    The Brier score is the one that should notice the shift. Keeping both is the
    point: README §7.4 found an excellent Brier next to a useless ranking, and
    only the pair tells those apart.
    """
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 400)
    idx = np.arange(400)

    honest = score_predictions(p, p, idx, top_k=48)
    shifted = score_predictions(np.clip(p * 0.5 + 0.25, 0, 1), p, idx, top_k=48)

    assert honest["share_of_ceiling"] == pytest.approx(1.0)
    assert shifted["share_of_ceiling"] == pytest.approx(honest["share_of_ceiling"])
    assert shifted["brier"] > honest["brier"] + 0.01


def test_a_useless_prediction_scores_near_the_floor():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 2000)
    noise = rng.uniform(0, 1, 2000)
    scored = score_predictions(noise, p, np.arange(2000), top_k=48)
    assert 0.5 < scored["gain"] < 2.0
    assert abs(scored["rank_corr"]) < 0.15


def test_rank_corr_and_auc_agree_with_their_definitions():
    values = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    assert rank_corr(values, values) == pytest.approx(1.0)
    assert rank_corr(values, values[::-1]) == pytest.approx(-1.0)

    # Learnability peaks in the middle, so ranking by it puts the one learnable
    # level (p = 0.5) on top: a perfect AUC.
    p = np.array([0.0, 0.02, 0.5, 0.98, 1.0])
    assert auc_learnable(p * (1 - p), p) == pytest.approx(1.0)


def test_scalars_for_matches_the_width_the_net_expects():
    data = {
        "attempts": np.ones(5),
        "bfs": np.zeros((5, 4), dtype=np.float32),
        "policy": np.zeros((5, 2), dtype=np.float32),
    }
    for features in BENCH_FEATURES:
        assert scalars_for(features, data).shape == (5, scalar_width(features))


# === the preregistered pick ===


def _report(**shares):
    return {
        "results": [
            {
                "step": 17,
                "features": {k: {"share_of_ceiling": v} for k, v in shares.items()},
            }
        ]
    }


def test_pick_never_returns_an_arm_the_teacher_cannot_run():
    """`bfs` is the ceiling and `policy` is bench-only; neither may be picked.

    Without this the rule would happily choose the privileged arm, which is
    exactly the cheat the whole exercise is set up to avoid.
    """
    pick = pick_arm(_report(level=0.1, wide=0.2, prop=0.3, bfs=0.99, level_bfs=0.99,
                            wide_policy=0.98))
    assert pick["arm"] == "prop"
    assert "bfs" not in pick["means"] and "wide_policy" not in pick["means"]
    validate({"oracle_features": pick["arm"], "oracle_buffer_capacity": 1,
              "oracle_batch_size": 1, "oracle_hidden": 1, "oracle_train_steps": 0})


def test_pick_breaks_near_ties_towards_the_simpler_model():
    """Within the margin the two are not distinguishable, so cost decides."""
    pick = pick_arm(_report(level=0.30, wide=0.40, prop=0.41))
    assert pick["arm"] == "wide"
    assert set(pick["contenders"]) == {"wide", "prop"}

    clear = pick_arm(_report(level=0.30, wide=0.40, prop=0.60))
    assert clear["arm"] == "prop"


# === the validation generator ===


def test_perfect_maze_levels_are_solvable_and_much_longer_than_the_training_ones():
    """The point of the val set: routes an order of magnitude longer, all solvable.

    `minigrid_walls` has a median optimal route of 11 steps and makes plenty of
    unsolvable levels (README §3.1). A perfect maze has exactly one route between
    any two cells, so every level is solvable and the route has to wind.
    """
    env = maze_env()
    val = jax.vmap(get_generator({"level_generator": "perfect_maze"}, env))(
        jax.random.split(jax.random.PRNGKey(0), 64)
    )
    train = jax.vmap(get_generator({"level_generator": "minigrid_walls", "n_walls": 25}, env))(
        jax.random.split(jax.random.PRNGKey(0), 64)
    )

    val_steps = np.asarray(solve_levels(val)["steps"])
    train_steps = np.asarray(solve_levels(train)["steps"])

    assert (val_steps < UNREACHABLE).all(), "a perfect maze is always solvable"
    assert np.median(val_steps) > 2 * np.median(train_steps[train_steps < UNREACHABLE])


def test_perfect_maze_is_a_maze_and_not_a_field_of_walls():
    """Cells on the even lattice are free and (odd, odd) is always wall.

    That is the convention the hand-drawn prefabs use - `StandardMaze` is exactly
    this shape - and it is what makes the val set structurally comparable to the
    levels we are finally scored on.
    """
    env = maze_env()
    levels = jax.vmap(get_generator({"level_generator": "perfect_maze"}, env))(
        jax.random.split(jax.random.PRNGKey(1), 32)
    )
    walls = np.asarray(levels.wall_map)

    assert not walls[:, ::2, ::2].any(), "cell centres must be free"
    assert walls[:, 1::2, 1::2].all(), "corners between cells are always wall"
    # A spanning tree over 49 cells knocks out 48 of the 84 candidate walls.
    assert (walls.reshape(32, -1).sum(axis=1) == 169 - 49 - 48).all()
