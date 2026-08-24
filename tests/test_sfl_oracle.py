"""Tests for the SFL-ORACLE teacher.

Three kinds. The budget arithmetic guards the claim every comparison rests on.
The `phase_stats` test guards the number the report will lead with - if
`selection_gain` is computed wrongly, a failed method looks like a working one.
The integration runs a handful of updates on CPU through every branch, including
the no-verify variant, which is where a mismatched pytree or a level batch that
changes shape between updates would otherwise wait for an hour of GPU time.
"""

from __future__ import annotations

import pytest

pytest.importorskip("jax", reason="needs the full JAX stack")

import jax.numpy as jnp  # noqa: E402

from tlab_ued.config import PRESETS, default_run_name, make_config  # noqa: E402
from tlab_ued.teachers.sfl_accel import sfl_budget_report  # noqa: E402
from tlab_ued.teachers.sfl_oracle import (  # noqa: E402
    oracle_budget_report,
    phase_length,
    validate,
)


# === the budget ===


def test_the_cascade_shortens_the_phase_and_hands_back_the_updates():
    accel = sfl_budget_report(make_config(preset="accel"))
    sfl = sfl_budget_report(make_config(preset="sfl_accel"))
    oracle = oracle_budget_report(make_config(preset="sfl_oracle"))

    # The frozen quantity, identical for all three: 30000 updates of 32 x 256.
    assert oracle["env_steps"] == accel["env_steps"] == sfl["env_steps"]
    # SFL plays 224 levels per phase, the cascade plays 64 - a quarter of the
    # measurement cost, because the ranking that picked those 64 was free.
    assert sfl["reserved_updates"] == 120 * 28
    assert oracle["reserved_updates"] == 120 * 8
    # Which is not a saving that disappears: it becomes gradient updates.
    assert oracle["expected_gradient_updates"] > sfl["expected_gradient_updates"]
    assert oracle["expected_gradient_updates"] > accel["expected_gradient_updates"]


def test_the_cheap_sfl_control_is_budget_identical_to_the_oracle():
    """`sfl_accel_cheap` exists to separate "selects better" from "trains more"."""
    oracle = oracle_budget_report(make_config(preset="sfl_oracle"))
    control = sfl_budget_report(make_config(preset="sfl_accel_cheap"))

    for key in ("env_steps", "reserved_updates", "expected_gradient_updates"):
        assert oracle[key] == control[key], key


def test_phase_length_is_whole_updates_and_teacher_specific():
    config = make_config(preset="sfl_oracle")
    assert phase_length(config) == 8
    assert phase_length(config) * config["num_train_envs"] == (
        config["sfl_num_levels"] * config["sfl_num_attempts"]
    )
    # The `sfl_*` and `oracle_*` defaults are in every config, so these helpers
    # have to ask who is driving before believing them.
    assert phase_length(make_config(preset="accel")) == 0
    assert phase_length(make_config(preset="sfl_accel")) == 0


def test_an_unverified_phase_is_one_update_and_reserves_nothing():
    """It inserts on prediction and then trains, so it is not withheld."""
    config = make_config(preset="sfl_oracle_noverify")
    assert phase_length(config) == 1
    assert oracle_budget_report(config)["reserved_updates"] == 0


@pytest.mark.parametrize(
    "overrides",
    [
        {"oracle_features": "steps"},  # not a feature set
        {"oracle_num_proposals": 8},  # fewer proposals than the shortlist keeps
        {"oracle_control_levels": 64},  # no room left for the oracle's own picks
        {"oracle_mutation_proposals": 0},  # 1 is unguided; 0 is nothing
        {"sfl_num_levels": 65},  # phase would not fit whole updates
        {"sfl_topk": 128},  # keeping more than were verified
    ],
)
def test_bad_sizing_is_refused_before_the_run(overrides):
    with pytest.raises(ValueError):
        validate({**make_config(preset="sfl_oracle"), **overrides})


def test_config_validation_runs_at_build_time():
    with pytest.raises(ValueError):
        make_config(preset="sfl_oracle", oracle_num_proposals=8)


def test_every_preset_has_its_own_run_directory():
    """Two presets sharing a name share a directory, and the sweep runner would
    report the second one finished before it started."""
    names = [default_run_name(make_config(preset=preset)) for preset in PRESETS]
    assert len(set(names)) == len(names), sorted(names)


# === the number the report leads with ===


def test_selection_gain_compares_the_picks_against_the_controls(tmp_path):
    from tlab_ued.train import build

    config = tiny_config(out_dir=str(tmp_path), oracle_control_levels=2, sfl_num_levels=4)
    _, teacher = build(config)

    # Four verified levels: the first two are the random controls (never solved,
    # so learnability 0), the last two are the oracle's picks at a coin flip.
    measured = jnp.array([0.0, 0.0, 0.5, 0.5])
    stats = teacher.phase_stats(jnp.array([0.5, 0.5, 0.5, 0.5]), measured)

    assert float(stats["control_learnability"]) == pytest.approx(0.0)
    assert float(stats["selected_learnability"]) == pytest.approx(0.25)
    # A control group that scores exactly zero makes the ratio undefined, and
    # late in a real run whole control groups do. The floor keeps that phase from
    # logging 10^5 and wrecking every average taken over the column.
    assert float(stats["selection_gain"]) == pytest.approx(0.25 / teacher.gain_floor)
    assert float(stats["selection_gain"]) < 1e3
    # The oracle said 0.5 for all four and was wrong about the controls.
    assert float(stats["brier"]) == pytest.approx(0.125)


def test_selection_gain_is_one_when_the_oracle_picks_no_better_than_chance(tmp_path):
    from tlab_ued.train import build

    config = tiny_config(out_dir=str(tmp_path), oracle_control_levels=2, sfl_num_levels=4)
    _, teacher = build(config)

    stats = teacher.phase_stats(jnp.full(4, 0.5), jnp.array([0.5, 0.5, 0.5, 0.5]))
    assert float(stats["selection_gain"]) == pytest.approx(1.0)


# === the run ===


def tiny_config(preset: str = "sfl_oracle", **overrides):
    """Small enough for CPU, sized so every branch runs.

    4 envs and 2 attempts is 2 levels per update; a 4-level shortlist is then a
    2-update phase, and with a period of 4 the other 2 updates go to the ACCEL
    branches. `overrides` replace these defaults rather than colliding with them.
    """
    defaults = dict(
        num_updates=8,
        eval_freq=4,
        eval_num_attempts=1,
        num_train_envs=4,
        num_steps=8,
        level_buffer_capacity=16,
        minimum_fill_ratio=0.0,
        checkpoint_save_interval=0,
        log_media="none",
        allow_student_changes=True,
        sfl_num_attempts=2,
        sfl_num_levels=4,
        sfl_topk=2,
        sfl_period=4,
        oracle_num_proposals=8,
        oracle_control_levels=1,
        oracle_mutation_proposals=2,
        oracle_buffer_capacity=16,
        oracle_batch_size=8,
        oracle_train_steps=1,
        oracle_warmup_updates=0,
        oracle_hidden=8,
    )
    return make_config(preset=preset, **{**defaults, **overrides})


def test_every_branch_runs_and_the_state_keeps_its_shape(tmp_path):
    from tlab_ued.train import main

    config = tiny_config(out_dir=str(tmp_path))
    train_state = main(config)
    ts = train_state.teacher_state

    assert int(train_state.update_count) == config["num_updates"]
    # 2 phase updates per period of 4, over 8 updates - and one insertion at the
    # end of each phase.
    assert int(ts["num_sfl_updates"]) == 4
    assert int(ts["num_oracle_inserts"]) == 2
    assert int(ts["num_replay_updates"]) > 0
    assert int(ts["num_mutation_updates"]) > 0
    # The oracle saw every rollout: 8 updates, at least one level each.
    assert int(ts["oracle"].count) >= 8
    assert ts["sfl_pred"].shape == (config["sfl_num_levels"],)


def test_branch_counts_still_add_up_to_the_updates(tmp_path):
    """Insertions are counted separately precisely so that this stays true."""
    from tlab_ued.train import main

    ts = main(tiny_config(out_dir=str(tmp_path))).teacher_state
    counted = sum(
        int(ts[key])
        for key in (
            "num_dr_updates",
            "num_replay_updates",
            "num_mutation_updates",
            "num_sfl_updates",
        )
    )
    assert counted == 8


def test_the_unverified_variant_inserts_without_spending_an_update(tmp_path):
    from tlab_ued.train import main

    config = tiny_config("sfl_oracle_noverify", out_dir=str(tmp_path))
    ts = main(config).teacher_state

    # The phase is one update long and that update is a replay, so no update is
    # withheld from the student...
    assert int(ts["num_sfl_updates"]) == 0
    # ...but the insertions happened: updates 0 and 4.
    assert int(ts["num_oracle_inserts"]) == 2
    assert int(ts["num_replay_updates"]) >= 2
    assert int(ts["sampler"]["size"]) >= config["sfl_topk"]


def test_oracle_diagnostics_reach_the_csv(tmp_path):
    import csv

    from tlab_ued.logging_utils import run_dir
    from tlab_ued.train import main

    config = tiny_config(out_dir=str(tmp_path))
    main(config)
    with open(f"{run_dir(config)}/metrics.csv") as f:
        rows = list(csv.DictReader(f))

    assert rows, "no metrics were written"
    for column in (
        "oracle/selection_gain",
        "oracle/rank_corr",
        "oracle/brier",
        "oracle/loss",
        "oracle/buffer_mean_p",
        "branch/num_oracle_inserts",
    ):
        assert column in rows[0], sorted(rows[0])
    assert 0.0 <= float(rows[-1]["oracle/buffer_mean_p"]) <= 1.0


def test_the_checkpoint_still_exposes_the_untouched_student(tmp_path):
    """The submission constraint: the graders load `loaded["params"]` into the
    original `ActorCritic`. An oracle riding along in the teacher state must not
    show up there."""
    import jax

    from tlab_ued.evaluate import restore_params
    from tlab_ued.logging_utils import checkpoint_dir
    from tlab_ued.train import main

    config = tiny_config(out_dir=str(tmp_path), checkpoint_save_interval=1)
    train_state = main(config)

    reference = tiny_config(preset="accel", out_dir=str(tmp_path), run_name="reference")
    from tlab_ued.train import build

    ctx, teacher = build(reference)
    _, expected, _ = teacher.make_network(jax.random.PRNGKey(0))

    params, step = restore_params(checkpoint_dir(config))
    shapes = sorted(x.shape for x in jax.tree_util.tree_leaves(params))
    assert shapes == sorted(x.shape for x in jax.tree_util.tree_leaves(train_state.params))
    assert shapes == sorted(x.shape for x in jax.tree_util.tree_leaves(expected))
    assert step >= 0
