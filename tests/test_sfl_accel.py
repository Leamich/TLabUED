"""Tests for the SFL-ACCEL teacher.

Two kinds. The arithmetic ones are cheap and guard the claim the whole
comparison rests on - that this method and ACCEL spend the same budget. The
integration one runs a handful of updates on CPU through every branch, which is
where a mismatched pytree or a level batch that changes shape between updates
would otherwise wait until an hour into a GPU run.
"""

from __future__ import annotations

import pytest

pytest.importorskip("jax", reason="needs the full JAX stack")

import jax.numpy as jnp  # noqa: E402

from tlab_ued.config import make_config  # noqa: E402
from tlab_ued.scoring import first_episode_success, learnability  # noqa: E402
from tlab_ued.teachers.sfl_accel import (  # noqa: E402
    phase_length,
    sfl_budget_report,
    validate,
)


# === the score ===


def test_learnability_peaks_at_a_coin_flip():
    scores = learnability(jnp.array([0.0, 0.25, 0.5, 0.75, 1.0]))
    assert scores[2] == pytest.approx(0.25)
    # The property MaxMC lacks: never-solved and always-solved are both worthless.
    assert scores[0] == 0.0 and scores[-1] == 0.0
    assert scores[1] == pytest.approx(scores[3])


def test_learnability_of_a_single_attempt_is_zero_either_way():
    """Why the teacher plays each level k times - one sample carries no signal."""
    assert float(learnability(jnp.array(0.0))) == 0.0
    assert float(learnability(jnp.array(1.0))) == 0.0


def test_first_episode_success_ignores_later_episodes():
    # Two envs, five steps. Env 0 fails its first episode and solves the second
    # (AutoReplay puts it back on the same level); env 1 solves the first.
    dones = jnp.array([[False, False], [True, True], [False, False], [True, False], [False, False]])
    rewards = jnp.array([[0.0, 0.0], [0.0, 0.7], [0.0, 0.0], [0.9, 0.0], [0.0, 0.0]])
    assert first_episode_success(dones, rewards).tolist() == [0.0, 1.0]


def test_first_episode_success_counts_a_timeout_as_a_failure():
    dones = jnp.array([[False], [False], [True]])
    rewards = jnp.zeros((3, 1))
    assert first_episode_success(dones, rewards).tolist() == [0.0]


# === the budget ===


def test_phase_costs_what_accel_spends_on_its_dr_branch():
    accel = make_config(preset="accel")
    sfl = make_config(preset="sfl_accel")

    accel_budget = sfl_budget_report(accel)
    sfl_budget = sfl_budget_report(sfl)

    # The frozen quantity: env steps. Identical, not merely close.
    assert sfl_budget["env_steps"] == accel_budget["env_steps"]
    # And the gradient updates, which is what would otherwise confound a win.
    ratio = sfl_budget["expected_gradient_updates"] / accel_budget["expected_gradient_updates"]
    assert 0.98 <= ratio <= 1.02, (sfl_budget, accel_budget)
    # The phase is paid for out of the DR share.
    assert sfl_budget["reserved_updates"] == pytest.approx(
        accel_budget["expected_dr_updates"], rel=0.05
    )


def test_accel_budget_split_matches_the_hand_computation():
    budget = sfl_budget_report(make_config(preset="accel"))
    assert budget["total_updates"] == 30000
    # The `sfl_*` defaults exist in every config; ACCEL must not be charged for
    # a phase it does not run.
    assert budget["reserved_updates"] == 0
    assert budget["env_steps"] == 30000 * 32 * 256
    # replay share p/(1+p) = 0.444..., mutation the same, DR the remainder.
    assert budget["expected_gradient_updates"] == pytest.approx(30000 * 0.8 / 1.8)
    assert budget["expected_dr_updates"] == pytest.approx(30000 - 2 * 30000 * 0.8 / 1.8)


def test_phase_length_is_whole_updates():
    config = make_config(preset="sfl_accel")
    assert phase_length(config) == 28
    assert phase_length(config) * config["num_train_envs"] == (
        config["sfl_num_levels"] * config["sfl_num_attempts"]
    )


def test_no_phase_means_no_reserved_updates():
    budget = sfl_budget_report(make_config(preset="sfl_accel_nophase"))
    assert budget["reserved_updates"] == 0
    assert phase_length(make_config(preset="sfl_accel_nophase")) == 0


@pytest.mark.parametrize(
    "overrides",
    [
        {"sfl_num_attempts": 1},  # a single attempt scores every level 0
        {"sfl_num_attempts": 5},  # does not divide num_train_envs
        {"sfl_num_levels": 225},  # phase would not fit whole updates
        {"sfl_num_levels": 4000},  # phase longer than the period
        {"sfl_topk": 1000},  # keeping more than were evaluated
    ],
)
def test_bad_sizing_is_refused_before_the_run(overrides):
    with pytest.raises(ValueError):
        validate({**make_config(preset="sfl_accel"), **overrides})


def test_config_validation_runs_at_build_time():
    with pytest.raises(ValueError):
        make_config(preset="sfl_accel", sfl_num_attempts=5)


# === the run ===


def tiny_config(preset: str = "sfl_accel", **overrides):
    """Small enough for CPU, sized so every branch is exercised.

    4 envs and 2 attempts means 2 levels scored per update; a 4-level phase is
    then 2 updates, and with a period of 4 the other 2 updates go to the ACCEL
    branches. `overrides` replace these rather than colliding with them.
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
    )
    return make_config(preset=preset, **{**defaults, **overrides})


def test_every_branch_runs_and_the_state_keeps_its_shape(tmp_path):
    from tlab_ued.train import main

    config = tiny_config(out_dir=str(tmp_path))
    train_state = main(config)
    teacher_state = train_state.teacher_state

    assert int(train_state.update_count) == config["num_updates"]
    # 2 phase updates per period of 4, over 8 updates.
    assert int(teacher_state["num_sfl_updates"]) == 4
    assert int(teacher_state["num_replay_updates"]) > 0
    assert int(teacher_state["num_mutation_updates"]) > 0
    # The phase inserted its top-k, so the buffer is not empty.
    assert int(teacher_state["sampler"]["size"]) >= config["sfl_topk"]
    # Every buffer entry carries the success rate its score was computed from.
    assert teacher_state["sampler"]["levels_extra"]["p"].shape == (
        config["level_buffer_capacity"],
    )
    assert teacher_state["sfl_topk_levels"].wall_map.shape[0] == config["sfl_topk"]


def test_branch_counts_add_up_to_the_updates(tmp_path):
    """The budget claim, measured rather than predicted."""
    from tlab_ued.train import main

    state = main(tiny_config(out_dir=str(tmp_path))).teacher_state
    counted = sum(
        int(state[key])
        for key in (
            "num_dr_updates",
            "num_replay_updates",
            "num_mutation_updates",
            "num_sfl_updates",
        )
    )
    assert counted == 8


def test_curriculum_metrics_reach_the_csv(tmp_path):
    import csv

    from tlab_ued.logging_utils import run_dir
    from tlab_ued.train import main

    config = tiny_config(out_dir=str(tmp_path))
    main(config)
    with open(f"{run_dir(config)}/metrics.csv") as f:
        rows = list(csv.DictReader(f))

    assert rows, "no metrics were written"
    for column in ("train/success_rate", "train/learnability", "sfl/topk_learnability"):
        assert column in rows[0], sorted(rows[0])
    assert 0.0 <= float(rows[0]["train/success_rate"]) <= 1.0


def test_the_phase_ablation_also_runs(tmp_path):
    from tlab_ued.train import main

    config = tiny_config("sfl_accel_nophase", sfl_period=0, out_dir=str(tmp_path))
    state = main(config).teacher_state
    assert int(state["num_sfl_updates"]) == 0
    assert int(state["num_replay_updates"]) > 0
