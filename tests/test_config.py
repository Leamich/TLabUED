"""Tests for the config layer.

Only the JAX-free parts run everywhere; anything that touches the parser needs
the real dependencies (the parser asks the teacher and score registries for
their choices), so those are skipped when JAX is absent.
"""

from __future__ import annotations

import pytest

from tlab_ued.config import (
    PRESETS,
    STUDENT_DEFAULTS,
    StudentConfigError,
    assert_student_frozen,
    default_run_name,
    finalize,
    make_config,
    to_argv,
    upstream_config,
)

jax = pytest.importorskip("jax", reason="parser and teachers need the full stack")


def test_defaults_pass_the_freeze_guard():
    assert_student_frozen(make_config(preset="accel"))


@pytest.mark.parametrize("field,value", [("lr", 3e-4), ("num_updates", 1000), ("epoch_ppo", 8)])
def test_changing_a_student_field_is_refused(field, value):
    config = make_config(preset="accel", **{field: value})
    with pytest.raises(StudentConfigError) as excinfo:
        assert_student_frozen(config)
    assert field in str(excinfo.value)


def test_the_guard_can_be_overridden_explicitly():
    # Ablations are allowed, they just have to be deliberate.
    assert_student_frozen(make_config(preset="accel", lr=3e-4, allow_student_changes=True))


def test_teacher_settings_are_not_frozen():
    assert_student_frozen(make_config(preset="accel", num_edits=10, replay_prob=0.5))


def test_smoke_shortens_the_budget_and_says_so():
    config = make_config(preset="accel", smoke=True)
    assert config["num_updates"] == 500
    assert config["allow_student_changes"] is True
    assert_student_frozen(config)


def test_presets_match_the_upstream_commands():
    # DR is its own script; PLR-perp is maze_plr with exploratory updates off;
    # ACCEL is maze_plr --use_accel.
    assert PRESETS["dr"]["teacher"] == "dr"
    assert PRESETS["plr"]["exploratory_grad_updates"] is False
    assert PRESETS["accel"]["use_accel"] is True
    for name in ("dr", "plr", "accel"):
        config = make_config(preset=name)
        for field, expected in STUDENT_DEFAULTS.items():
            assert config[field] == expected, f"{name} changed the student field {field}"


def test_run_names_distinguish_the_baselines():
    names = {name: default_run_name(make_config(preset=name)) for name in ("dr", "plr", "accel")}
    assert len(set(names.values())) == 3, names


def test_num_updates_must_divide_into_eval_steps():
    with pytest.raises(ValueError):
        finalize({**make_config(preset="dr"), "num_updates": 501, "eval_freq": 250})


def test_to_argv_round_trips_through_the_parser():
    from tlab_ued.config import from_args

    config = make_config(preset="accel", seed=3, num_edits=7)
    restored = from_args(to_argv(config))
    for key, value in config.items():
        if key in ("out_dir", "group_name"):
            continue
        assert restored[key] == value, key


def test_upstream_config_drops_only_our_extras():
    config = make_config(preset="accel")
    stripped = upstream_config(config)
    assert "teacher" not in stripped and "log_media" not in stripped
    # Everything upstream's main() reads must survive.
    for key in ("use_accel", "exploratory_grad_updates", "score_function", "num_updates",
                "eval_levels", "run_name", "group_name", "seed", "n_walls"):
        assert key in stripped, key
