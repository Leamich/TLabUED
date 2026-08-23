"""Smoke tests for the teacher plugins.

These run a couple of updates on CPU, so they are slow-ish but catch the errors
that would otherwise only surface an hour into a GPU run: a branch returning a
mismatched pytree, a teacher state that changes shape between updates, a
checkpoint that cannot be read back.
"""

from __future__ import annotations

import pytest

pytest.importorskip("jax", reason="needs the full JAX stack")

import jax  # noqa: E402

from tlab_ued.config import make_config  # noqa: E402
from tlab_ued.scoring import SCORE_FUNCTIONS  # noqa: E402
from tlab_ued.teachers import TEACHERS  # noqa: E402


def tiny_config(**overrides):
    """The smallest config that still exercises every branch."""
    return make_config(
        num_updates=4,
        eval_freq=2,
        eval_num_attempts=1,
        num_train_envs=2,
        num_steps=8,
        level_buffer_capacity=8,
        minimum_fill_ratio=0.0,
        checkpoint_save_interval=0,
        log_media="none",
        allow_student_changes=True,
        **overrides,
    )


@pytest.mark.parametrize("teacher", sorted(TEACHERS))
def test_teacher_runs_a_few_updates(teacher, tmp_path):
    from tlab_ued.train import main

    config = tiny_config(preset=teacher, teacher=teacher, out_dir=str(tmp_path))
    train_state = main(config)
    assert int(train_state.update_count) == config["num_updates"]


@pytest.mark.parametrize("score", sorted(SCORE_FUNCTIONS))
def test_every_score_function_drives_accel(score, tmp_path):
    from tlab_ued.train import main

    config = tiny_config(preset="accel", score_function=score, out_dir=str(tmp_path))
    assert int(main(config).update_count) == config["num_updates"]


def test_dr_and_plr_wrap_the_env_differently():
    from jaxued.wrappers import AutoReplayWrapper, AutoResetWrapper

    from tlab_ued.train import build

    dr_ctx, _ = build(tiny_config(preset="dr"))
    plr_ctx, _ = build(tiny_config(preset="plr"))
    assert isinstance(dr_ctx.env, AutoResetWrapper)
    assert isinstance(plr_ctx.env, AutoReplayWrapper)


def test_checkpoint_exposes_params_at_the_top_level(tmp_path):
    """The graders' harness restores untargeted and reads `loaded["params"]`."""
    from tlab_ued.evaluate import restore_params
    from tlab_ued.logging_utils import checkpoint_dir
    from tlab_ued.train import main

    config = tiny_config(preset="accel", checkpoint_save_interval=1, out_dir=str(tmp_path))
    train_state = main(config)

    params, step = restore_params(checkpoint_dir(config))
    # Untargeted restore yields plain dicts, so compare the leaves rather than
    # the container types: same number of tensors, same shapes.
    restored = [x.shape for x in jax.tree_util.tree_leaves(params)]
    original = [x.shape for x in jax.tree_util.tree_leaves(train_state.params)]
    assert sorted(restored) == sorted(original)
    assert step >= 0
