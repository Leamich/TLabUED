"""Tests for reading a run's level buffer back out of its checkpoints.

The load path is orbax and needs no coverage here - what needs it is the slicing:
a buffer of 4000 slots with 300 filled must diagnose 300 levels, not 4000, or the
report describes initialisation placeholders as if the teacher had chosen them.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax", reason="the diagnostics are jax code")
pytest.importorskip("jaxued")

from tlab_ued.buffer_diagnostics import buffer_levels, diagnose_checkpoint  # noqa: E402


def _sampler(capacity: int, size: int):
    """A buffer whose filled part is distinguishable from its empty part.

    Filled slots get a goal at (1, 1); empty ones keep the (0, 0) an
    uninitialised slot would have.
    """
    goal = np.zeros((capacity, 2), dtype=np.uint32)
    goal[:size] = 1
    return {
        "levels": {
            "wall_map": np.zeros((capacity, 13, 13), dtype=bool),
            "goal_pos": goal,
            "agent_pos": np.full((capacity, 2), 5, dtype=np.uint32),
            "agent_dir": np.zeros((capacity,), dtype=np.uint32),
            "width": np.full((capacity,), 13, dtype=np.uint32),
            "height": np.full((capacity,), 13, dtype=np.uint32),
        },
        "scores": np.linspace(0.0, 1.0, capacity, dtype=np.float32),
        "size": np.asarray(size),
    }


def test_only_the_filled_part_of_the_buffer_is_read():
    levels, size = buffer_levels(_sampler(64, 7))
    assert size == 7
    assert np.asarray(levels.wall_map).shape[0] == 7
    assert (np.asarray(levels.goal_pos) == 1).all()


def test_an_empty_buffer_yields_no_levels():
    levels, size = buffer_levels(_sampler(64, 0))
    assert size == 0
    assert np.asarray(levels.wall_map).shape[0] == 0


def test_size_past_capacity_is_clamped():
    # A full buffer keeps counting inserts in some samplers; the slice must not
    # run off the end of the array.
    _, size = buffer_levels(_sampler(64, 4000))
    assert size == 64


def test_a_checkpoint_without_a_sampler_says_so(tmp_path, monkeypatch):
    import tlab_ued.buffer_diagnostics as bd

    monkeypatch.setattr(bd, "restore_tree", lambda _: {"params": {}, "teacher_state": {}})
    with pytest.raises(KeyError, match="no level buffer"):
        diagnose_checkpoint(str(tmp_path))
