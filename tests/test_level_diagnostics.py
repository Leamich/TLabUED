"""Tests for the BFS level diagnostics.

The interesting claim is that `solve_levels` counts *env steps under optimal
play*, not grid cells - turns cost a step, the goal cell is not walkable, and
the episode ends by stepping into it. So the reference is the environment
itself: a brute-force search over `Maze.step_env` transitions, which cannot
share a bug with the vectorised implementation.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pytest

jax = pytest.importorskip("jax", reason="the diagnostics are jax code")
jnp = pytest.importorskip("jax.numpy")

from jaxued.environments import Maze  # noqa: E402
from jaxued.environments.maze import Level  # noqa: E402

from tlab_ued.config import make_config  # noqa: E402
from tlab_ued.level_diagnostics import (  # noqa: E402
    UNREACHABLE,
    diagnose,
    mutation_ladder,
    solve_levels,
    summarize,
)
from tlab_ued.levels import get_generator, get_mutator  # noqa: E402

DIR_TO_VEC = ((1, 0), (0, 1), (-1, 0), (0, -1))

# 7x7 so the brute force stays quick; the env is built to match, because a
# forward move is clipped at `max_width/max_height`, not at the level's edge.
EMPTY_ROOM = """
>.....G
.......
.......
.......
.......
.......
.......
"""

SEALED_GOAL = """
>......
.......
..###..
..#G#..
..###..
.......
.......
"""

FACING_GOAL = """
.......
.......
...>G..
.......
.......
.......
.......
"""

# The goal is six cells away in a straight line, but the only way around the
# wall is the long way down and back up - the case a cell-distance heuristic
# would get wrong.
CORRIDOR = """
>#....G
.#.....
.#.....
.#.....
.#.....
.#.....
.......
"""


def env7() -> Maze:
    return Maze(max_height=7, max_width=7, agent_view_size=5, normalize_obs=True)


def brute_force_steps(env: Maze, level: Level, max_depth: int = 200):
    """Minimum env steps to end the episode, searched over real transitions."""
    params = env.default_params
    rng = jax.random.PRNGKey(0)
    init_state = env.init_state_from_level(level)

    start = (int(init_state.agent_pos[0]), int(init_state.agent_pos[1]), int(init_state.agent_dir))
    seen = {start}
    frontier = [start]
    for depth in range(1, max_depth + 1):
        nxt = []
        for x, y, d in frontier:
            state = init_state.replace(
                agent_pos=jnp.array([x, y], dtype=jnp.uint32),
                agent_dir=jnp.uint8(d),
                time=0,
                terminal=False,
            )
            for action in (0, 1, 2):  # left, right, forward
                _, next_state, reward, _, _ = env.step_env(rng, state, action, params)
                if float(reward) > 0:
                    return depth
                key = (
                    int(next_state.agent_pos[0]),
                    int(next_state.agent_pos[1]),
                    int(next_state.agent_dir),
                )
                if key not in seen:
                    seen.add(key)
                    nxt.append(key)
        frontier = nxt
        if not frontier:
            return None
    return None


def python_reference_steps(wall_map, goal_pos, agent_pos, agent_dir):
    """Plain-Python BFS over `(x, y, dir)`, used to cross-check whole batches."""
    wall = np.asarray(wall_map, dtype=bool)
    h, w = wall.shape
    gx, gy = int(goal_pos[0]), int(goal_pos[1])
    blocked = wall.copy()
    blocked[gy, gx] = True

    start = (int(agent_pos[0]), int(agent_pos[1]), int(agent_dir))
    dist = {start: 0}
    queue = deque([start])
    while queue:
        x, y, d = queue.popleft()
        cost = dist[(x, y, d)]
        dx, dy = DIR_TO_VEC[d]
        if (x + dx, y + dy) == (gx, gy):
            return cost + 1
        neighbours = [(x, y, (d + 1) % 4), (x, y, (d - 1) % 4)]
        nx, ny = x + dx, y + dy
        if 0 <= nx < w and 0 <= ny < h and not blocked[ny, nx]:
            neighbours.append((nx, ny, d))
        for key in neighbours:
            if key not in dist:
                dist[key] = cost + 1
                queue.append(key)
    return None


def stack(*level_strs):
    return Level.stack([Level.from_str(s) for s in level_strs])


# --- the numbers themselves -------------------------------------------------


def test_matches_a_brute_force_search_over_the_real_env():
    env = env7()
    level_strs = [EMPTY_ROOM, FACING_GOAL, CORRIDOR]
    steps = np.asarray(solve_levels(stack(*level_strs))["steps"])
    for level_str, got in zip(level_strs, steps):
        assert int(got) == brute_force_steps(env, Level.from_str(level_str))


def test_one_forward_step_when_already_facing_the_goal():
    # The episode ends on the move *into* the goal, so this is 1, not 0.
    assert int(solve_levels(stack(FACING_GOAL))["steps"][0]) == 1


def test_a_walled_off_goal_is_unreachable():
    out = solve_levels(stack(SEALED_GOAL))
    assert int(out["steps"][0]) == UNREACHABLE
    assert brute_force_steps(env7(), Level.from_str(SEALED_GOAL)) is None
    # The rest of the room is still walkable, so this is not "nothing reachable".
    assert int(out["reachable_cells"][0]) > 1


def test_turning_costs_a_step():
    # Agent at (0,0) facing right, goal at (6,0): 6 forward moves, no turns.
    assert int(solve_levels(stack(EMPTY_ROOM))["steps"][0]) == 6
    # Same room, but the agent starts facing away: two turns first.
    turned_away = EMPTY_ROOM.replace(">", "<", 1)
    assert int(solve_levels(stack(turned_away))["steps"][0]) == 8


def test_matches_a_python_bfs_on_generated_levels():
    config = make_config(preset="accel")
    env = Maze(max_height=13, max_width=13, agent_view_size=5, normalize_obs=True)
    sample = get_generator(config, env)
    levels = jax.vmap(sample)(jax.random.split(jax.random.PRNGKey(0), 64))

    steps = np.asarray(solve_levels(levels)["steps"])
    wall_maps = np.asarray(levels.wall_map)
    goals = np.asarray(levels.goal_pos)
    agents = np.asarray(levels.agent_pos)
    dirs = np.asarray(levels.agent_dir)
    for i in range(len(steps)):
        expected = python_reference_steps(wall_maps[i], goals[i], agents[i], dirs[i])
        assert int(steps[i]) == (UNREACHABLE if expected is None else expected)


def test_reachable_and_free_cell_counts():
    out = solve_levels(stack(EMPTY_ROOM))
    # 49 cells, one of them the goal: the agent can stand on the other 48.
    assert int(out["free_cells"][0]) == 48
    assert int(out["reachable_cells"][0]) == 48
    assert int(out["num_walls"][0]) == 0


# --- the summary layer ------------------------------------------------------


def test_summary_percentages_are_consistent():
    stats = summarize(stack(EMPTY_ROOM, FACING_GOAL, CORRIDOR, SEALED_GOAL))
    assert stats["num_levels"] == 4
    assert stats["solvable_pct"] == pytest.approx(75.0)
    assert stats["unreachable_pct"] == pytest.approx(25.0)
    assert stats["malformed_pct"] == 0.0
    assert sum(stats["buckets_pct"].values()) == pytest.approx(100.0)
    # Best attainable return under the env's time penalty.
    assert 0.0 < stats["optimal_return_mean"] < 1.0


def test_summary_survives_an_all_unsolvable_batch():
    stats = summarize(stack(SEALED_GOAL, SEALED_GOAL))
    assert stats["solvable_pct"] == 0.0
    assert np.isnan(stats["steps_mean"])
    assert stats["buckets_pct"]["unsolvable"] == pytest.approx(100.0)


def test_over_the_time_limit_counts_separately():
    stats = summarize(stack(EMPTY_ROOM), max_steps_in_episode=3)
    assert stats["solvable_pct"] == 0.0
    assert stats["unreachable_pct"] == 0.0
    assert stats["over_time_limit_pct"] == pytest.approx(100.0)


# --- wiring -----------------------------------------------------------------


def _level_fns(config):
    env = Maze(max_height=13, max_width=13, agent_view_size=5, normalize_obs=True)
    return get_generator(config, env), get_mutator(config, env)


def test_diagnose_is_off_when_the_flag_is_zero():
    config = make_config(preset="accel", diagnose_levels=0)
    assert diagnose(config, *_level_fns(config)) == {}


def test_diagnose_reports_the_mutator_only_for_mutating_teachers():
    config = make_config(preset="accel", diagnose_levels=16)
    reports = diagnose(config, *_level_fns(config))
    assert set(reports) == {"generator", "mutator"}

    config = make_config(preset="plr", diagnose_levels=16)
    assert set(diagnose(config, *_level_fns(config))) == {"generator"}


def test_diagnose_does_not_depend_on_the_training_stream():
    # Same seed, same numbers - and, more to the point, drawing them consumes
    # nothing that training would have drawn.
    config = make_config(preset="dr", diagnose_levels=16, seed=3)
    first = diagnose(config, *_level_fns(config))
    second = diagnose(config, *_level_fns(config))
    assert first["generator"]["steps_mean"] == second["generator"]["steps_mean"]

    other = make_config(preset="dr", diagnose_levels=16, seed=4)
    assert (
        diagnose(other, *_level_fns(other))["generator"]["steps_mean"]
        != first["generator"]["steps_mean"]
    )

# === The mutation ladder ===


def test_ladder_reports_both_arms_at_every_requested_round():
    config = make_config(preset="accel", seed=0)
    ladder = mutation_ladder(
        config, *_level_fns(config), num_levels=32, rounds=(0, 2), proposals=2
    )
    assert set(ladder) == {"meta", "random", "hardest_of_2"}
    for arm in ("random", "hardest_of_2"):
        assert set(ladder[arm]) == {"0", "2"}


def test_both_arms_start_from_the_same_levels():
    # Round 0 is "no mutation applied yet", so the arms can only diverge after
    # it - otherwise the ladder would be comparing two different populations.
    config = make_config(preset="accel", seed=0)
    ladder = mutation_ladder(
        config, *_level_fns(config), num_levels=32, rounds=(0, 1), proposals=2
    )
    assert ladder["random"]["0"] == ladder["hardest_of_2"]["0"]


def test_selection_never_prefers_a_broken_child():
    """The selection arm scores an unsolvable child -1, so it can only lose.

    With enough proposals per parent, the arm should therefore keep solvability
    at least as high as the unselected one - the whole point of the -1.
    """
    config = make_config(preset="accel", seed=0)
    ladder = mutation_ladder(
        config, *_level_fns(config), num_levels=64, rounds=(6,), proposals=4
    )
    assert ladder["hardest_of_4"]["6"]["solvable_pct"] >= ladder["random"]["6"]["solvable_pct"]


def test_the_ladder_does_not_depend_on_the_training_stream():
    config = make_config(preset="accel", seed=5)
    first = mutation_ladder(config, *_level_fns(config), num_levels=32, rounds=(1,), proposals=2)
    second = mutation_ladder(config, *_level_fns(config), num_levels=32, rounds=(1,), proposals=2)
    assert first == second
