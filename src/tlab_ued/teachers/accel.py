"""ACCEL - evolving curricula via edits to high-regret levels.

Faithful port of the `--use_accel` path of JaxUED `examples/maze_plr.py` @ 0f8f128
(Parker-Holder et al., 2022). This is the assignment's main baseline to beat.

ACCEL is PLR plus one extra branch: immediately after a *replay* update, the
levels that were just replayed are mutated (`--num_edits` wall edits), rolled
out, scored and offered to the buffer. Difficulty therefore ratchets up in small
steps from levels the student already finds informative, instead of being
resampled from scratch.

Branch selection is upstream's arithmetic on `update_state`:
    branch = (1 - s) * replay_decision + 2 * s
so s == REPLAY forces the mutation branch, and otherwise the usual PLR coin flip
decides between new levels and replay.
"""

from __future__ import annotations

from typing import List

import chex
import jax
import jax.numpy as jnp

from jaxued.utils import compute_max_returns

from tlab_ued.teachers.base import BranchFn, TrainState
from tlab_ued.teachers.plr import PLRTeacher, UpdateState


class ACCELTeacher(PLRTeacher):
    name = "accel"

    @property
    def branches(self) -> List[BranchFn]:
        return [self.on_new_levels, self.on_replay_levels, self.on_mutate_levels]

    def select_branch(self, rng: chex.PRNGKey, train_state: TrainState) -> chex.Array:
        s = train_state.teacher_state["update_state"]
        replay_decision = self.level_sampler.sample_replay_decision(
            train_state.teacher_state["sampler"], rng
        )
        return (1 - s) * replay_decision + 2 * s

    def on_mutate_levels(self, rng: chex.PRNGKey, train_state: TrainState):
        """Edit the levels just replayed, score the children, offer them to the buffer.

        The policy is updated on these rollouts only if
        `config["exploratory_grad_updates"]` is set - with it off, mutation is
        pure exploration of the level space and costs no gradient signal.
        """
        ctx = self.ctx
        ts = train_state.teacher_state
        sampler = ts["sampler"]

        rng, rng_mutate, rng_reset = jax.random.split(rng, 3)
        parent_levels = ts["replay_last_level_batch"]
        child_levels = jax.vmap(ctx.mutate_level, (0, 0, None))(
            jax.random.split(rng_mutate, ctx.num_envs), parent_levels, self.config["num_edits"]
        )
        init_obs, init_env_state = ctx.reset_to_levels(rng_reset, child_levels)

        carry, traj, advantages, targets = ctx.rollout(
            rng, train_state, ctx.initial_carry(), init_obs, init_env_state
        )
        rng, train_state, _, _, _, _ = carry
        obs, actions, rewards, dones, log_probs, values, info = traj

        max_returns = compute_max_returns(dones, rewards)
        scores = ctx.score(traj, advantages, targets, max_returns, child_levels)
        sampler, _ = self.level_sampler.insert_batch(
            sampler, child_levels, scores, {"max_return": max_returns}
        )

        (rng, train_state), losses = ctx.ppo_update(
            rng,
            train_state,
            ctx.initial_carry(),
            (obs, actions, dones, log_probs, values, targets, advantages),
            update_grad=self.config["exploratory_grad_updates"],
        )

        metrics = {
            "losses": jax.tree_util.tree_map(lambda x: x.mean(), losses),
            "mean_num_blocks": child_levels.wall_map.sum() / ctx.num_envs,
        }
        train_state = train_state.replace(
            teacher_state={
                **ts,
                "sampler": sampler,
                # Back to DR so the next update is a normal PLR coin flip.
                "update_state": jnp.asarray(UpdateState.DR, dtype=jnp.int32),
                "num_mutation_updates": ts["num_mutation_updates"] + 1,
                "mutation_last_level_batch": child_levels,
            }
        )
        return (rng, train_state), metrics
