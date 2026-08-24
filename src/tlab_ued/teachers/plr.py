"""PLR / PLR-perp - prioritised level replay.

Faithful port of the non-ACCEL path of JaxUED `examples/maze_plr.py` @ 0f8f128.

Each update the teacher flips a (buffer-state dependent) coin:
  - *new levels*: sample a fresh random batch, roll out, score it, and insert it
    into the level buffer. The gradient is applied only if
    `--exploratory_grad_updates`; with `--no-exploratory_grad_updates` this is
    PLR-perp (Jiang et al., 2021), the assignment's PLR baseline.
  - *replay*: draw a batch from the buffer by score/staleness priority, roll out,
    refresh its score, and always apply the gradient.

The score function - the part that decides what "worth replaying" means - is a
plugin (see `tlab_ued.scoring`).
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any, Dict, List

import chex
import jax
import jax.numpy as jnp

from jaxued.utils import compute_max_returns
from jaxued.wrappers import AutoReplayWrapper

from tlab_ued.teachers.base import (
    BranchFn,
    Teacher,
    TrainState,
    branch_budget,
    make_level_sampler,
)


class UpdateState(IntEnum):
    """Which kind of update ran last (ACCEL mutates only after a replay)."""

    DR = 0
    REPLAY = 1


class PLRTeacher(Teacher):
    name = "plr"

    def __init__(self, ctx):
        super().__init__(ctx)
        self.level_sampler = make_level_sampler(self.config)

    @classmethod
    def wrap_env(cls, config: Dict[str, Any], base_env, sample_random_level):
        # Replay the *same* level when an episode ends: a rollout stays on the
        # batch of levels the teacher chose for this update.
        return AutoReplayWrapper(base_env)

    def init_teacher_state(self, rng: chex.PRNGKey) -> Any:
        """The level buffer plus the bookkeeping upstream keeps in TrainState.

        Consumes no rng - matching maze_plr, where `create_train_state` uses a
        fixed `PRNGKey(0)` for the placeholder level.
        """
        ctx = self.ctx
        pholder_level = ctx.sample_random_level(jax.random.PRNGKey(0))
        sampler = self.level_sampler.initialize(pholder_level, {"max_return": -jnp.inf})
        pholder_level_batch = jax.tree_util.tree_map(
            lambda x: jnp.array([x]).repeat(ctx.num_envs, axis=0), pholder_level
        )
        zero = jnp.asarray(0, dtype=jnp.int32)
        return {
            "sampler": sampler,
            "update_state": zero,
            "num_dr_updates": zero,
            "num_replay_updates": zero,
            "num_mutation_updates": zero,
            "dr_last_level_batch": pholder_level_batch,
            "replay_last_level_batch": pholder_level_batch,
            "mutation_last_level_batch": pholder_level_batch,
        }

    # --- branches -------------------------------------------------------------
    @property
    def branches(self) -> List[BranchFn]:
        return [self.on_new_levels, self.on_replay_levels]

    def select_branch(self, rng: chex.PRNGKey, train_state: TrainState) -> chex.Array:
        return self.level_sampler.sample_replay_decision(
            train_state.teacher_state["sampler"], rng
        ).astype(int)

    def on_new_levels(self, rng: chex.PRNGKey, train_state: TrainState):
        """Sample fresh random levels, score them, and consider them for the buffer.

        The policy is updated on these rollouts only if
        `config["exploratory_grad_updates"]` is set.
        """
        ctx = self.ctx
        ts = train_state.teacher_state
        sampler = ts["sampler"]

        rng, rng_levels, rng_reset = jax.random.split(rng, 3)
        new_levels = ctx.sample_levels(rng_levels)
        init_obs, init_env_state = ctx.reset_to_levels(rng_reset, new_levels)

        carry, traj, advantages, targets = ctx.rollout(
            rng, train_state, ctx.initial_carry(), init_obs, init_env_state
        )
        rng, train_state, _, _, _, _ = carry
        obs, actions, rewards, dones, log_probs, values, info = traj

        max_returns = compute_max_returns(dones, rewards)
        scores = ctx.score(traj, advantages, targets, max_returns, new_levels)
        sampler, _ = self.level_sampler.insert_batch(
            sampler, new_levels, scores, {"max_return": max_returns}
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
            "mean_num_blocks": new_levels.wall_map.sum() / ctx.num_envs,
        }
        train_state = train_state.replace(
            teacher_state={
                **ts,
                "sampler": sampler,
                "update_state": jnp.asarray(UpdateState.DR, dtype=jnp.int32),
                "num_dr_updates": ts["num_dr_updates"] + 1,
                "dr_last_level_batch": new_levels,
            }
        )
        return (rng, train_state), metrics

    def on_replay_levels(self, rng: chex.PRNGKey, train_state: TrainState):
        """Replay high-priority levels from the buffer and always learn from them."""
        ctx = self.ctx
        ts = train_state.teacher_state
        sampler = ts["sampler"]

        rng, rng_levels, rng_reset = jax.random.split(rng, 3)
        sampler, (level_inds, levels) = self.level_sampler.sample_replay_levels(
            sampler, rng_levels, ctx.num_envs
        )
        init_obs, init_env_state = ctx.reset_to_levels(rng_reset, levels)

        carry, traj, advantages, targets = ctx.rollout(
            rng, train_state, ctx.initial_carry(), init_obs, init_env_state
        )
        rng, train_state, _, _, _, _ = carry
        obs, actions, rewards, dones, log_probs, values, info = traj

        # A replayed level keeps the best return ever seen on it, which is what
        # makes MaxMC a (rough) regret proxy.
        max_returns = jnp.maximum(
            self.level_sampler.get_levels_extra(sampler, level_inds)["max_return"],
            compute_max_returns(dones, rewards),
        )
        scores = ctx.score(traj, advantages, targets, max_returns, levels)
        sampler = self.level_sampler.update_batch(
            sampler, level_inds, scores, {"max_return": max_returns}
        )

        (rng, train_state), losses = ctx.ppo_update(
            rng,
            train_state,
            ctx.initial_carry(),
            (obs, actions, dones, log_probs, values, targets, advantages),
            update_grad=True,
        )

        metrics = {
            "losses": jax.tree_util.tree_map(lambda x: x.mean(), losses),
            "mean_num_blocks": levels.wall_map.sum() / ctx.num_envs,
        }
        train_state = train_state.replace(
            teacher_state={
                **ts,
                "sampler": sampler,
                "update_state": jnp.asarray(UpdateState.REPLAY, dtype=jnp.int32),
                "num_replay_updates": ts["num_replay_updates"] + 1,
                "replay_last_level_batch": levels,
            }
        )
        return (rng, train_state), metrics

    # --- reporting -------------------------------------------------------------
    def startup_report(self) -> Dict[str, Any]:
        """How the budget is expected to split across branches."""
        return branch_budget(self.config)

    def log_dict(self, train_state: TrainState) -> Dict[str, Dict[str, Any]]:
        """Level-buffer health, exactly as upstream's `train_state_to_log_dict`."""
        ts = train_state.teacher_state
        sampler = ts["sampler"]
        idx = jnp.arange(self.level_sampler.capacity) < sampler["size"]
        s = jnp.maximum(idx.sum(), 1)
        return {
            "log": {
                "level_sampler/size": sampler["size"],
                "level_sampler/episode_count": sampler["episode_count"],
                "level_sampler/max_score": sampler["scores"].max(),
                "level_sampler/weighted_score": (
                    sampler["scores"] * self.level_sampler.level_weights(sampler)
                ).sum(),
                "level_sampler/mean_score": (sampler["scores"] * idx).sum() / s,
            },
            "info": {
                "num_dr_updates": ts["num_dr_updates"],
                "num_replay_updates": ts["num_replay_updates"],
                "num_mutation_updates": ts["num_mutation_updates"],
            },
        }

    def media(self, train_state: TrainState) -> Dict[str, Any]:
        """Rendered levels: the batches just trained on, and the buffer extremes."""
        if self.config.get("log_media", "all") == "none":
            return {}
        ctx = self.ctx
        ts = train_state.teacher_state
        sampler = ts["sampler"]
        render_batch = jax.vmap(ctx.renderer.render_level, (0, None))

        highest_scoring_level = self.level_sampler.get_levels(sampler, sampler["scores"].argmax())
        highest_weighted_level = self.level_sampler.get_levels(
            sampler, self.level_sampler.level_weights(sampler).argmax()
        )
        return {
            "dr_levels": render_batch(ts["dr_last_level_batch"], ctx.env_params),
            "replay_levels": render_batch(ts["replay_last_level_batch"], ctx.env_params),
            "mutation_levels": render_batch(ts["mutation_last_level_batch"], ctx.env_params),
            "highest_scoring_level": ctx.renderer.render_level(
                highest_scoring_level, ctx.env_params
            ),
            "highest_weighted_level": ctx.renderer.render_level(
                highest_weighted_level, ctx.env_params
            ),
        }
