"""Domain randomisation - the reference teacher.

Faithful port of JaxUED `examples/maze_dr.py` @ 0f8f128.

DR is *not* "PLR with the replay probability turned off": it wraps the env in
`AutoResetWrapper`, so an episode that finishes is replaced by a freshly sampled
random level mid-rollout, and the env state, observation and LSTM hidden state
persist across updates instead of being reset every update. That difference is
why it lives in its own teacher rather than as a flag on PLR.
"""

from __future__ import annotations

from typing import Any, Dict, List

import chex
import jax

from jaxued.wrappers import AutoResetWrapper

from tlab_ued import student
from tlab_ued.teachers.base import BranchFn, Teacher, TrainState


class DRTeacher(Teacher):
    name = "dr"

    @classmethod
    def wrap_env(cls, config: Dict[str, Any], base_env, sample_random_level):
        # Resample a new random level whenever an episode ends.
        return AutoResetWrapper(base_env, sample_random_level)

    def create_train_state(self, rng: chex.PRNGKey) -> TrainState:
        """Reproduces maze_dr's rng flow exactly.

        Upstream uses the incoming rng for the placeholder reset, *then* splits:
        one half initialises the network, the other seeds the starting batch of
        levels. maze_plr does not split at all, which is why the two baselines
        start from different parameters at the same seed.
        """
        ctx = self.ctx
        rng_rest, rng_network = jax.random.split(rng)
        network, params, tx = student.make_network_and_tx(
            self.config,
            ctx.env,
            ctx.env_params,
            ctx.sample_random_level,
            rng,
            init_rng=rng_network,
        )
        return TrainState.create(
            apply_fn=network.apply,
            params=params,
            tx=tx,
            update_count=0,
            teacher_state=self.init_teacher_state(rng_rest),
        )

    def init_teacher_state(self, rng: chex.PRNGKey) -> Any:
        """DR's memory is the env it is midway through, not a level buffer."""
        ctx = self.ctx
        rng_levels, rng_reset = jax.random.split(rng)
        new_levels = ctx.sample_levels(rng_levels)
        init_obs, init_env_state = ctx.reset_to_levels(rng_reset, new_levels)
        return {
            "last_hstate": ctx.initial_carry(),
            "last_obs": init_obs,
            "last_env_state": init_env_state,
        }

    @property
    def branches(self) -> List[BranchFn]:
        return [self.on_random_levels]

    def on_random_levels(self, rng: chex.PRNGKey, train_state: TrainState):
        """One PPO update on the continuing stream of random levels."""
        ctx = self.ctx
        ts = train_state.teacher_state

        carry, traj, advantages, targets = ctx.rollout(
            rng, train_state, ts["last_hstate"], ts["last_obs"], ts["last_env_state"]
        )
        rng, train_state, hstate, last_obs, last_env_state, _ = carry
        obs, actions, rewards, dones, log_probs, values, info = traj

        (rng, train_state), losses = ctx.ppo_update(
            rng,
            train_state,
            ts["last_hstate"],
            (obs, actions, dones, log_probs, values, targets, advantages),
            update_grad=True,
        )

        metrics = {"losses": jax.tree_util.tree_map(lambda x: x.mean(), losses)}
        train_state = train_state.replace(
            teacher_state={
                "last_hstate": hstate,
                "last_obs": last_obs,
                "last_env_state": last_env_state,
            }
        )
        return (rng, train_state), metrics

    def log_dict(self, train_state: TrainState) -> Dict[str, Dict[str, Any]]:
        return {"log": {}, "info": {"num_dr_updates": train_state.update_count}}
