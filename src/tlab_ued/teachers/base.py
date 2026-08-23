"""The teacher plugin interface.

A *teacher* owns everything on the environment-design side of the loop:
  - which env wrapper the student trains under (`wrap_env`),
  - what memory it keeps across updates (`teacher_state` inside `TrainState`),
  - what happens on a given update (`branches` + `select_branch`),
  - what it reports (`log_dict`, `media`).

The student - `tlab_ued.student` - is frozen and shared by every teacher.

Why this shape: the two upstream examples differ in more than a score function.
`maze_dr.py` wraps the env in `AutoResetWrapper` and carries the env state across
updates; `maze_plr.py` wraps it in `AutoReplayWrapper` and resets to a chosen
batch of levels each update, and it dispatches between up to three branches with
`jax.lax.switch`. Both fit under this interface, which means a new idea does too.

PRNG discipline: branch bodies keep upstream's exact `jax.random.split` pattern.
Changing the number or order of splits changes the run even when the logic is
equivalent, so parity with upstream (see `tlab_ued.parity`) depends on it.
"""

from __future__ import annotations

import abc
import dataclasses
from typing import Any, Callable, Dict, List, Sequence, Tuple

import chex
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState as BaseTrainState

from tlab_ued import levels as level_registry
from tlab_ued import student
from tlab_ued.scoring import RolloutSignals, compute_score
from tlab_ued.student import ActorCritic


class TrainState(BaseTrainState):
    """Student state (params, opt_state) plus opaque teacher memory.

    `teacher_state` is whatever pytree the active teacher needs - the PLR level
    buffer, DR's carried env state, or a future teacher's auxiliary model. Its
    structure is deliberately not fixed here.

    Checkpoint compatibility: `params` stays at the top level and keeps the
    upstream `ActorCritic` structure, which is all the assignment's evaluation
    harness reads (`loaded_checkpoint["params"]`). Teacher memory riding along in
    the same checkpoint does not affect that.
    """

    update_count: int = 0
    teacher_state: Any = None


@dataclasses.dataclass
class TrainContext:
    """Everything a branch body needs, assembled once by `tlab_ued.train`."""

    config: Dict[str, Any]
    env: Any  # wrapped env used for training rollouts
    eval_env: Any  # unwrapped Maze, used for evaluation
    env_params: Any
    renderer: Any
    sample_random_level: Callable
    mutate_level: Callable

    # --- convenience accessors -------------------------------------------------
    @property
    def num_envs(self) -> int:
        return self.config["num_train_envs"]

    @property
    def num_steps(self) -> int:
        return self.config["num_steps"]

    def initial_carry(self, batch_dims: Sequence[int] = None):
        return ActorCritic.initialize_carry(tuple(batch_dims or (self.num_envs,)))

    def reset_to_levels(self, rng_reset: chex.PRNGKey, levels):
        """vmapped reset onto a batch of levels (upstream's exact call)."""
        return jax.vmap(self.env.reset_to_level, in_axes=(0, 0, None))(
            jax.random.split(rng_reset, self.num_envs), levels, self.env_params
        )

    def sample_levels(self, rng_levels: chex.PRNGKey):
        return jax.vmap(self.sample_random_level)(jax.random.split(rng_levels, self.num_envs))

    # --- student calls (frozen behaviour, consumes rng exactly like upstream) ---
    def rollout(self, rng, train_state, init_hstate, init_obs, init_env_state):
        """Collect a rollout and compute GAE. Returns (carry, traj, advantages, targets)."""
        (
            (rng, train_state, hstate, last_obs, last_env_state, last_value),
            traj,
        ) = student.sample_trajectories_rnn(
            rng,
            self.env,
            self.env_params,
            train_state,
            init_hstate,
            init_obs,
            init_env_state,
            self.num_envs,
            self.num_steps,
        )
        obs, actions, rewards, dones, log_probs, values, info = traj
        advantages, targets = student.compute_gae(
            self.config["gamma"], self.config["gae_lambda"], last_value, values, rewards, dones
        )
        carry = (rng, train_state, hstate, last_obs, last_env_state, last_value)
        return carry, traj, advantages, targets

    def ppo_update(self, rng, train_state, init_hstate, batch, update_grad: bool):
        """PPO update with the frozen hyperparameters."""
        return student.update_actor_critic_rnn(
            rng,
            train_state,
            init_hstate,
            batch,
            self.num_envs,
            self.num_steps,
            self.config["num_minibatches"],
            self.config["epoch_ppo"],
            self.config["clip_eps"],
            self.config["entropy_coeff"],
            self.config["critic_coeff"],
            update_grad=update_grad,
        )

    def score(self, traj, advantages, targets, max_returns, levels, **extras) -> chex.Array:
        """Run the configured score function over a batch of rollouts."""
        obs, actions, rewards, dones, log_probs, values, info = traj
        signals = RolloutSignals(
            dones=dones,
            values=values,
            rewards=rewards,
            advantages=advantages,
            targets=targets,
            log_probs=log_probs,
            actions=actions,
            max_returns=max_returns,
            levels=levels,
            extras=extras or None,
        )
        return compute_score(self.config, signals)


BranchFn = Callable[[chex.PRNGKey, TrainState], Tuple[Tuple[chex.PRNGKey, TrainState], Dict]]


class Teacher(abc.ABC):
    """Base class for level-selection strategies.

    Subclasses are registered in `tlab_ued.teachers.TEACHERS` and selected with
    `--teacher <name>`.
    """

    name: str = "base"

    def __init__(self, ctx: TrainContext):
        self.ctx = ctx
        self.config = ctx.config

    # --- environment ----------------------------------------------------------
    @classmethod
    def wrap_env(cls, config: Dict[str, Any], base_env, sample_random_level):
        """Wrap the raw `Maze` for training. Called before the context exists."""
        raise NotImplementedError

    # --- initialisation -------------------------------------------------------
    def create_train_state(self, rng: chex.PRNGKey) -> TrainState:
        """Build the initial train state.

        Default follows `maze_plr.py`: the network is initialised with the
        incoming rng directly. `maze_dr.py` splits first, so `DRTeacher`
        overrides this - the two upstream baselines genuinely start from
        different parameters for the same seed, and we reproduce that.
        """
        network, params, tx = self.make_network(rng)
        return TrainState.create(
            apply_fn=network.apply,
            params=params,
            tx=tx,
            update_count=0,
            teacher_state=self.init_teacher_state(rng),
        )

    def make_network(self, rng: chex.PRNGKey):
        return student.make_network_and_tx(
            self.config, self.ctx.env, self.ctx.env_params, self.ctx.sample_random_level, rng
        )

    @abc.abstractmethod
    def init_teacher_state(self, rng: chex.PRNGKey) -> Any:
        """The teacher's initial memory (level buffer, carried env state, ...)."""

    # --- the update ------------------------------------------------------------
    @property
    @abc.abstractmethod
    def branches(self) -> List[BranchFn]:
        """Candidate update bodies, indexed by `select_branch`."""

    def select_branch(self, rng: chex.PRNGKey, train_state: TrainState) -> chex.Array:
        """Which branch to run this update. Single-branch teachers can ignore it."""
        return jnp.zeros((), dtype=jnp.int32)

    # --- reporting -------------------------------------------------------------
    def log_dict(self, train_state: TrainState) -> Dict[str, Dict[str, Any]]:
        """Scalars for wandb/CSV.

        Returns `{"log": {...logged...}, "info": {...used for control flow...}}`,
        matching upstream's `train_state_to_log_dict`.
        """
        return {"log": {}, "info": {}}

    def media(self, train_state: TrainState) -> Dict[str, Any]:
        """Rendered images keyed by name. Honour `config["log_media"]`."""
        return {}


def make_level_sampler(config: Dict[str, Any]):
    """The upstream `LevelSampler`, configured from the teacher-side flags."""
    from jaxued.level_sampler import LevelSampler

    return LevelSampler(
        capacity=config["level_buffer_capacity"],
        replay_prob=config["replay_prob"],
        staleness_coeff=config["staleness_coeff"],
        minimum_fill_ratio=config["minimum_fill_ratio"],
        prioritization=config["prioritization"],
        prioritization_params={"temperature": config["temperature"], "k": config["topk_k"]},
        duplicate_check=config["buffer_duplicate_check"],
    )


def build_level_fns(config: Dict[str, Any], base_env):
    """Resolve the configured level generator and mutator."""
    return (
        level_registry.get_generator(config, base_env),
        level_registry.get_mutator(config, base_env),
    )
