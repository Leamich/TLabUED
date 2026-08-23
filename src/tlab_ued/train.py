"""Teacher-agnostic training loop.

Structurally this is JaxUED's `examples/maze_plr.py`, with the environment-design
decisions lifted out into `tlab_ued.teachers` and the score function into
`tlab_ued.scoring`. What stays here is everything that must be identical across
methods for the comparison to mean anything: the student, the budget, the
evaluation protocol, checkpointing and logging.

    python -m tlab_ued.train --preset accel --seed 0
    python -m tlab_ued.train --preset plr --seed 1 --out_dir /workspace/tlab_ued
    python -m tlab_ued.train --teacher dr --smoke

The outer loop runs `num_updates // eval_freq` iterations; each one is a single
jitted call that does `eval_freq` PPO updates and then evaluates on the held-out
levels. `--checkpoint_save_interval` counts *those* iterations, so the
assignment's `17` means one checkpoint every 4250 updates, ~7 per full run.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Tuple

import chex
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp

from jaxued.environments import Maze, MazeRenderer
from jaxued.environments.maze import Level

from tlab_ued import levels as level_registry
from tlab_ued import student
from tlab_ued.config import assert_student_frozen, from_args
from tlab_ued.logging_utils import Logger, checkpoint_dir
from tlab_ued.student import ActorCritic
from tlab_ued.teachers import TrainContext, TrainState, get_teacher_cls


def build(config: Dict[str, Any]) -> Tuple[TrainContext, Any]:
    """Assemble the environment, the level functions and the teacher.

    Shared by training, evaluation and the parity check so that all three see
    exactly the same objects.
    """
    base_env = Maze(
        max_height=13,
        max_width=13,
        agent_view_size=config["agent_view_size"],
        normalize_obs=True,
    )
    sample_random_level = level_registry.get_generator(config, base_env)
    mutate_level = level_registry.get_mutator(config, base_env)
    renderer = MazeRenderer(base_env, tile_size=8)

    teacher_cls = get_teacher_cls(config)
    env = teacher_cls.wrap_env(config, base_env, sample_random_level)

    ctx = TrainContext(
        config=config,
        env=env,
        eval_env=base_env,
        env_params=env.default_params,
        renderer=renderer,
        sample_random_level=sample_random_level,
        mutate_level=mutate_level,
    )
    return ctx, teacher_cls(ctx)


def make_eval_fn(ctx: TrainContext):
    """Evaluate the policy on the held-out levels named by `--eval_levels`.

    These levels are for measurement only: docs/TASK.md forbids using them for
    training, tuning or as generation templates.
    """
    config = ctx.config

    def eval_fn(rng: chex.PRNGKey, train_state: TrainState):
        rng, rng_reset = jax.random.split(rng)
        eval_levels = Level.load_prefabs(config["eval_levels"])
        num_levels = len(config["eval_levels"])
        init_obs, init_env_state = jax.vmap(ctx.eval_env.reset_to_level, (0, 0, None))(
            jax.random.split(rng_reset, num_levels), eval_levels, ctx.env_params
        )
        states, rewards, episode_lengths = student.evaluate_rnn(
            rng,
            ctx.eval_env,
            ctx.env_params,
            train_state,
            ActorCritic.initialize_carry((num_levels,)),
            init_obs,
            init_env_state,
            ctx.env_params.max_steps_in_episode,
        )
        mask = jnp.arange(ctx.env_params.max_steps_in_episode)[..., None] < episode_lengths
        cum_rewards = (rewards * mask).sum(axis=0)
        return states, cum_rewards, episode_lengths

    return eval_fn


def setup_checkpointing(config: Dict[str, Any]) -> ocp.CheckpointManager:
    """Orbax manager writing to `<out_dir>/checkpoints/<run_name>/<seed>/models`.

    The path layout matches upstream because that is what the assignment asks to
    be submitted, and `config.json` is written beside it so a checkpoint can be
    evaluated without knowing how it was produced.
    """
    save_dir = checkpoint_dir(config)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(dict(config), f, indent=True, default=str)
    return ocp.CheckpointManager(
        os.path.join(save_dir, "models"),
        options=ocp.CheckpointManagerOptions(
            save_interval_steps=config["checkpoint_save_interval"],
            max_to_keep=config["max_number_of_checkpoints"],
        ),
    )


def main(config: Dict[str, Any]):
    """Train one run. Returns the final train state."""
    if config["mode"] == "eval":
        from tlab_ued.evaluate import evaluate_checkpoint

        return evaluate_checkpoint(config)

    assert_student_frozen(config)
    ctx, teacher = build(config)
    eval_fn = make_eval_fn(ctx)
    log_media = config.get("log_media", "all")

    def train_step(carry, _):
        """One PPO update, on levels the teacher chose."""
        rng, train_state = carry
        # Upstream splits here in every example, even where the second key goes
        # unused; keeping the split keeps the PRNG stream aligned with theirs.
        rng, rng_branch = jax.random.split(rng)
        branches = teacher.branches
        if len(branches) == 1:
            (rng, train_state), metrics = branches[0](rng, train_state)
        else:
            (rng, train_state), metrics = jax.lax.switch(
                teacher.select_branch(rng_branch, train_state), branches, rng, train_state
            )
        train_state = train_state.replace(update_count=train_state.update_count + 1)
        return (rng, train_state), metrics

    @jax.jit
    def train_and_eval_step(runner_state, _):
        """`eval_freq` updates followed by one evaluation on the held-out levels."""
        (rng, train_state), metrics = jax.lax.scan(
            train_step, runner_state, None, config["eval_freq"]
        )

        rng, rng_eval = jax.random.split(rng)
        states, cum_rewards, episode_lengths = jax.vmap(eval_fn, (0, None))(
            jax.random.split(rng_eval, config["eval_num_attempts"]), train_state
        )

        # Averaged over attempts: a level counts as solved when the episode
        # returned any positive reward.
        out = {k: jax.tree_util.tree_map(lambda x: x.mean(), v) for k, v in metrics.items()}
        out["update_count"] = train_state.update_count
        out["eval_solve_rates"] = jnp.where(cum_rewards > 0, 1.0, 0.0).mean(axis=0)
        out["eval_returns"] = cum_rewards.mean(axis=0)

        states, episode_lengths = jax.tree_util.tree_map(
            lambda x: x[0], (states, episode_lengths)  # first attempt only, for rendering
        )
        out["eval_ep_lengths"] = episode_lengths

        if log_media != "none":
            out["media"] = teacher.media(train_state)
        if log_media == "all":
            images = jax.vmap(jax.vmap(ctx.renderer.render_state, (0, None)), (0, None))(
                states, ctx.env_params
            )
            # wandb wants the colour channel before the image dims for video.
            out["eval_animation"] = (images.transpose(0, 1, 4, 2, 3), episode_lengths)

        return (rng, train_state), out

    rng = jax.random.PRNGKey(config["seed"])
    rng_init, rng_train = jax.random.split(rng)
    train_state = jax.jit(teacher.create_train_state)(rng_init)

    checkpoint_manager = (
        setup_checkpointing(config) if config["checkpoint_save_interval"] > 0 else None
    )

    start_eval_step = 0
    if config.get("resume") and checkpoint_manager is not None:
        latest = checkpoint_manager.latest_step()
        if latest is not None:
            train_state = checkpoint_manager.restore(
                latest, args=ocp.args.StandardRestore(train_state)
            )
            start_eval_step = latest + 1
            # The rng stream cannot be restored from a checkpoint, so a resumed
            # run is statistically equivalent but not bit-identical to an
            # uninterrupted one. Folding in the step keeps it from replaying the
            # same randomness.
            rng_train = jax.random.fold_in(rng_train, start_eval_step)
            print(f"Resumed from eval step {latest} ({train_state.update_count} updates)", flush=True)

    logger = Logger(config, tags=[config["teacher"], config["score_function"]])
    runner_state = (rng_train, train_state)
    total_eval_steps = config["num_updates"] // config["eval_freq"]

    try:
        for eval_step in range(start_eval_step, total_eval_steps):
            start_time = time.time()
            runner_state, metrics = train_and_eval_step(runner_state, None)
            metrics = jax.block_until_ready(metrics)
            metrics["time_delta"] = time.time() - start_time
            scalars = logger.log_eval(metrics, teacher.log_dict(runner_state[1]))
            eta_h = (total_eval_steps - eval_step - 1) * metrics["time_delta"] / 3600
            print(
                f"[{eval_step + 1}/{total_eval_steps}] "
                f"solve_rate/mean={scalars['solve_rate/mean']:.3f} "
                f"sps={scalars['sps']:.0f} eta={eta_h:.2f}h",
                flush=True,
            )
            if checkpoint_manager is not None:
                checkpoint_manager.save(eval_step, args=ocp.args.StandardSave(runner_state[1]))
                checkpoint_manager.wait_until_finished()
    finally:
        logger.finish()

    # Marker the sweep runner uses to tell "finished" from "killed mid-run".
    with open(os.path.join(logger.dir, "DONE"), "w") as f:
        f.write(f"{runner_state[1].update_count} updates\n")
    return runner_state[1]


if __name__ == "__main__":
    main(from_args())
