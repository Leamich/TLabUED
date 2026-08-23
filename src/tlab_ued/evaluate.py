"""Evaluate a saved checkpoint on the held-out levels.

    python -m tlab_ued.train --mode eval \
        --checkpoint_directory /workspace/tlab_ued/checkpoints/accel_maxmc/0

Deliberately mirrors upstream's `eval_checkpoint`: it restores the checkpoint
*untargeted* and reads only `loaded["params"]`. That is exactly what the
assignment's own evaluation harness does, which is why extra teacher state in
our checkpoints is harmless and why the `ActorCritic` architecture must not
change.

Writes `results.npz` (states, cum_rewards, episode_lengths, levels) next to a
`solve_rates.csv` under `<out_dir>/results/<run_name>/<seed>/`.
"""

from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, Optional

import jax
import numpy as np
import orbax.checkpoint as ocp


def load_checkpoint_config(checkpoint_directory: str) -> Dict[str, Any]:
    with open(os.path.join(checkpoint_directory, "config.json")) as f:
        return json.load(f)


def restore_params(checkpoint_directory: str, step: Optional[int] = None):
    """Restore just the policy parameters, the upstream way."""
    manager = ocp.CheckpointManager(
        os.path.join(os.path.abspath(checkpoint_directory), "models"),
        item_handlers=ocp.StandardCheckpointHandler(),
    )
    if step is None or step < 0:
        step = manager.latest_step()
    if step is None:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_directory}")
    return manager.restore(step)["params"], step


def evaluate_checkpoint(config: Dict[str, Any]):
    """Run the eval protocol on a checkpoint and persist the results."""
    from tlab_ued.train import build, make_eval_fn

    checkpoint_directory = config["checkpoint_directory"]
    if not checkpoint_directory:
        raise ValueError("--checkpoint_directory is required in eval mode")

    # The checkpoint's own config decides how the env is built; the caller's
    # config decides how it is evaluated (eval_levels, attempts, output dir).
    saved = load_checkpoint_config(checkpoint_directory)
    build_config = {**saved, **{k: config[k] for k in ("eval_levels", "eval_num_attempts")}}
    build_config["out_dir"] = config["out_dir"]

    ctx, teacher = build(build_config)
    eval_fn = make_eval_fn(ctx)

    rng_init, rng_eval = jax.random.split(jax.random.PRNGKey(10000))
    train_state = jax.jit(teacher.create_train_state)(rng_init)
    params, step = restore_params(checkpoint_directory, config.get("checkpoint_to_eval", -1))
    train_state = train_state.replace(params=params)

    states, cum_rewards, episode_lengths = jax.vmap(eval_fn, (0, None))(
        jax.random.split(rng_eval, build_config["eval_num_attempts"]), train_state
    )

    cum_rewards = np.asarray(cum_rewards)
    solve_rates = (cum_rewards > 0).mean(axis=0)
    level_names = list(build_config["eval_levels"])

    save_loc = os.path.join(
        config["out_dir"], "results", str(saved.get("run_name", "unknown")), str(saved.get("seed", 0))
    )
    os.makedirs(save_loc, exist_ok=True)
    np.savez_compressed(
        os.path.join(save_loc, "results.npz"),
        states=np.asarray(states),
        cum_rewards=cum_rewards,
        episode_lengths=np.asarray(episode_lengths),
        levels=level_names,
    )
    with open(os.path.join(save_loc, "solve_rates.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run_name", "seed", "checkpoint_step", "level", "solve_rate", "mean_return"])
        for name, rate, ret in zip(level_names, solve_rates, cum_rewards.mean(axis=0)):
            writer.writerow(
                [saved.get("run_name"), saved.get("seed"), step, name, float(rate), float(ret)]
            )
        writer.writerow(
            [
                saved.get("run_name"),
                saved.get("seed"),
                step,
                "mean",
                float(solve_rates.mean()),
                float(cum_rewards.mean()),
            ]
        )

    print(f"checkpoint step {step}: solve_rate/mean = {solve_rates.mean():.3f}", flush=True)
    for name, rate in zip(level_names, solve_rates):
        print(f"  {name:<18} {rate:.3f}", flush=True)
    return states, cum_rewards, episode_lengths
