"""Measuring the oracle offline: is there anything to select, and who can select it?

A full run is a bad instrument for a question about the oracle. The seed-to-seed
sd of held-out solve rate is 0.07-0.10 (README §8.2) while the effects being
compared are 0.18, so anything smaller than "obvious" needs more seeds than the
GPU budget has. Meanwhile every run leaves eight checkpoints behind, and a
checkpoint is a frozen policy - exactly the object the oracle is trying to
predict. So the question can be asked with 8192 levels of statistical power
instead of 3 seeds, for about an hour of GPU and no training at all.

Three subcommands, in the order they answer things:

    collect    Play N fresh levels k times each with the policy from every
               checkpoint, and write down what happened. This is the only part
               that costs env steps. Every candidate feature is recorded at the
               same time - the exact BFS solve, and the two scalars the frozen
               student produces on the start frame - so that adding a model to
               the ladder later never means collecting again.

    bench      Fit each candidate feature set to that data and score it. Reports
               `ceiling/floor` first, because it is prior to every model: it is
               the learnability of the best 48 of 8192 levels over the average
               one, computed from measurements alone. If that ratio is ~1 late in
               training, the generator has stopped producing anything worth
               selecting and no oracle - however good - can help (README §7.2).
               Only if it is large does `share_of_ceiling` mean anything.

    staleness  Take the oracle *out of* a checkpoint and ask it about a policy
               from later in training, next to the KL between the two policies.
               This is the "when should the teacher stop trusting the model"
               question, answered without running anything new.

What the bench measures is an *upper bound* on the online oracle: it fits to
convergence on a fixed dataset, while the real one takes two Adam steps per
update against a policy moving under it. A feature set that cannot rank levels
here will certainly not rank them there.

    python -m tlab_ued.oracle_bench collect --run sfl_oracle_learnability_level_bfs --seed 0
    python -m tlab_ued.oracle_bench bench --out results/oracle_bench.json
    python -m tlab_ued.oracle_bench staleness --run sfl_oracle_learnability_level_bfs --seed 0
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# The feature sets the ladder compares. `bfs` and `level_bfs` are the privileged
# ceiling - they read an exact solve of the level - and exist here to say how far
# the honest arms are from it. They are never part of a submitted run.
BENCH_FEATURES: Tuple[str, ...] = (
    "level",
    "wide",
    "prop",
    "level_policy",
    "wide_policy",
    "prop_policy",
    "bfs",
    "level_bfs",
)

# Selection size: `sfl_num_levels` (64) minus `oracle_control_levels` (16), i.e.
# the number of levels the real phase actually picks on the oracle's say-so.
TOP_K = 48

LEVEL_FIELDS = ("wall_map", "goal_pos", "agent_pos", "agent_dir")


# === data collection ===


def checkpoint_config(run: str, seed: int, out_dir: str) -> Dict[str, Any]:
    """The config a run was launched with, from beside its checkpoints."""
    path = os.path.join(out_dir, "checkpoints", run, str(seed), "config.json")
    with open(path) as f:
        return json.load(f)


def build_context(config: Dict[str, Any], generator: str):
    """`train.build`, with the level generator swapped for the one being sampled.

    Swapping the generator changes nothing about the network - it is the same
    `ActorCritic` the checkpoint holds - so the same restored policy can be asked
    about `minigrid_walls` levels and about `perfect_maze` ones.
    """
    from tlab_ued.train import build

    return build({**config, "level_generator": generator, "mode": "train"})


def restore_params(step_dir: str) -> Dict[str, Any]:
    """The student's parameters from one checkpoint, as numpy.

    Goes through `buffer_diagnostics.restore_tree` rather than orbax's
    `CheckpointManager.restore`: these checkpoints were written on a GPU and
    carry a device sharding in their metadata, which a target-less restore tries
    to honour. Restoring every leaf as `np.ndarray` sidesteps devices, so the
    same code path works on the pod and on a laptop.
    """
    from tlab_ued.buffer_diagnostics import restore_tree

    tree = restore_tree(step_dir)
    if "params" not in tree:
        raise KeyError(f"{step_dir} has no 'params' - is it a training checkpoint?")
    return tree["params"]


def restore_oracle(step_dir: str) -> Optional[Dict[str, Any]]:
    """The oracle's parameters from one checkpoint, if the teacher kept one."""
    from tlab_ued.buffer_diagnostics import restore_tree

    oracle = (restore_tree(step_dir).get("teacher_state") or {}).get("oracle")
    if oracle is None:
        return None
    params = oracle.get("params")
    # Flax stores {"params": {...}}; orbax gives it back the same way.
    return params if params is None or "params" in params else {"params": params}


def _play_chunk(ctx, train_state, rng, levels, attempts: int):
    """Play `attempts` episodes of each level; return (successes, V(s0), entropy).

    Deliberately the held-out evaluation protocol rather than a training rollout:
    the unwrapped env, a zero LSTM carry, one episode per env, sampled actions.
    That is the same thing `first_episode_success` extracts from the teacher's
    `AutoReplayWrapper` rollouts, so the `p` measured here is the `p` the teacher
    would have measured.

    The two policy scalars come out of the very first forward pass, which is the
    whole cost of the `policy` feature set: what the frozen student's critic says
    about a level before it has seen anything but the 5x5 patch it starts on.
    """
    import jax
    import jax.numpy as jnp

    from tlab_ued.student import ActorCritic
    from tlab_ued.teachers.sfl_accel import per_level, repeat_levels

    env, env_params = ctx.eval_env, ctx.env_params
    batch = repeat_levels(levels, attempts)
    n = jax.tree_util.tree_flatten(batch)[0][0].shape[0]

    rng, rng_reset = jax.random.split(rng)
    obs, state = jax.vmap(env.reset_to_level, (0, 0, None))(
        jax.random.split(rng_reset, n), batch, env_params
    )
    hstate = ActorCritic.initialize_carry((n,))

    # Step 0 twice over: once to read the critic and the policy's entropy, then
    # the scan proper. Cheaper than threading an extra output through the scan.
    x = jax.tree_util.tree_map(lambda a: a[None, ...], (obs, jnp.zeros(n, dtype=bool)))
    _, pi0, value0 = train_state.apply_fn(train_state.params, x, hstate)
    start_value = value0.squeeze(0)
    start_entropy = pi0.entropy().squeeze(0)

    def step(carry, _):
        rng, hstate, obs, state, done, alive, solved = carry
        rng, rng_action, rng_step = jax.random.split(rng, 3)
        x = jax.tree_util.tree_map(lambda a: a[None, ...], (obs, done))
        hstate, pi, _ = train_state.apply_fn(train_state.params, x, hstate)
        action = pi.sample(seed=rng_action).squeeze(0)
        obs, state, reward, done, _ = jax.vmap(env.step, (0, 0, 0, None))(
            jax.random.split(rng_step, n), state, action, env_params
        )
        # The only positive reward in this maze is the terminal one, so "solved"
        # is "was still in its first episode and got paid".
        solved = solved | (alive & (reward > 0))
        return (rng, hstate, obs, state, done, alive & ~done, solved), None

    init = (rng, hstate, obs, state, jnp.zeros(n, dtype=bool), jnp.ones(n, dtype=bool),
            jnp.zeros(n, dtype=bool))
    (_, _, _, _, _, _, solved), _ = jax.lax.scan(
        step, init, None, env_params.max_steps_in_episode
    )

    successes = per_level(solved.astype(jnp.float32), attempts, "mean") * attempts
    return (
        successes,
        per_level(start_value, attempts, "mean"),
        per_level(start_entropy, attempts, "mean"),
    )


def make_player(ctx, attempts: int):
    """A jitted `(train_state, rng, levels) -> (successes, V(s0), entropy)`.

    Built once and reused for every checkpoint: `train_state` is an argument
    rather than a closed-over constant, so the eight checkpoints of a run share
    one compilation instead of paying for eight.
    """
    import jax

    return jax.jit(lambda ts, rng, levels: _play_chunk(ctx, ts, rng, levels, attempts))


def collect_checkpoint(
    ctx,
    teacher,
    step_dir: str,
    num_levels: int,
    attempts: int,
    chunk_levels: int,
    seed: int,
    generator: str,
    play=None,
) -> Dict[str, np.ndarray]:
    """Measure `p` for `num_levels` fresh levels under one checkpoint's policy."""
    import jax
    import jax.numpy as jnp

    from tlab_ued.oracle import bfs_features

    rng = jax.random.PRNGKey(0xB3C4 + seed)
    rng, rng_levels, rng_init = jax.random.split(rng, 3)
    levels = jax.vmap(ctx.sample_random_level)(jax.random.split(rng_levels, num_levels))

    train_state = jax.jit(teacher.create_train_state)(rng_init)
    train_state = train_state.replace(params=restore_params(step_dir))

    play = play or make_player(ctx, attempts)
    successes, values, entropies = [], [], []
    for start in range(0, num_levels, chunk_levels):
        rng, rng_chunk = jax.random.split(rng)
        part = jax.tree_util.tree_map(lambda a: a[start : start + chunk_levels], levels)
        s, v, e = play(train_state, rng_chunk, part)
        successes.append(np.asarray(s))
        values.append(np.asarray(v))
        entropies.append(np.asarray(e))

    scalars = np.asarray(bfs_features(levels, ctx.env_params.max_steps_in_episode))
    out = {
        "successes": np.concatenate(successes).astype(np.float32),
        "attempts": np.full(num_levels, float(attempts), dtype=np.float32),
        "start_value": np.concatenate(values).astype(np.float32),
        "start_entropy": np.concatenate(entropies).astype(np.float32),
        "bfs": scalars.astype(np.float32),
        "generator": np.asarray(generator),
    }
    for field in LEVEL_FIELDS:
        out[f"level_{field}"] = np.asarray(getattr(levels, field))
    out["level_width"] = np.asarray(jnp.asarray(levels.width))
    out["level_height"] = np.asarray(jnp.asarray(levels.height))
    return out


def collect_run(
    run: str,
    seed: int,
    out_dir: str = ".",
    generator: str = "minigrid_walls",
    num_levels: int = 8192,
    attempts: int = 8,
    chunk_levels: int = 256,
    steps: Optional[Sequence[int]] = None,
    bench_dir: str = "results/bench",
) -> List[str]:
    """Every checkpoint of one run against one generator. Writes one npz each."""
    from tlab_ued.buffer_diagnostics import checkpoint_steps

    config = checkpoint_config(run, seed, out_dir)
    ctx, teacher = build_context(config, generator)

    models = os.path.join(out_dir, "checkpoints", run, str(seed), "models")
    wanted = checkpoint_steps(models) if steps is None else sorted(steps)
    os.makedirs(bench_dir, exist_ok=True)

    play = make_player(ctx, attempts)
    written = []
    for step in wanted:
        path = os.path.join(bench_dir, f"{run}_{seed}_{generator}_{step}.npz")
        # Skipping what is already on disk is what makes the whole pipeline
        # resumable: a pod that died mid-collection picks up where it stopped.
        if os.path.exists(path):
            print(f"  {os.path.basename(path)} exists, skipping", flush=True)
            written.append(path)
            continue
        started = time.time()
        data = collect_checkpoint(
            ctx,
            teacher,
            os.path.join(models, str(step)),
            num_levels,
            attempts,
            chunk_levels,
            seed,
            generator,
            play,
        )
        np.savez_compressed(path, step=step, run=run, seed=seed, **data)
        p = data["successes"] / data["attempts"]
        elapsed = time.time() - started
        rate = num_levels * attempts * ctx.env_params.max_steps_in_episode / max(elapsed, 1e-6)
        print(
            f"  step {step:>4}: mean p {p.mean():.3f}  "
            f"learnability {(p * (1 - p)).mean():.4f}  "
            f"[{elapsed:.0f}s, {rate / 1e3:.0f}k steps/s]",
            flush=True,
        )
        written.append(path)
    return written


# === the bench ===


def load_dataset(path: str) -> Dict[str, Any]:
    """One npz back into arrays plus a `Level` pytree."""
    from jaxued.environments.maze import Level

    raw = np.load(path, allow_pickle=False)
    level = Level(
        wall_map=raw["level_wall_map"],
        goal_pos=raw["level_goal_pos"],
        agent_pos=raw["level_agent_pos"],
        agent_dir=raw["level_agent_dir"],
        width=raw["level_width"],
        height=raw["level_height"],
    )
    return {
        "levels": level,
        "successes": raw["successes"],
        "attempts": raw["attempts"],
        "bfs": raw["bfs"],
        "policy": np.stack([raw["start_value"], raw["start_entropy"]], axis=-1),
        "step": int(raw["step"]),
        "run": str(raw["run"]),
        "seed": int(raw["seed"]),
        "generator": str(raw["generator"]),
    }


def scalars_for(features: str, data: Dict[str, Any]) -> np.ndarray:
    """The scalar block a feature set asks for, in the order `OracleNet` expects."""
    from tlab_ued.oracle import parse_features

    tokens = parse_features(features)
    parts = []
    if "bfs" in tokens:
        parts.append(data["bfs"])
    if "policy" in tokens:
        parts.append(data["policy"])
    if not parts:
        return np.zeros((len(data["attempts"]), 1), dtype=np.float32)
    return np.concatenate(parts, axis=-1).astype(np.float32)


def learnability_of(p: np.ndarray) -> np.ndarray:
    return p * (1.0 - p)


def population_stats(p: np.ndarray, top_k: int = TOP_K) -> Dict[str, float]:
    """What is there to be selected, before any model looks at it.

    `ceiling` is the mean learnability of the best `top_k` levels in the sample -
    what a perfect ranker would get. `floor` is the mean over everything - what
    picking at random gets. Their ratio is the entire headroom of the selection
    problem at this point in training, and it is measured, not modelled.
    """
    learn = learnability_of(p)
    ceiling = float(np.sort(learn)[-top_k:].mean())
    floor = float(learn.mean())
    # Both ends collapse to zero once every level is solved, and 0/0 has to be
    # reported as "no advantage available" (1.0) rather than as "selection is
    # worse than chance" (0.0). The distinction is the whole late-training story:
    # a headroom of 1 means the generator stopped offering choices, which is not
    # a failure of any ranker.
    degenerate = ceiling <= 1e-9 and floor <= 1e-9
    return {
        "ceiling": ceiling,
        "floor": floor,
        "headroom": 1.0 if degenerate else ceiling / max(floor, 1e-9),
        "mean_p": float(p.mean()),
        "frac_learnable": float(((p > 0.15) & (p < 0.85)).mean()),
    }


def fit_predictor(
    features: str,
    data: Dict[str, Any],
    train_idx: np.ndarray,
    steps: int = 1000,
    batch_size: int = 256,
    lr: float = 3e-3,
    hidden: int = 64,
    seed: int = 0,
):
    """Fit one feature set to the measured labels; return predicted `p` for all.

    The objective is `oracle.binomial_nll`, the same one the online oracle is
    trained on. The optimiser is not the same - this fits to convergence on a
    fixed dataset - and that is the point: the number this produces is the best
    the feature set could do, so a failure here is a failure of the features
    rather than of the training schedule.
    """
    import jax
    import jax.numpy as jnp
    import optax

    from tlab_ued.oracle import OracleNet, binomial_nll, encode_planes

    planes = np.asarray(encode_planes(data["levels"]))
    scalars = scalars_for(features, data)
    attempts, successes = data["attempts"], data["successes"]

    net = OracleNet(features=features, hidden=hidden)
    rng = jax.random.PRNGKey(seed)
    params = net.init(rng, jnp.asarray(planes[:2]), jnp.asarray(scalars[:2]))
    tx = optax.adam(lr)
    opt_state = tx.init(params)

    tr_planes = jnp.asarray(planes[train_idx])
    tr_scalars = jnp.asarray(scalars[train_idx])
    tr_attempts = jnp.asarray(attempts[train_idx])
    tr_successes = jnp.asarray(successes[train_idx])
    n_train = len(train_idx)

    def loss_fn(params, idx):
        logits = net.apply(params, tr_planes[idx], tr_scalars[idx])
        return binomial_nll(logits, tr_attempts[idx], tr_successes[idx])

    @jax.jit
    def step(carry, key):
        params, opt_state = carry
        idx = jax.random.randint(key, (min(batch_size, n_train),), 0, n_train)
        loss, grads = jax.value_and_grad(loss_fn)(params, idx)
        updates, opt_state = tx.update(grads, opt_state, params)
        return (optax.apply_updates(params, updates), opt_state), loss

    (params, _), losses = jax.lax.scan(
        step, (params, opt_state), jax.random.split(jax.random.PRNGKey(seed + 1), steps)
    )
    predicted = np.asarray(
        jax.nn.sigmoid(net.apply(params, jnp.asarray(planes), jnp.asarray(scalars)))
    )
    return predicted, float(np.asarray(losses)[-50:].mean())


def rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman, via ranks and a plain correlation."""
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = ra.std() * rb.std()
    return float((ra * rb).mean() / denom) if denom > 0 else 0.0


def auc_learnable(predicted_learn: np.ndarray, p: np.ndarray) -> float:
    """AUC for "is this level in the learnable band", ranked by predicted learnability.

    The band is `0.15 < p < 0.85`, i.e. levels the student neither always nor
    never solves. Reported next to the rank correlation because when almost every
    level has `p ~ 1` the correlation is dominated by the ordering of a mass of
    ties, while this asks the question the teacher actually acts on.
    """
    positive = (p > 0.15) & (p < 0.85)
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(np.argsort(predicted_learn)).astype(np.float64)
    return float((order[positive].sum() - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg))


def score_predictions(
    predicted: np.ndarray, p: np.ndarray, test_idx: np.ndarray, top_k: int = TOP_K
) -> Dict[str, float]:
    """How well one fitted model would have selected, on levels it never saw."""
    pred_test, p_test = predicted[test_idx], p[test_idx]
    learn_test = learnability_of(p_test)
    pop = population_stats(p_test, top_k)

    chosen = np.argsort(-learnability_of(pred_test))[:top_k]
    selected = float(learn_test[chosen].mean())
    return {
        "selected_learnability": selected,
        # The two ratios that matter, and they say different things:
        # against the floor, "is this better than not selecting at all";
        # against the ceiling, "how much of what was there did it get".
        "gain": selected / max(pop["floor"], 1e-9),
        "share_of_ceiling": selected / max(pop["ceiling"], 1e-9),
        "rank_corr": rank_corr(pred_test, p_test),
        "auc_learnable": auc_learnable(learnability_of(pred_test), p_test),
        "brier": float(((pred_test - p_test) ** 2).mean()),
        "bias": float((pred_test - p_test).mean()),
    }


def bench_file(
    path: str,
    features: Sequence[str] = BENCH_FEATURES,
    train_frac: float = 0.75,
    top_k: int = TOP_K,
    steps: int = 1000,
) -> Dict[str, Any]:
    """Fit and score every feature set on one collected checkpoint."""
    data = load_dataset(path)
    p = data["successes"] / np.maximum(data["attempts"], 1.0)
    n = len(p)

    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    cut = int(train_frac * n)
    train_idx, test_idx = perm[:cut], perm[cut:]

    out: Dict[str, Any] = {
        "run": data["run"],
        "seed": data["seed"],
        "step": data["step"],
        "generator": data["generator"],
        "num_levels": n,
        "population": population_stats(p[test_idx], top_k),
        "features": {},
    }
    for name in features:
        predicted, loss = fit_predictor(name, data, train_idx, steps=steps)
        scored = score_predictions(predicted, p, test_idx, top_k)
        scored["train_loss"] = loss
        out["features"][name] = scored
    return out


def bench_all(
    bench_dir: str = "results/bench",
    features: Sequence[str] = BENCH_FEATURES,
    top_k: int = TOP_K,
    steps: int = 1000,
    pattern: str = "*.npz",
) -> Dict[str, Any]:
    paths = sorted(glob.glob(os.path.join(bench_dir, pattern)))
    if not paths:
        raise FileNotFoundError(f"no collected data in {bench_dir}; run `collect` first")
    results = []
    for path in paths:
        print(f"  {os.path.basename(path)}", flush=True)
        results.append(bench_file(path, features, top_k=top_k, steps=steps))
    return {"top_k": top_k, "results": results}


def pick_arm(report: Dict[str, Any], min_step: int = 1, margin: float = 0.02) -> Dict[str, Any]:
    """The preregistered rule that turns the bench into one `--oracle_features`.

    Written down before the data existed so that the choice is a rule and not a
    look at the answer: highest mean `share_of_ceiling` over checkpoints past
    `min_step` and over every generator, restricted to feature sets that are
    honest (no `bfs`) and that the teacher can actually run (no `policy`, which
    is bench-only until it earns the plumbing). Ties inside `margin` go to the
    simpler model, in the order `wide`, `prop`, `level`.
    """
    from tlab_ued.oracle import parse_features

    runnable = [
        f
        for f in BENCH_FEATURES
        if "bfs" not in parse_features(f) and "policy" not in parse_features(f)
    ]
    preference = ["wide", "prop", "level"]

    scores: Dict[str, List[float]] = {f: [] for f in runnable}
    for entry in report["results"]:
        if entry["step"] < min_step:
            continue
        for name in runnable:
            value = entry["features"].get(name, {}).get("share_of_ceiling")
            if value is not None and np.isfinite(value):
                scores[name].append(float(value))

    means = {f: (float(np.mean(v)) if v else float("-inf")) for f, v in scores.items()}
    best = max(means.values())
    contenders = [f for f, m in means.items() if m >= best - margin]
    ordered = sorted(contenders, key=lambda f: preference.index(f) if f in preference else 99)
    return {"arm": ordered[0], "means": means, "contenders": contenders}


# === staleness: the oracle from checkpoint t, asked about the policy at t + delta ===


def staleness_run(
    run: str,
    seed: int,
    out_dir: str = ".",
    bench_dir: str = "results/bench",
    generator: str = "minigrid_walls",
    kl_levels: int = 256,
) -> Dict[str, Any]:
    """Cross every checkpoint's oracle with every checkpoint's measured `p`.

    Two matrices come out of one pass:

    * `share_of_ceiling[t][t+delta]` - how well the oracle *as it was* at update
      t would have selected levels for the policy as it is at t + delta. The
      diagonal is the fair baseline; the decay along a row is what "stale" costs.
    * `kl[t][t+delta]` - how far the policy actually moved, so that decay can be
      read against drift rather than against wall-clock updates.

    If the rows are flat, the teacher has no reason to refresh the oracle on a
    schedule, and a KL-gated phase is solving a problem that does not exist.
    """
    import jax

    from tlab_ued.buffer_diagnostics import checkpoint_steps
    from tlab_ued.oracle import OracleNet, encode_planes, scalar_width

    config = checkpoint_config(run, seed, out_dir)
    features = str(config.get("oracle_features", "level_bfs"))
    models = os.path.join(out_dir, "checkpoints", run, str(seed), "models")
    steps = checkpoint_steps(models)

    datasets = {}
    for step in steps:
        path = os.path.join(bench_dir, f"{run}_{seed}_{generator}_{step}.npz")
        if os.path.exists(path):
            datasets[step] = load_dataset(path)
    if not datasets:
        raise FileNotFoundError(f"no collected data for {run}/{seed} in {bench_dir}")

    net = OracleNet(features=features, hidden=int(config.get("oracle_hidden", 64)))
    grid: Dict[str, Dict[str, Any]] = {}
    for step in steps:
        params = restore_oracle(os.path.join(models, str(step)))
        if params is None:
            continue
        row = {}
        for later, data in datasets.items():
            if later < step:
                continue
            planes = encode_planes(data["levels"])
            scalars = scalars_for(features, data)
            assert scalars.shape[-1] == scalar_width(features)
            predicted = np.asarray(jax.nn.sigmoid(net.apply(params, planes, scalars)))
            p = data["successes"] / np.maximum(data["attempts"], 1.0)
            idx = np.arange(len(p))
            row[str(later)] = score_predictions(predicted, p, idx)
        grid[str(step)] = row

    return {
        "run": run,
        "seed": seed,
        "features": features,
        "steps": steps,
        "oracle_grid": grid,
        "kl": policy_drift(run, seed, out_dir, generator, kl_levels),
    }


def policy_drift(
    run: str, seed: int, out_dir: str, generator: str, num_levels: int
) -> Dict[str, Dict[str, float]]:
    """Mean KL between the policy at each checkpoint and every later one.

    Measured on one shared rollout: the *earliest* policy plays `num_levels`
    levels and the observations it visited are replayed through every policy.
    Sharing the states is what makes the numbers comparable - a KL computed on
    each policy's own trajectories would confound "the policy changed" with "it
    goes somewhere else now".
    """
    import jax
    import jax.numpy as jnp

    from tlab_ued.buffer_diagnostics import checkpoint_steps
    from tlab_ued.student import ActorCritic

    config = checkpoint_config(run, seed, out_dir)
    ctx, teacher = build_context(config, generator)
    models = os.path.join(out_dir, "checkpoints", run, str(seed), "models")
    steps = checkpoint_steps(models)

    rng = jax.random.PRNGKey(0xD817)
    rng, rng_levels, rng_init, rng_roll = jax.random.split(rng, 4)
    levels = jax.vmap(ctx.sample_random_level)(jax.random.split(rng_levels, num_levels))
    train_state = jax.jit(teacher.create_train_state)(rng_init)

    # The last checkpoint drives, because a fully trained policy visits states
    # both policies have opinions about; an untrained one mostly spins in place.
    driver = train_state.replace(params=restore_params(os.path.join(models, str(steps[-1]))))
    obs_seq, done_seq, mask = _record_states(ctx, driver, rng_roll, levels)

    def logits_of(params):
        _, pi, _ = train_state.apply_fn(
            params, (obs_seq, done_seq), ActorCritic.initialize_carry((num_levels,))
        )
        return pi.logits

    logits = {}
    for step in steps:
        logits[step] = logits_of(restore_params(os.path.join(models, str(step))))

    weight = mask.astype(jnp.float32)
    total = jnp.maximum(weight.sum(), 1.0)
    out: Dict[str, Dict[str, float]] = {}
    for i, step in enumerate(steps):
        log_p = jax.nn.log_softmax(logits[step], axis=-1)
        p = jnp.exp(log_p)
        row = {}
        for later in steps[i:]:
            log_q = jax.nn.log_softmax(logits[later], axis=-1)
            kl = (p * (log_p - log_q)).sum(axis=-1)
            row[str(later)] = float((kl * weight).sum() / total)
        out[str(step)] = row
    return out


def _record_states(ctx, train_state, rng, levels):
    """Roll one policy out and keep (obs, done, still-in-first-episode)."""
    import jax
    import jax.numpy as jnp

    from tlab_ued.student import ActorCritic

    env, env_params = ctx.eval_env, ctx.env_params
    n = jax.tree_util.tree_flatten(levels)[0][0].shape[0]
    rng, rng_reset = jax.random.split(rng)
    obs, state = jax.vmap(env.reset_to_level, (0, 0, None))(
        jax.random.split(rng_reset, n), levels, env_params
    )

    def step(carry, _):
        rng, hstate, obs, state, done, alive = carry
        rng, rng_action, rng_step = jax.random.split(rng, 3)
        x = jax.tree_util.tree_map(lambda a: a[None, ...], (obs, done))
        hstate, pi, _ = train_state.apply_fn(train_state.params, x, hstate)
        action = pi.sample(seed=rng_action).squeeze(0)
        next_obs, state, _, next_done, _ = jax.vmap(env.step, (0, 0, 0, None))(
            jax.random.split(rng_step, n), state, action, env_params
        )
        return (rng, hstate, next_obs, state, next_done, alive & ~next_done), (obs, done, alive)

    init = (rng, ActorCritic.initialize_carry((n,)), obs, state,
            jnp.zeros(n, dtype=bool), jnp.ones(n, dtype=bool))
    _, (obs_seq, done_seq, mask) = jax.lax.scan(
        step, init, None, env_params.max_steps_in_episode
    )
    return obs_seq, done_seq, mask


# === reporting ===


def format_bench(report: Dict[str, Any]) -> str:
    """Two tables: what there was to select, then who selected it."""
    lines = ["", "Population: what is there to be selected (measured, model-free)", ""]
    lines.append("  run/gen                          ckpt   mean_p   learnable   floor    ceiling  headroom")
    for entry in report["results"]:
        pop = entry["population"]
        tag = f"{entry['run'][:20]}/{entry['generator'][:9]}"
        lines.append(
            f"  {tag:<32} {entry['step']:>4}   {pop['mean_p']:.3f}    {pop['frac_learnable']:.3f}   "
            f"{pop['floor']:.4f}   {pop['ceiling']:.4f}   {pop['headroom']:>6.1f}"
        )

    lines += ["", "Feature sets: share of that ceiling actually captured (held-out levels)", ""]
    names = list(report["results"][0]["features"]) if report["results"] else []
    lines.append("  ckpt  gen        " + "".join(f"{n:>14}" for n in names))
    for entry in report["results"]:
        row = "".join(
            f"{entry['features'][n]['share_of_ceiling']:>14.3f}" for n in names
        )
        lines.append(f"  {entry['step']:>4}  {entry['generator'][:9]:<9} {row}")

    lines += ["", "  (same, as rank correlation with measured p)", ""]
    lines.append("  ckpt  gen        " + "".join(f"{n:>14}" for n in names))
    for entry in report["results"]:
        row = "".join(f"{entry['features'][n]['rank_corr']:>14.3f}" for n in names)
        lines.append(f"  {entry['step']:>4}  {entry['generator'][:9]:<9} {row}")
    return "\n".join(lines)


def format_staleness(report: Dict[str, Any]) -> str:
    lines = [
        "",
        f"Oracle staleness ({report['run']}, seed {report['seed']}, features {report['features']})",
        "",
        "  share_of_ceiling: rows = oracle from this checkpoint, columns = policy measured then",
        "",
    ]
    steps = [s for s in report["steps"] if str(s) in report["oracle_grid"]]
    lines.append("  oracle\\policy " + "".join(f"{s:>9}" for s in steps))
    for step in steps:
        row = report["oracle_grid"][str(step)]
        cells = "".join(
            f"{row[str(s)]['share_of_ceiling']:>9.3f}" if str(s) in row else f"{'-':>9}"
            for s in steps
        )
        lines.append(f"  {step:>12} {cells}")

    lines += ["", "  mean KL(pi_row || pi_col) on shared states", ""]
    lines.append("  policy\\policy " + "".join(f"{s:>9}" for s in steps))
    for step in steps:
        row = report["kl"].get(str(step), {})
        cells = "".join(
            f"{row[str(s)]:>9.3f}" if str(s) in row else f"{'-':>9}" for s in steps
        )
        lines.append(f"  {step:>12} {cells}")
    return "\n".join(lines)


def write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=float)


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out_dir", default=".", help="where checkpoints/ lives")
    common.add_argument("--bench_dir", default="results/bench")

    c = sub.add_parser("collect", parents=[common], help="measure p under each checkpoint")
    c.add_argument("--run", required=True)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--generator", default="minigrid_walls")
    c.add_argument("--num_levels", type=int, default=8192)
    c.add_argument("--attempts", type=int, default=8)
    c.add_argument("--chunk_levels", type=int, default=256)
    c.add_argument("--steps", type=int, nargs="*", default=None)

    b = sub.add_parser("bench", parents=[common], help="fit and score the feature sets")
    b.add_argument("--features", nargs="*", default=list(BENCH_FEATURES))
    b.add_argument("--top_k", type=int, default=TOP_K)
    b.add_argument("--fit_steps", type=int, default=1000)
    b.add_argument("--pattern", default="*.npz")
    b.add_argument("--out", default="results/oracle_bench.json")

    s = sub.add_parser("staleness", parents=[common], help="oracle from t vs policy at t+delta")
    s.add_argument("--run", required=True)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--generator", default="minigrid_walls")
    s.add_argument("--kl_levels", type=int, default=256)
    s.add_argument("--out", default="results/oracle_staleness.json")

    p = sub.add_parser("pick", help="print the arm the preregistered rule selects")
    p.add_argument("--report", default="results/oracle_bench.json")
    p.add_argument("--min_step", type=int, default=1)

    args = parser.parse_args(argv)

    if args.command == "collect":
        collect_run(
            args.run,
            args.seed,
            args.out_dir,
            args.generator,
            args.num_levels,
            args.attempts,
            args.chunk_levels,
            args.steps,
            args.bench_dir,
        )
        return {}

    if args.command == "bench":
        report = bench_all(args.bench_dir, args.features, args.top_k, args.fit_steps, args.pattern)
        report["pick"] = pick_arm(report)
        print(format_bench(report), flush=True)
        print(f"\n  preregistered pick: {report['pick']['arm']}\n", flush=True)
        write_json(args.out, report)
        with open(os.path.splitext(args.out)[0] + ".md", "w") as f:
            f.write("# Oracle bench\n\n```\n" + format_bench(report) + "\n```\n")
        return report

    if args.command == "staleness":
        report = staleness_run(
            args.run, args.seed, args.out_dir, args.bench_dir, args.generator, args.kl_levels
        )
        print(format_staleness(report), flush=True)
        write_json(args.out, report)
        return report

    with open(args.report) as f:
        report = json.load(f)
    pick = report.get("pick") or pick_arm(report, args.min_step)
    print(pick["arm"])
    return pick


if __name__ == "__main__":
    main()
