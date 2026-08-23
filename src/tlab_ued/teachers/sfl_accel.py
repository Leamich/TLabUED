"""SFL-ACCEL - ACCEL driven by learnability instead of MaxMC.

Sampling For Learnability (Rutherford et al., 2024, arXiv:2408.15099) scores a
level by `p(1-p)`, where `p` is the student's success rate on it: levels it
always solves and levels it never solves are both worthless, and the ones in
between are where a gradient step has somewhere to go. The paper's method
periodically evaluates a large batch of *fresh random* levels at a fixed policy
and trains on the most learnable ones; ACCEL instead evolves levels by mutating
whatever its regret proxy ranked highest. This teacher is the obvious cross:
learnability decides what enters the buffer, and ACCEL's mutation operator
evolves it from there.

Three things follow from `p` being unmeasurable in a single rollout - one
episode per level gives `p` in {0,1}, whose `p(1-p)` is 0 either way:

1. **Levels are evaluated with `--sfl_num_attempts` attempts each.** The 32 envs
   of an update carry `32/k` distinct levels, `k` times each, so the update
   costs exactly what ACCEL's costs and returns a real `p_hat` in `{0, 1/k, ...}`.
   Only the *scoring* branches pay this; the replay branch - the one that
   produces gradients - still trains on 32 distinct levels, exactly like ACCEL.
2. **Replayed levels keep a running estimate.** Each replay contributes one more
   Bernoulli sample, folded into the buffer entry's stored `p` with
   `--sfl_p_decay`. Without it a level's score would be frozen at whatever it was
   when the level entered, and the buffer would fill with levels that *were*
   learnable ten thousand updates ago.
3. **The SFL phase.** Every `--sfl_period` updates, `--sfl_num_levels` fresh
   random levels are evaluated at the current (frozen) policy over
   `sfl_num_levels * k / num_envs` consecutive gradient-free updates, and the
   `--sfl_topk` most learnable are inserted. This is the paper's mechanism, and
   the only way a level is judged against a whole population rather than against
   whatever else happened to be in that update's batch.

Budget. Everything above is paid for out of the frozen 30000 updates; nothing is
added. The defaults size the phase to cost what ACCEL spends on its DR branch
(11.1% of updates), and the preset sets `replay_prob=1.0` so that outside a phase
every update is a replay (gradient) or a mutation - the same 44.4/44.4 split
ACCEL settles into. Both methods therefore see 245.76M env steps and ~13.3k
gradient updates; `sfl_budget_report` prints the actual counts from a finished
run so the claim is checked rather than asserted.
"""

from __future__ import annotations

from typing import Any, Dict, List

import chex
import jax
import jax.numpy as jnp

from jaxued.utils import compute_max_returns

from tlab_ued.scoring import RolloutSignals, first_episode_success, get_score_fn
from tlab_ued.teachers.accel import ACCELTeacher
from tlab_ued.teachers.base import BranchFn, TrainState, branch_budget
from tlab_ued.teachers.plr import UpdateState

# Branch indices, in the order `branches` returns them.
BRANCH_DR, BRANCH_REPLAY, BRANCH_MUTATE, BRANCH_SFL = 0, 1, 2, 3

# Scalar placeholder for the PPO losses on branches that run no PPO update. The
# SFL phase deliberately does not touch the network, and `jax.lax.switch` wants
# every branch to return the same pytree.
_NO_LOSSES = (
    jnp.zeros((), dtype=jnp.float32),
    (
        jnp.zeros((), dtype=jnp.float32),
        jnp.zeros((), dtype=jnp.float32),
        jnp.zeros((), dtype=jnp.float32),
    ),
)


def phase_length(config: Dict[str, Any]) -> int:
    """How many updates one SFL evaluation phase occupies.

    Zero when the phase is switched off *or* when another teacher is driving:
    the `sfl_*` defaults are present in every config, so this has to ask who is
    reading them before it believes them.
    """
    if config.get("teacher") != "sfl_accel" or not config.get("sfl_period"):
        return 0
    return (config["sfl_num_levels"] * config["sfl_num_attempts"]) // config["num_train_envs"]


def sfl_budget_report(config: Dict[str, Any]) -> Dict[str, float]:
    """The expected budget split of an SFL-ACCEL run.

    `branch_budget` does the arithmetic; the only SFL-specific part is how many
    updates the evaluation phases reserve. Run it on the `accel` preset and on
    the `sfl_accel` preset: the two should agree on env steps and on gradient
    updates, which is what "same budget" means here.
    """
    length = phase_length(config)
    phase_updates = (config["num_updates"] // config["sfl_period"]) * length if length else 0
    return branch_budget(config, reserved_updates=phase_updates)


def repeat_levels(levels, k: int):
    """Each level k times in a row: [l0, l0, l1, l1, ...] for k = 2."""
    return jax.tree_util.tree_map(lambda x: jnp.repeat(x, k, axis=0), levels)


def per_level(values: chex.Array, k: int, reduce: str = "mean") -> chex.Array:
    """Fold a (num_envs,) per-attempt array back to (num_envs/k,) per level."""
    grouped = values.reshape(-1, k)
    return grouped.mean(axis=1) if reduce == "mean" else grouped.max(axis=1)


class SFLACCELTeacher(ACCELTeacher):
    """ACCEL whose every insertion decision is made by learnability."""

    name = "sfl_accel"

    def __init__(self, ctx):
        super().__init__(ctx)
        config = self.config
        validate(config)
        self.attempts = int(config["sfl_num_attempts"])
        self.levels_per_update = ctx.num_envs // self.attempts
        self.phase_len = phase_length(config)
        self.period = int(config["sfl_period"])
        self.topk = int(config["sfl_topk"])
        self.num_candidates = int(config["sfl_num_levels"])
        self.p_decay = float(config["sfl_p_decay"])
        self.score_fn = get_score_fn(config)

    # --- scoring --------------------------------------------------------------
    def score_from_success(self, success_rate: chex.Array) -> chex.Array:
        """Run the configured score function over per-level success rates."""
        return self.score_fn(self.config, RolloutSignals(extras={"success_rate": success_rate}))

    def evaluate_levels(self, rng: chex.PRNGKey, train_state: TrainState, levels):
        """Roll out `levels` with `k` attempts each and measure their success rate.

        `levels` has a leading dimension of `num_envs / k`; the returned
        trajectory covers all `num_envs` envs, so a caller that wants to learn
        from these rollouts still gets a full batch.
        """
        ctx = self.ctx
        rng, rng_reset = jax.random.split(rng)
        batch = repeat_levels(levels, self.attempts)
        init_obs, init_env_state = ctx.reset_to_levels(rng_reset, batch)

        carry, traj, advantages, targets = ctx.rollout(
            rng, train_state, ctx.initial_carry(), init_obs, init_env_state
        )
        obs, actions, rewards, dones, log_probs, values, info = traj

        success = first_episode_success(dones, rewards)
        success_rate = per_level(success, self.attempts, "mean")
        max_returns = per_level(compute_max_returns(dones, rewards), self.attempts, "max")
        return carry, traj, advantages, targets, batch, success_rate, max_returns

    # --- state ----------------------------------------------------------------
    def init_teacher_state(self, rng: chex.PRNGKey) -> Any:
        """PLR's state, plus a per-level success rate and the SFL phase buffers."""
        ctx = self.ctx
        state = super().init_teacher_state(rng)
        pholder_level = ctx.sample_random_level(jax.random.PRNGKey(0))

        # The buffer carries `p` alongside `max_return`: `max_return` is what
        # MaxMC would need and is worth logging either way, `p` is what this
        # teacher scores on.
        sampler = self.level_sampler.initialize(
            pholder_level, {"max_return": -jnp.inf, "p": jnp.float32(0.0)}
        )

        def repeat(n):
            return jax.tree_util.tree_map(
                lambda x: jnp.array([x]).repeat(n, axis=0), pholder_level
            )

        zero = jnp.asarray(0, dtype=jnp.int32)
        return {
            **state,
            "sampler": sampler,
            "num_sfl_updates": zero,
            # The candidate population of the phase in flight, and the running
            # measurements over it.
            "sfl_levels": repeat(self.num_candidates),
            "sfl_success": jnp.zeros(self.num_candidates, dtype=jnp.float32),
            "sfl_max_return": jnp.full(self.num_candidates, -jnp.inf, dtype=jnp.float32),
            # What the last phase decided to keep - for rendering and for the log.
            "sfl_topk_levels": repeat(self.topk),
            "sfl_topk_score": jnp.zeros((), dtype=jnp.float32),
            "sfl_population_score": jnp.zeros((), dtype=jnp.float32),
        }

    # --- branches -------------------------------------------------------------
    @property
    def branches(self) -> List[BranchFn]:
        return [
            self.on_new_levels,
            self.on_replay_levels,
            self.on_mutate_levels,
            self.on_sfl_eval,
        ]

    def select_branch(self, rng: chex.PRNGKey, train_state: TrainState) -> chex.Array:
        """ACCEL's branch arithmetic, pre-empted by the SFL phase.

        A phase starts on the periodic boundary whatever the branch state was, so
        one owed mutation is dropped per phase (120 of ~13000 over a full run).
        Deferring it instead would make the phase boundary depend on the coin
        flips, and with it the budget accounting.
        """
        s = train_state.teacher_state["update_state"]
        replay_decision = self.level_sampler.sample_replay_decision(
            train_state.teacher_state["sampler"], rng
        )
        accel_branch = (1 - s) * replay_decision + 2 * s
        if self.phase_len == 0:
            return accel_branch
        in_phase = (train_state.update_count % self.period) < self.phase_len
        return jnp.where(in_phase, BRANCH_SFL, accel_branch)

    def on_new_levels(self, rng: chex.PRNGKey, train_state: TrainState):
        """Fresh random levels, evaluated with k attempts and offered to the buffer.

        With `replay_prob=1.0` this branch only runs while the buffer is below
        `minimum_fill_ratio`, i.e. during warm-up; the SFL phase is what supplies
        fresh levels afterwards.
        """
        ctx = self.ctx
        ts = train_state.teacher_state

        rng, rng_levels = jax.random.split(rng)
        levels = jax.vmap(ctx.sample_random_level)(
            jax.random.split(rng_levels, self.levels_per_update)
        )
        carry, traj, advantages, targets, batch, success_rate, max_returns = self.evaluate_levels(
            rng, train_state, levels
        )
        rng, train_state, _, _, _, _ = carry
        obs, actions, rewards, dones, log_probs, values, info = traj

        scores = self.score_from_success(success_rate)
        sampler, _ = self.level_sampler.insert_batch(
            ts["sampler"], levels, scores, {"max_return": max_returns, "p": success_rate}
        )

        (rng, train_state), losses = ctx.ppo_update(
            rng,
            train_state,
            ctx.initial_carry(),
            (obs, actions, dones, log_probs, values, targets, advantages),
            update_grad=self.config["exploratory_grad_updates"],
        )

        train_state = train_state.replace(
            teacher_state={
                **ts,
                "sampler": sampler,
                "update_state": jnp.asarray(UpdateState.DR, dtype=jnp.int32),
                "num_dr_updates": ts["num_dr_updates"] + 1,
                "dr_last_level_batch": batch,
            }
        )
        return (rng, train_state), metrics(losses, batch, ctx.num_envs, success_rate, scores)

    def on_replay_levels(self, rng: chex.PRNGKey, train_state: TrainState):
        """ACCEL's replay branch, with the score refreshed from a running `p`.

        The rollout is unchanged: 32 distinct levels, one attempt each, gradient
        applied. That single Bernoulli sample cannot score the level on its own,
        but it updates the estimate the score is computed from:
            p <- (1 - decay) * p + decay * solved
        """
        ctx = self.ctx
        ts = train_state.teacher_state

        rng, rng_levels, rng_reset = jax.random.split(rng, 3)
        sampler, (level_inds, levels) = self.level_sampler.sample_replay_levels(
            ts["sampler"], rng_levels, ctx.num_envs
        )
        init_obs, init_env_state = ctx.reset_to_levels(rng_reset, levels)

        carry, traj, advantages, targets = ctx.rollout(
            rng, train_state, ctx.initial_carry(), init_obs, init_env_state
        )
        rng, train_state, _, _, _, _ = carry
        obs, actions, rewards, dones, log_probs, values, info = traj

        extra = self.level_sampler.get_levels_extra(sampler, level_inds)
        max_returns = jnp.maximum(extra["max_return"], compute_max_returns(dones, rewards))
        success_rate = (1.0 - self.p_decay) * extra["p"] + self.p_decay * first_episode_success(
            dones, rewards
        )
        scores = self.score_from_success(success_rate)
        sampler = self.level_sampler.update_batch(
            sampler, level_inds, scores, {"max_return": max_returns, "p": success_rate}
        )

        (rng, train_state), losses = ctx.ppo_update(
            rng,
            train_state,
            ctx.initial_carry(),
            (obs, actions, dones, log_probs, values, targets, advantages),
            update_grad=True,
        )

        train_state = train_state.replace(
            teacher_state={
                **ts,
                "sampler": sampler,
                "update_state": jnp.asarray(UpdateState.REPLAY, dtype=jnp.int32),
                "num_replay_updates": ts["num_replay_updates"] + 1,
                "replay_last_level_batch": levels,
            }
        )
        return (rng, train_state), metrics(losses, levels, ctx.num_envs, success_rate, scores)

    def on_mutate_levels(self, rng: chex.PRNGKey, train_state: TrainState):
        """Mutate the levels just replayed, and judge the children by learnability.

        ACCEL mutates all 32 replayed levels and scores each child from one
        rollout. Here `32/k` of them are mutated and each child is played `k`
        times, which is the trade this method makes everywhere: fewer candidates,
        but a score that can tell "hard" from "impossible".
        """
        ctx = self.ctx
        ts = train_state.teacher_state

        rng, rng_mutate = jax.random.split(rng)
        parents = jax.tree_util.tree_map(
            lambda x: x[: self.levels_per_update], ts["replay_last_level_batch"]
        )
        children = jax.vmap(ctx.mutate_level, (0, 0, None))(
            jax.random.split(rng_mutate, self.levels_per_update),
            parents,
            self.config["num_edits"],
        )
        carry, traj, advantages, targets, batch, success_rate, max_returns = self.evaluate_levels(
            rng, train_state, children
        )
        rng, train_state, _, _, _, _ = carry
        obs, actions, rewards, dones, log_probs, values, info = traj

        scores = self.score_from_success(success_rate)
        sampler, _ = self.level_sampler.insert_batch(
            ts["sampler"], children, scores, {"max_return": max_returns, "p": success_rate}
        )

        (rng, train_state), losses = ctx.ppo_update(
            rng,
            train_state,
            ctx.initial_carry(),
            (obs, actions, dones, log_probs, values, targets, advantages),
            update_grad=self.config["exploratory_grad_updates"],
        )

        train_state = train_state.replace(
            teacher_state={
                **ts,
                "sampler": sampler,
                "update_state": jnp.asarray(UpdateState.DR, dtype=jnp.int32),
                "num_mutation_updates": ts["num_mutation_updates"] + 1,
                "mutation_last_level_batch": batch,
            }
        )
        return (rng, train_state), metrics(losses, batch, ctx.num_envs, success_rate, scores)

    def on_sfl_eval(self, rng: chex.PRNGKey, train_state: TrainState):
        """One step of an SFL evaluation phase over a fresh candidate population.

        The phase spans `phase_len` consecutive updates. Step 0 draws the
        population; every step measures `num_envs / k` of it; the last step ranks
        the whole population and inserts the `topk` most learnable. The policy is
        never updated here - the population is judged at one fixed policy, which
        is what makes the ranking a ranking and not a moving target.
        """
        ctx = self.ctx
        ts = train_state.teacher_state
        pos = train_state.update_count % self.period
        first_step, last_step = pos == 0, pos == self.phase_len - 1

        rng, rng_levels = jax.random.split(rng)
        candidates = jax.lax.cond(
            first_step,
            lambda: jax.vmap(ctx.sample_random_level)(
                jax.random.split(rng_levels, self.num_candidates)
            ),
            lambda: ts["sfl_levels"],
        )
        offset = pos * self.levels_per_update
        slice_idx = offset + jnp.arange(self.levels_per_update)
        levels = jax.tree_util.tree_map(lambda x: x[slice_idx], candidates)

        carry, traj, advantages, targets, batch, success_rate, max_returns = self.evaluate_levels(
            rng, train_state, levels
        )
        rng, train_state, _, _, _, _ = carry

        # A level's attempts all land in the same update, so this is a plain
        # write rather than an accumulation.
        measured = jnp.where(first_step, 0.0, ts["sfl_success"]).at[slice_idx].set(success_rate)
        best_return = (
            jnp.where(first_step, -jnp.inf, ts["sfl_max_return"]).at[slice_idx].set(max_returns)
        )

        population_scores = self.score_from_success(measured)
        top_scores, top_inds = jax.lax.top_k(population_scores, self.topk)
        top_levels = jax.tree_util.tree_map(lambda x: x[top_inds], candidates)
        sampler = jax.lax.cond(
            last_step,
            lambda: self.level_sampler.insert_batch(
                ts["sampler"],
                top_levels,
                top_scores,
                {"max_return": best_return[top_inds], "p": measured[top_inds]},
            )[0],
            lambda: ts["sampler"],
        )

        def keep(new, old):
            """The phase's verdict, held until the next phase replaces it."""
            return jax.tree_util.tree_map(lambda a, b: jnp.where(last_step, a, b), new, old)

        train_state = train_state.replace(
            teacher_state={
                **ts,
                "sampler": sampler,
                "update_state": jnp.asarray(UpdateState.DR, dtype=jnp.int32),
                "num_sfl_updates": ts["num_sfl_updates"] + 1,
                "sfl_levels": candidates,
                "sfl_success": measured,
                "sfl_max_return": best_return,
                "sfl_topk_levels": keep(top_levels, ts["sfl_topk_levels"]),
                "sfl_topk_score": keep(top_scores.mean(), ts["sfl_topk_score"]),
                "sfl_population_score": keep(
                    population_scores.mean(), ts["sfl_population_score"]
                ),
            }
        )
        out = metrics(
            _NO_LOSSES,
            batch,
            ctx.num_envs,
            success_rate,
            self.score_from_success(success_rate),
        )
        return (rng, train_state), out

    # --- reporting ------------------------------------------------------------
    def startup_report(self) -> Dict[str, Any]:
        """The budget split, printed before the first update.

        The numbers to compare against the ACCEL run's own report: identical env
        steps, and gradient updates within a fraction of a percent.
        """
        return {
            **sfl_budget_report(self.config),
            "sfl_phase_updates": self.phase_len,
            "levels_scored_per_update": self.levels_per_update,
            "attempts_per_level": self.attempts,
        }

    def log_dict(self, train_state: TrainState) -> Dict[str, Dict[str, Any]]:
        """PLR's buffer stats, plus what the learnability machinery is doing."""
        out = super().log_dict(train_state)
        ts = train_state.teacher_state
        sampler = ts["sampler"]
        filled = jnp.arange(self.level_sampler.capacity) < sampler["size"]
        size = jnp.maximum(filled.sum(), 1)
        out["log"].update(
            {
                # Mean success rate over the buffer: the curriculum's difficulty,
                # in the only unit that means the same thing at every point in
                # training.
                "level_sampler/mean_p": (sampler["levels_extra"]["p"] * filled).sum() / size,
                "level_sampler/mean_max_return": (
                    jnp.where(filled, sampler["levels_extra"]["max_return"], 0.0).sum() / size
                ),
                # How much better the kept levels are than the population they
                # were drawn from. If this ever collapses to ~1, the phase is
                # selecting nothing and the levels are all equally (un)learnable.
                "sfl/topk_learnability": ts["sfl_topk_score"],
                "sfl/population_learnability": ts["sfl_population_score"],
                "branch/num_dr_updates": ts["num_dr_updates"],
                "branch/num_replay_updates": ts["num_replay_updates"],
                "branch/num_mutation_updates": ts["num_mutation_updates"],
                "branch/num_sfl_updates": ts["num_sfl_updates"],
            }
        )
        out["info"]["num_sfl_updates"] = ts["num_sfl_updates"]
        return out

    def media(self, train_state: TrainState) -> Dict[str, Any]:
        if self.config.get("log_media", "all") == "none":
            return {}
        out = super().media(train_state)
        out["sfl_levels"] = jax.vmap(self.ctx.renderer.render_level, (0, None))(
            train_state.teacher_state["sfl_topk_levels"], self.ctx.env_params
        )
        return out


def metrics(losses, level_batch, num_envs: int, success_rate, scores) -> Dict[str, Any]:
    """The per-update metrics every branch returns - same keys, same shapes.

    `success_rate` and `learnability` are the two numbers that say what the
    curriculum is actually feeding the student, so every branch reports them,
    including the ones that apply no gradient.
    """
    return {
        "losses": jax.tree_util.tree_map(lambda x: x.mean(), losses),
        "mean_num_blocks": level_batch.wall_map.sum() / num_envs,
        "success_rate": success_rate.mean(),
        "learnability": scores.mean(),
    }


def validate(config: Dict[str, Any]) -> None:
    """Fail before the first rollout rather than inside a jitted branch."""
    num_envs, k = config["num_train_envs"], config["sfl_num_attempts"]
    if k < 2:
        raise ValueError(
            f"sfl_num_attempts={k}: with one attempt per level every learnability score is "
            "exactly 0. Use at least 2 (4 gives a graded score)."
        )
    if num_envs % k != 0:
        raise ValueError(f"num_train_envs={num_envs} must be divisible by sfl_num_attempts={k}")
    if config["sfl_period"]:
        if (config["sfl_num_levels"] * k) % num_envs != 0:
            raise ValueError(
                f"sfl_num_levels * sfl_num_attempts ({config['sfl_num_levels']} * {k}) must be "
                f"divisible by num_train_envs ({num_envs}) for the phase to fit whole updates"
            )
        if phase_length(config) > config["sfl_period"]:
            raise ValueError(
                f"an SFL phase is {phase_length(config)} updates but sfl_period is "
                f"{config['sfl_period']}: the phase would never end"
            )
        if config["sfl_topk"] > config["sfl_num_levels"]:
            raise ValueError("sfl_topk cannot exceed sfl_num_levels")
