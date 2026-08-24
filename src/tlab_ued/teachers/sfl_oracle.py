"""SFL-ORACLE - SFL-ACCEL with a learned model of the student's success rate.

`teachers/sfl_accel.py` buys its learnability scores with rollouts: 224 candidate
levels x 4 attempts per phase, 28 gradient-free updates, 11.2% of the budget, and
a candidate population that is still tiny next to the level space. This teacher
keeps that machinery and adds a second, much cheaper way to get `p` - a small
model that *predicts* it from the whole level (see `tlab_ued.oracle`), trained
online on the labels every rollout already produces.

Four places the prediction is spent, all of them costing zero env steps:

1. **The phase becomes a cascade.** `--oracle_num_proposals` (8192) fresh levels
   are ranked by predicted learnability; only the best `--sfl_num_levels` (64)
   are actually played. Selection pressure goes up (8192:64 rather than 224:32)
   and the phase gets shorter (8 updates rather than 28), which hands the
   difference back to gradient updates.
2. **Verification keeps it honest.** Those 64 are still measured by rollout, and
   what finally enters the buffer is the top `--sfl_topk` by *measured*
   learnability. `--oracle_control_levels` of the 64 are drawn uniformly at
   random instead of from the top of the ranking, which turns every phase into a
   controlled experiment on the oracle itself: `oracle/selection_gain` is how
   much more learnable its picks were than levels nobody picked.
3. **The buffer is re-scored for free.** A PLR score is only as fresh as the last
   replay of that level; at each phase the oracle re-scores all 4000 entries in
   one forward pass, so a level that stopped being learnable 3000 updates ago
   stops being sampled without having to be replayed first.
4. **Mutation stops being blind.** Each parent emits
   `--oracle_mutation_proposals` children and the one predicted closest to a coin
   flip is the one played - ACCEL's edit operator, hill-climbing on predicted
   learnability.

`--no-oracle_verify` removes stage 2 entirely: the phase collapses to a single
update that inserts the oracle's top-k on prediction alone and runs an ordinary
gradient replay alongside, so the *whole* 11% SFL spends on measurement returns
to training. That is the aggressive arm, and whether it works is a question about
the oracle's calibration rather than about learnability.

Budget. Env steps are unchanged and unchangeable: every update is 32 x 256 steps,
30000 updates, 245.76M. What changes is how many of those updates carry a
gradient - `oracle_budget_report` prints the split before the first update, and
every run logs the realised counts to `metrics.csv` as `branch/num_*_updates`.
The oracle's own training is not free but it is not measured in env steps: a
50k-parameter convnet taking `--oracle_train_steps` Adam steps per update,
against the student's 5 PPO epochs over 32 x 256 LSTM steps. Watch
`steps_per_second` in the CSV rather than assuming.
"""

from __future__ import annotations

from typing import Any, Dict, List

import chex
import jax
import jax.numpy as jnp
import numpy as np

from tlab_ued.oracle import LearnabilityOracle, calibration, correlation, rank
from tlab_ued.oracle import validate as validate_oracle_model
from tlab_ued.teachers.base import BranchFn, TrainState, branch_budget
from tlab_ued.teachers.plr import UpdateState
from tlab_ued.teachers.sfl_accel import BRANCH_SFL, HOOK_SALT, SFLACCELTeacher, metrics

# The oracle's phase diagnostics, held in the teacher state between phases so
# that a log written mid-period still describes the last one.
STAT_KEYS = (
    "brier",
    "control_brier",
    "bias",
    "rank_corr",
    "control_rank_corr",
    "selected_learnability",
    "control_learnability",
    "selection_gain",
    "predicted_learnability",
)


def phase_length(config: Dict[str, Any]) -> int:
    """How many updates one phase of this teacher occupies.

    With verification, the same arithmetic as SFL's: the verified population,
    `k` attempts each, divided over the envs of an update. Without it, a single
    update - which still carries a gradient, because ranking proposals costs no
    rollout and the update would otherwise be wasted.
    """
    if config.get("teacher") != "sfl_oracle" or not config.get("sfl_period"):
        return 0
    if not config.get("oracle_verify", True):
        return 1
    return (config["sfl_num_levels"] * config["sfl_num_attempts"]) // config["num_train_envs"]


def oracle_budget_report(config: Dict[str, Any]) -> Dict[str, float]:
    """The expected budget split, to be read next to the ACCEL run's own.

    Only *rollout-only* phase updates are reserved: the no-verify phase update
    trains the student like any other replay, so it is not withheld from the
    gradient count.
    """
    length = phase_length(config)
    reserved = 0
    if length and config.get("oracle_verify", True):
        reserved = (config["num_updates"] // config["sfl_period"]) * length
    return branch_budget(config, reserved_updates=reserved)


class SFLOracleTeacher(SFLACCELTeacher):
    """SFL-ACCEL whose candidate ranking comes from a model, not from rollouts."""

    name = "sfl_oracle"

    def __init__(self, ctx):
        super().__init__(ctx)
        config = self.config
        validate(config)
        self.oracle = LearnabilityOracle(config, ctx.env_params.max_steps_in_episode)
        self.num_proposals = int(config["oracle_num_proposals"])
        self.mutation_proposals = int(config["oracle_mutation_proposals"])
        self.rescore = bool(config["oracle_rescore_buffer"])
        self.verify = bool(config["oracle_verify"])
        self.warmup = int(config["oracle_warmup_updates"])
        self.control = int(config["oracle_control_levels"])
        # The parent computed these for its own phase; this teacher's phase is a
        # different shape, and `sfl_accel.phase_length` deliberately answers 0 for
        # anyone but `sfl_accel`.
        self.phase_len = phase_length(config)
        if not self.verify:
            # Nothing is measured, so the "population" of a phase is exactly what
            # it inserts, and a control group would have nothing to be measured
            # against.
            self.num_candidates = self.topk
            self.control = 0
        self.min_fill = self.level_sampler.capacity * float(config["minimum_fill_ratio"])
        # Floor for the `selection_gain` denominator: the smallest non-zero mean
        # learnability a control group can have, i.e. one of its levels scoring a
        # single success out of `k`. See `phase_stats`.
        one_success = float(self.score_from_success(jnp.asarray(1.0 / self.attempts)))
        self.gain_floor = one_success / max(self.control, 1)

    # --- state ----------------------------------------------------------------
    def init_teacher_state(self, rng: chex.PRNGKey) -> Any:
        """SFL's state, plus the oracle and the last phase's verdict on it."""
        state = super().init_teacher_state(rng)
        pholder_level = self.ctx.sample_random_level(jax.random.PRNGKey(0))
        return {
            **state,
            # Folded, not split: `create_train_state` hands the same key to the
            # student's initialiser, and the oracle must not perturb it.
            "oracle": self.oracle.initialize(
                jax.random.fold_in(rng, 0x0AC1), pholder_level
            ),
            # What the oracle predicted for the levels the current phase is
            # verifying - kept so the phase can score itself at the end.
            "sfl_pred": jnp.zeros(self.num_candidates, dtype=jnp.float32),
            "oracle_stats": {k: jnp.zeros((), dtype=jnp.float32) for k in STAT_KEYS},
            # Insertions the oracle has driven, i.e. completed phases. Counted
            # apart from `num_sfl_updates` because without verification an
            # insertion costs no update at all, and the four branch counters have
            # to keep summing to the number of updates.
            "num_oracle_inserts": jnp.asarray(0, dtype=jnp.int32),
        }

    # --- the oracle -----------------------------------------------------------
    def after_rollout(
        self, rng: chex.PRNGKey, ts: Dict[str, Any], levels, attempts, successes
    ) -> Dict[str, Any]:
        """Every rollout is a labelled example. Record it, then take a few steps.

        This is what makes the oracle track the policy rather than describe an old
        one: it runs on every update, on whatever that update happened to play.
        The bias that introduces - most labels come from buffer levels, which are
        not a random sample of anything - is the reason the phase keeps a
        uniformly random control group.
        """
        oracle = self.oracle.observe(ts["oracle"], levels, attempts, successes)
        return {**ts, "oracle": self.oracle.train(oracle, rng)}

    def predicted_score(self, oracle_state, levels) -> chex.Array:
        """The configured score function over the oracle's predicted `p`."""
        return self.score_from_success(self.oracle.predict(oracle_state, levels))

    def warm(self, update_count: chex.Array) -> chex.Array:
        """Has the oracle seen enough to be worth listening to?

        Before this, selections fall back to uniform random - which is exactly
        what an uninformative ranking should do, and avoids spending the first
        phases pursuing the initialisation's arbitrary preferences.
        """
        return update_count >= self.warmup

    def propose(self, rng: chex.PRNGKey, ts: Dict[str, Any], update_count: chex.Array):
        """Rank a large fresh population and return the shortlist to play.

        The shortlist is `[control | selected]`: the first
        `--oracle_control_levels` are uniform draws from the population, the rest
        are the oracle's top picks. Keeping the control group *inside* the
        shortlist means it is measured under identical conditions, at the same
        policy, in the same updates.
        """
        ctx = self.ctx
        rng_levels, rng_control, rng_tie = jax.random.split(rng, 3)
        proposals = jax.vmap(ctx.sample_random_level)(
            jax.random.split(rng_levels, self.num_proposals)
        )
        predicted = self.oracle.predict(ts["oracle"], proposals)
        ranking = jnp.where(
            self.warm(update_count),
            self.score_from_success(predicted),
            jax.random.uniform(rng_tie, (self.num_proposals,)),
        )
        _, top = jax.lax.top_k(ranking, self.num_candidates - self.control)
        if self.control:
            control = jax.random.choice(
                rng_control, self.num_proposals, (self.control,), replace=False
            )
            top = jnp.concatenate([control, top])
        shortlist = jax.tree_util.tree_map(lambda x: x[top], proposals)
        return shortlist, predicted[top]

    def rescore_buffer(self, ts: Dict[str, Any], sampler):
        """Refresh every buffer entry's score from the oracle. One forward pass.

        Only `scores` is touched. `levels_extra["p"]` stays the last *measured*
        success rate, so `level_sampler/mean_p` keeps meaning what it means in the
        SFL run; the replay branch overwrites the score from that measurement the
        moment a level is played again.
        """
        if not self.rescore:
            return sampler
        filled = jnp.arange(self.level_sampler.capacity) < sampler["size"]
        predicted = self.predicted_score(ts["oracle"], sampler["levels"])
        return {**sampler, "scores": jnp.where(filled, predicted, sampler["scores"])}

    # --- branches -------------------------------------------------------------
    @property
    def branches(self) -> List[BranchFn]:
        phase = self.on_sfl_eval if self.verify else self.on_oracle_insert
        return [self.on_new_levels, self.on_replay_levels, self.on_mutate_levels, phase]

    def select_branch(self, rng: chex.PRNGKey, train_state: TrainState) -> chex.Array:
        """ACCEL's arithmetic, pre-empted by the phase - as in SFL, with one guard.

        The no-verify phase runs a replay inside it, so unlike SFL's purely
        exploratory phase it cannot start before the buffer is replayable.
        """
        ts = train_state.teacher_state
        s = ts["update_state"]
        replay_decision = self.level_sampler.sample_replay_decision(ts["sampler"], rng)
        accel_branch = (1 - s) * replay_decision + 2 * s
        if self.phase_len == 0:
            return accel_branch
        in_phase = (train_state.update_count % self.period) < self.phase_len
        if not self.verify:
            in_phase = in_phase & (ts["sampler"]["size"] >= self.min_fill)
        return jnp.where(in_phase, BRANCH_SFL, accel_branch)

    def propose_children(self, rng: chex.PRNGKey, train_state: TrainState, parents):
        """Best-of-G mutation: G children per parent, keep the most learnable-looking.

        ACCEL's operator is unchanged - these are the same random edits it would
        make. What changes is that the teacher gets to look at G of them before
        spending a rollout, which is the cheapest possible form of the search
        ACCEL does by evolving the buffer over many generations.
        """
        g = self.mutation_proposals
        if g <= 1:
            return super().propose_children(rng, train_state, parents)

        ctx, n = self.ctx, self.levels_per_update
        rng_mutate, rng_tie = jax.random.split(rng)
        keys = jax.random.split(rng_mutate, n * g)
        keys = jnp.reshape(keys, (n, g) + keys.shape[1:])
        children = jax.vmap(jax.vmap(ctx.mutate_level, (0, None, None)), (0, 0, None))(
            keys, parents, self.config["num_edits"]
        )

        flat = jax.tree_util.tree_map(lambda x: x.reshape((n * g,) + x.shape[2:]), children)
        predicted = self.predicted_score(train_state.teacher_state["oracle"], flat)
        ranking = jnp.where(
            self.warm(train_state.update_count),
            predicted.reshape(n, g),
            jax.random.uniform(rng_tie, (n, g)),
        )
        best = jnp.argmax(ranking, axis=1)
        return jax.tree_util.tree_map(lambda x: x[jnp.arange(n), best], children)

    def on_sfl_eval(self, rng: chex.PRNGKey, train_state: TrainState):
        """One step of a verification phase: play the oracle's shortlist.

        Structurally SFL's phase, with the population chosen by the oracle rather
        than by the level generator, and a seventh of the size. The first step
        proposes and re-scores the buffer; every step measures
        `num_envs / k` of the shortlist; the last inserts the most learnable by
        *measured* `p` and writes down how well the prediction did.
        """
        ctx = self.ctx
        ts = train_state.teacher_state
        pos = train_state.update_count % self.period
        first_step, last_step = pos == 0, pos == self.phase_len - 1

        rng, rng_propose = jax.random.split(rng)
        candidates, predicted = jax.lax.cond(
            first_step,
            lambda: self.propose(rng_propose, ts, train_state.update_count),
            lambda: (ts["sfl_levels"], ts["sfl_pred"]),
        )
        sampler = jax.lax.cond(
            first_step,
            lambda: self.rescore_buffer(ts, ts["sampler"]),
            lambda: ts["sampler"],
        )

        offset = pos * self.levels_per_update
        slice_idx = offset + jnp.arange(self.levels_per_update)
        levels = jax.tree_util.tree_map(lambda x: x[slice_idx], candidates)

        carry, traj, advantages, targets, batch, success_rate, max_returns = self.evaluate_levels(
            rng, train_state, levels
        )
        rng, train_state, _, _, _, _ = carry

        # A level's attempts all land in the same update, so this is a plain write
        # rather than an accumulation.
        measured = jnp.where(first_step, 0.0, ts["sfl_success"]).at[slice_idx].set(success_rate)
        best_return = (
            jnp.where(first_step, -jnp.inf, ts["sfl_max_return"]).at[slice_idx].set(max_returns)
        )

        scores = self.score_from_success(measured)
        top_scores, top_inds = jax.lax.top_k(scores, self.topk)
        top_levels = jax.tree_util.tree_map(lambda x: x[top_inds], candidates)
        sampler = jax.lax.cond(
            last_step,
            lambda: self.level_sampler.insert_batch(
                sampler,
                top_levels,
                top_scores,
                {"max_return": best_return[top_inds], "p": measured[top_inds]},
            )[0],
            lambda: sampler,
        )

        def keep(new, old):
            """The phase's verdict, held until the next phase replaces it."""
            return jax.tree_util.tree_map(lambda a, b: jnp.where(last_step, a, b), new, old)

        ts = self.after_rollout(
            jax.random.fold_in(rng, HOOK_SALT), ts, levels, *self.attempt_counts(success_rate)
        )
        train_state = train_state.replace(
            teacher_state={
                **ts,
                "sampler": sampler,
                "update_state": jnp.asarray(UpdateState.DR, dtype=jnp.int32),
                "num_sfl_updates": ts["num_sfl_updates"] + 1,
                "num_oracle_inserts": ts["num_oracle_inserts"] + last_step.astype(jnp.int32),
                "sfl_levels": candidates,
                "sfl_pred": predicted,
                "sfl_success": measured,
                "sfl_max_return": best_return,
                "sfl_topk_levels": keep(top_levels, ts["sfl_topk_levels"]),
                "sfl_topk_score": keep(top_scores.mean(), ts["sfl_topk_score"]),
                "sfl_population_score": keep(scores.mean(), ts["sfl_population_score"]),
                "oracle_stats": keep(
                    self.phase_stats(predicted, measured), ts["oracle_stats"]
                ),
            }
        )
        out = metrics(
            _no_losses(),
            batch,
            ctx.num_envs,
            success_rate,
            self.score_from_success(success_rate),
        )
        return (rng, train_state), out

    def on_oracle_insert(self, rng: chex.PRNGKey, train_state: TrainState):
        """The no-verify phase: insert on prediction alone, then train as usual.

        Nothing here costs a rollout, so the update is not spent on the insertion
        - it goes on to be an ordinary gradient replay. That is the whole claim of
        `--no-oracle_verify`: measurement was the only reason SFL had to give up
        11% of its updates.
        """
        ts = train_state.teacher_state
        rng, rng_propose = jax.random.split(rng)

        levels, predicted = self.propose(rng_propose, ts, train_state.update_count)
        scores = self.score_from_success(predicted)
        sampler, _ = self.level_sampler.insert_batch(
            self.rescore_buffer(ts, ts["sampler"]),
            levels,
            scores,
            # Never played, so there is no return to record and no measured `p`:
            # 0.0 is the honest lower bound on both.
            {"max_return": jnp.zeros_like(predicted), "p": predicted},
        )

        train_state = train_state.replace(
            teacher_state={
                **ts,
                "sampler": sampler,
                # Not a branch counter: this update goes on to be a replay, and is
                # counted as one.
                "num_oracle_inserts": ts["num_oracle_inserts"] + 1,
                "sfl_levels": levels,
                "sfl_pred": predicted,
                "sfl_topk_levels": levels,
                "sfl_topk_score": scores.mean(),
                "sfl_population_score": scores.mean(),
            }
        )
        return self.on_replay_levels(rng, train_state)

    # --- reporting ------------------------------------------------------------
    def phase_stats(self, predicted: chex.Array, measured: chex.Array) -> Dict[str, chex.Array]:
        """What one phase learned about its own oracle.

        `selection_gain` is the number to look at: measured learnability of the
        levels the oracle chose, over measured learnability of the uniformly
        random controls that rode along with them. 1.0 means the oracle is
        choosing no better than chance; SFL's own phase, for comparison, logs the
        same ratio as `sfl/topk_learnability` over `sfl/population_learnability`
        - but that one selects and evaluates on the *same* noisy statistic, so it
        is inflated by regression to the mean. This ratio is out of sample: the
        oracle commits to a ranking, and only then are the rollouts run.

        The denominator is floored at the smallest non-zero value it can take
        (one control level scoring one success out of `k`, spread over the control
        group). Late in training whole control groups score exactly zero, and an
        unfloored ratio turns those phases into 10^5 spikes that wreck any average
        taken over the column. `selected_learnability` and `control_learnability`
        are logged separately and are the pair to trust.
        """
        stats = dict(calibration(predicted, measured))
        learnability = self.score_from_success(measured)
        control = learnability[: self.control].mean() if self.control else jnp.zeros(())
        selected = learnability[self.control :].mean()
        stats.update(
            {
                "selected_learnability": selected,
                "control_learnability": control,
                "selection_gain": selected / jnp.maximum(control, self.gain_floor),
                # Rank correlation on the controls alone. The shortlist version
                # below is attenuated to near zero by construction: the oracle
                # picked those levels *because* it predicts them all at the same
                # `p`, and a correlation over a range-restricted sample carries no
                # information about the ranking that produced it. This one is
                # measured on levels drawn uniformly, so it is the honest answer
                # to "can it rank levels at all".
                "control_rank_corr": (
                    correlation(rank(predicted[: self.control]), rank(measured[: self.control]))
                    if self.control > 2
                    else jnp.zeros(())
                ),
                # What the oracle *thought* it was picking. Far above
                # `selected_learnability` is the winner's curse: an argmax over a
                # prediction selects partly for the prediction being wrong.
                "predicted_learnability": self.score_from_success(predicted).mean(),
                # Calibration on the control group alone. The full-set Brier score
                # is measured on levels the oracle chose, which is the one
                # population it cannot be trusted to be honest about; this one is
                # measured on levels drawn uniformly, and is the number that says
                # whether the oracle generalises off its own training
                # distribution.
                "control_brier": (
                    ((predicted[: self.control] - measured[: self.control]) ** 2).mean()
                    if self.control
                    else jnp.zeros(())
                ),
            }
        )
        return {k: jnp.asarray(stats[k], dtype=jnp.float32) for k in STAT_KEYS}

    def startup_report(self) -> Dict[str, Any]:
        """The budget split and the shape of the cascade, before the first update."""
        return {
            **oracle_budget_report(self.config),
            "sfl_phase_updates": self.phase_len,
            "levels_scored_per_update": self.levels_per_update,
            "attempts_per_level": self.attempts,
            "oracle_proposals_per_phase": self.num_proposals,
            "oracle_verified_per_phase": self.num_candidates if self.verify else 0,
            "oracle_parameters": self.num_oracle_params(),
        }

    def num_oracle_params(self) -> int:
        """Parameter count, from shapes only - no allocation, no compute."""
        pholder = self.ctx.sample_random_level(jax.random.PRNGKey(0))
        batch = jax.tree_util.tree_map(lambda x: jnp.asarray(x)[None], pholder)
        planes, scalars = self.oracle.features_of(batch)
        shapes = jax.eval_shape(
            self.oracle.net.init, jax.random.PRNGKey(0), planes, scalars
        )
        return int(sum(np.prod(leaf.shape) for leaf in jax.tree_util.tree_leaves(shapes)))

    def log_dict(self, train_state: TrainState) -> Dict[str, Dict[str, Any]]:
        """SFL's log, plus everything the oracle knows about itself."""
        out = super().log_dict(train_state)
        ts = train_state.teacher_state
        oracle = ts["oracle"]
        out["log"].update({f"oracle/{k}": v for k, v in ts["oracle_stats"].items()})
        filled = jnp.arange(self.level_sampler.capacity) < ts["sampler"]["size"]
        predicted = self.oracle.predict(oracle, ts["sampler"]["levels"])
        out["log"].update(
            {
                "oracle/loss": oracle.loss,
                "oracle/observations": oracle.count,
                "branch/num_oracle_inserts": ts["num_oracle_inserts"],
                # The oracle's view of the buffer: where it thinks the curriculum
                # currently sits, on the same scale as `level_sampler/mean_p`.
                # The two disagreeing is the interesting case - one of them is
                # then describing a policy that no longer exists.
                "oracle/buffer_mean_p": (predicted * filled).sum()
                / jnp.maximum(filled.sum(), 1),
            }
        )
        return out


def _no_losses():
    """Zero PPO losses, for the phase updates that never call the optimiser."""
    zero = jnp.zeros((), dtype=jnp.float32)
    return (zero, (zero, zero, zero))


def validate(config: Dict[str, Any]) -> None:
    """Fail before the first rollout rather than inside a jitted branch."""
    validate_oracle_model(config)
    if int(config["oracle_num_proposals"]) < 1:
        raise ValueError("oracle_num_proposals must be at least 1")
    if int(config["oracle_mutation_proposals"]) < 1:
        raise ValueError("oracle_mutation_proposals must be at least 1 (1 = unguided)")
    if not config.get("sfl_period"):
        return  # no phase: guided mutation only, nothing below applies

    verify = bool(config.get("oracle_verify", True))
    shortlist = int(config["sfl_num_levels"]) if verify else int(config["sfl_topk"])
    if shortlist > int(config["oracle_num_proposals"]):
        raise ValueError(
            f"the phase would keep {shortlist} of {config['oracle_num_proposals']} proposals: "
            "oracle_num_proposals must be at least as large"
        )
    if not verify:
        return
    if int(config["oracle_control_levels"]) >= int(config["sfl_num_levels"]):
        raise ValueError(
            f"oracle_control_levels={config['oracle_control_levels']} leaves no room for the "
            f"oracle's own picks in a shortlist of {config['sfl_num_levels']}"
        )
    if (config["sfl_num_levels"] * config["sfl_num_attempts"]) % config["num_train_envs"]:
        raise ValueError(
            f"sfl_num_levels * sfl_num_attempts ({config['sfl_num_levels']} * "
            f"{config['sfl_num_attempts']}) must be divisible by num_train_envs "
            f"({config['num_train_envs']}) for the phase to fit whole updates"
        )
    if phase_length(config) > config["sfl_period"]:
        raise ValueError(
            f"a verification phase is {phase_length(config)} updates but sfl_period is "
            f"{config['sfl_period']}: the phase would never end"
        )
    if config["sfl_topk"] > config["sfl_num_levels"]:
        raise ValueError("sfl_topk cannot exceed sfl_num_levels")
