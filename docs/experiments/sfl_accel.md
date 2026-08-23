# SFL-ACCEL: the method, the budget, and the two runs

**Question.** ACCEL's weak link is MaxMC: a regret proxy that, per Rutherford et al. 2024
(arXiv:2408.15099), ranks levels by something other than what the student can learn from - and on
this generator it ranks *impossible* levels highest, because a goal walled off behind a wall
produces a large, permanent value error. SFL replaces it with **learnability**, `p(1-p)`, where `p`
is the level's success rate under the current policy. Does putting learnability in charge of what
enters ACCEL's level buffer beat MaxMC-ACCEL on held-out levels at the same budget?

## The method

Implemented in [`teachers/sfl_accel.py`](../../src/tlab_ued/teachers/sfl_accel.py); the score
itself is `learnability` in [`scoring.py`](../../src/tlab_ued/scoring.py).

Everything follows from one fact: **`p` cannot be estimated from a single rollout.** Episodes are
250 steps and rollouts are 256, so each env contributes one episode; `p_hat` is 0 or 1 and
`p_hat(1-p_hat)` is 0 either way. A naive "swap MaxMC for `p(1-p)`" scores every level zero. So:

1. **Multi-attempt scoring.** Whenever levels are *scored*, the 32 envs carry `32/k` distinct
   levels, `k = --sfl_num_attempts` times each (default 4 → 8 levels/update). Same rollout, same
   cost; fewer levels, but a `p_hat` with a gradient in it.
2. **A running estimate on replay.** The replay branch is left exactly as ACCEL's - 32 distinct
   levels, one attempt, gradient applied - so the student's data distribution is unchanged. Its one
   Bernoulli sample folds into the buffer entry's stored `p`:
   `p ← (1-decay)·p + decay·solved`, `decay = --sfl_p_decay` (0.25, ≈10-replay memory). Without
   this the buffer fills with levels that *were* learnable ten thousand updates ago.
3. **The SFL phase.** Every `--sfl_period` updates (250), `--sfl_num_levels` fresh random levels
   (224) are evaluated at the frozen current policy over 28 consecutive gradient-free updates, and
   the `--sfl_topk` most learnable (32) are inserted. This is the paper's mechanism: the only
   place a level is ranked against a whole population rather than against the seven others that
   happened to share its update.

ACCEL's mutation operator is unchanged, and still evolves the buffer's high-scoring levels - it is
just that "high-scoring" now means "the student solves it about half the time" rather than "the
critic is confused about it". Setting `--sfl_period 0` (preset `sfl_accel_nophase`) keeps 1 and 2
and drops 3, which is the ablation separating "learnability" from "SFL".

## Budget

The frozen budget is 30000 updates × 32 envs × 256 steps = **245.76M env steps**. In steady state
ACCEL spends them as 11.1% DR / 44.4% replay / 44.4% mutation, so 13,333 rollouts carry a gradient
and **over half of ACCEL's env steps already go to scoring rather than to learning**.

SFL-ACCEL adds nothing. The phase is not free time, it is updates: each phase step is one ordinary
iteration of `train.py`'s scan, drawn from the same 30000. The defaults size the phases
(120 × 28 = 3360 updates, 11.2%) to cost exactly what ACCEL spends on its DR branch, and the preset
sets `replay_prob=1.0` so that outside a phase every update is a replay or its mutation:

| | env steps | gradient updates | mutation | DR | SFL phase |
|---|---|---|---|---|---|
| `accel` | 245.76M | 13,333 | 13,333 | 3,333 | - |
| `sfl_accel` | 245.76M | 13,320 | 13,320 | warm-up only | 3,360 |

0.1% apart on gradient updates, identical on env steps. `branch_budget` (in `teachers/base.py`)
computes this; every run prints it before the first update, and every run logs the *actual* branch
counts to `metrics.csv` as `branch/num_*_updates` - the claim is checked, not asserted. The
phase's gradient-free updates also skip the PPO call entirely, so SFL-ACCEL is slightly *cheaper*
in wall-clock than ACCEL at the same env-step budget.

## The two runs

| run | command | seed |
|---|---|---|
| ACCEL baseline | `--preset accel` | 1 |
| SFL-ACCEL | `--preset sfl_accel` | 0 |

```bash
python -m tlab_ued.sweep --jobs accel:1 sfl_accel:0 --out_dir /workspace/tlab_ued --wait
```

**What one seed each does and does not buy.** Different seeds for the two arms means seed variance
and method effect are not separable: a 5-point solve-rate gap on the dev set is inside what seeds
do on their own in this domain, so the *primary* evidence from these two runs is not the headline
number. It is the mechanism evidence, which is per-run and seed-robust:

* `train/success_rate` - the difficulty of what the student is actually being trained on, over
  time. SFL's claim is that this sits near 0.5 and ACCEL's that it does not.
* `sfl/topk_learnability` vs `sfl/population_learnability` - how much better the kept levels are
  than the population they came from. If these converge, the phase is selecting nothing.
* `level_sampler/mean_p` - the buffer's mean success rate: is the curriculum drifting into
  impossible levels (→0) or into trivial ones (→1)?
* `level_diagnostics.json`, written before training from a BFS over the generator's own levels
  (and the mutator's children): what fraction of candidate levels are solvable at all, and how
  hard the solvable ones are. This is the ACCEL run's other job - it is the same generator for
  both methods, so one run's diagnostics describe both, and it is what tells us whether
  "learnability starves unsolvable levels" is even a claim with room to matter.

A third run (`sfl_accel_nophase`) and second seeds are the obvious follow-ups if the mechanism
signals look right.

## Checks before launching

```bash
python -m pytest tests/ -q                                     # includes CPU runs of every branch
python -m tlab_ued.parity --presets dr plr accel --num_updates 500   # must still be bit-identical
python -m tlab_ued.train --preset sfl_accel --smoke            # every path, two eval steps
python -m tlab_ued.level_diagnostics --preset accel            # the training distribution, no GPU
```

Parity matters here because the SFL work touched shared files (`scoring.py`, `teachers/base.py`,
`logging_utils.py`). The baselines must still reproduce upstream to the last bit.
