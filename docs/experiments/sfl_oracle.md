# SFL-ORACLE: predicting learnability instead of measuring it

**Question.** SFL's score is the right one - `p(1-p)`, where `p` is the student's success rate -
and [`sfl_accel.md`](sfl_accel.md) is about putting it in charge of ACCEL's buffer. Its problem is
that `p` has to be *bought*: 224 candidate levels x 4 attempts is 28 rollout-only updates every
250, and 224 is a narrow window on a space of 2^169 wall maps. Can a small model *predict* `p` from
the level itself well enough to (a) rank a population 36x larger for free, (b) cut the measurement
budget by three quarters, and (c) still pick better levels than measurement does?

Two facts make this less obviously silly than "learn the value function again":

* **The oracle sees what the student cannot.** The student gets a 5x5 egocentric view and an LSTM.
  The oracle gets the whole level - wall map, goal, agent pose - and, optionally, the exact BFS
  solution. Predicting "will this policy solve this maze" is a much easier problem from above than
  from inside.
* **Measurement is a *worse* estimator than it looks.** With `k = 4` attempts, `p_hat` lands in
  {0, .25, .5, .75, 1}, so `p_hat(1-p_hat)` takes exactly **three** distinct values: 0, 0.1875,
  0.25. SFL's "top 32 of 224" is therefore mostly a tie-break among however many levels happened to
  score 2-out-of-4, resolved by array index. A model that pools information across thousands of
  levels can be the lower-variance estimator of the two - and it produces a continuous ranking
  rather than three buckets.

## The method

The model is in [`oracle.py`](../../src/tlab_ued/oracle.py), the teacher in
[`teachers/sfl_oracle.py`](../../src/tlab_ued/teachers/sfl_oracle.py). It is a ~50k-parameter
convnet over 7 level planes (walls, goal, agent, four facing planes carried at the agent's cell)
plus four scalars from `level_diagnostics.solve_levels` (optimal steps, solvable-within-the-episode,
reachable fraction, wall density), trained online with a binomial NLL on the labels every rollout
already produces. Nothing about the student changes; nothing costs env steps.

Four places the prediction gets spent, in rising order of how much trust they need:

1. **The phase becomes a cascade.** `--oracle_num_proposals` (8192) fresh levels are ranked by
   predicted learnability; the best `--sfl_num_levels` (64) are played; the top `--sfl_topk` (32) by
   *measured* learnability enter the buffer. Selection is 8192:64 by prediction and then 2:1 by
   measurement, at 8 rollout updates instead of 28. Ranking those 8192 by rollout instead would
   cost 1024 updates per phase - over 120 phases, four times the entire training budget.
2. **A control group makes it falsifiable.** `--oracle_control_levels` (8) of those 64 are drawn
   uniformly at random rather than from the top of the ranking, and are measured in the same
   updates at the same policy. `oracle/selection_gain` - measured learnability of the picks over
   measured learnability of the controls - is therefore an experiment the run performs on itself,
   120 times, not an assumption.
3. **The buffer is re-scored for free.** A PLR score is as old as the last replay of that level.
   Every phase, all 4000 entries are re-scored in one forward pass, so a level that stopped being
   learnable 3000 updates ago stops being sampled without having to be replayed first. Only
   `scores` is touched: `levels_extra["p"]` stays the last measured value, so
   `level_sampler/mean_p` keeps meaning what it means in the SFL run, and the replay branch
   overwrites the prediction with a measurement the moment the level is played.
4. **Mutation stops being blind.** Each parent emits `--oracle_mutation_proposals` (8) children and
   the one predicted closest to a coin flip is the one that gets the rollout. Same operator, same
   edit distribution - the teacher just gets to look before it spends.

`--no-oracle_verify` deletes stage 1's second half: the phase collapses to a single update that
inserts the oracle's top 32 on prediction alone *and* runs an ordinary gradient replay, so all 960
measurement updates return to the student. Whether that works is a question about calibration, and
it is the arm most likely to fail interestingly.

**The warm-up.** For the first `--oracle_warmup_updates` (1000) updates the ranking is ignored and
selections are uniform random. The head is zero-initialised, so an unfit oracle predicts 0.5
everywhere and is uninformative rather than confidently wrong; the warm-up means the first four
phases do not chase whatever asymmetry the initialisation happened to have.

## Budget

The frozen quantity is env steps: 30000 updates x 32 envs x 256 steps = **245.76M**, identical in
every row below, because every update is exactly one rollout. What differs is how many of those
updates carry a gradient - which is the point. Measurement is overhead, and the oracle converts
overhead into training.

| run | env steps | gradient updates | mutation | rollout-only phase |
|---|---|---|---|---|
| `accel` | 245.76M | 13,333 | 13,333 | - |
| `sfl_accel` | 245.76M | 13,320 | 13,320 | 3,360 |
| `sfl_accel_cheap` | 245.76M | 14,520 | 14,520 | 960 |
| `sfl_oracle` | 245.76M | 14,520 | 14,520 | 960 |
| `sfl_oracle_noverify` | 245.76M | 15,000 | 15,000 | 0 |

`oracle_budget_report` computes this and every run prints it before the first update;
`branch/num_*_updates` in `metrics.csv` is the realised count, so the table is checked rather than
asserted. `branch/num_oracle_inserts` counts *phases*, and is deliberately not a branch counter:
without verification an insertion costs no update at all, and the four branch counters have to keep
summing to 30000.

**This is why `sfl_accel_cheap` exists.** `sfl_oracle` has 9% more gradient updates than `accel`,
so a win over ACCEL alone would not tell us whether the cascade selects better levels or merely
trains more. `sfl_accel_cheap` is SFL with the oracle's phase length and no oracle: same 960
reserved updates, same 14,520 gradient updates, 64 measured candidates picked at random instead of
8192 ranked. Oracle vs cheap-SFL is the clean comparison; cheap-SFL vs SFL is the price of a
smaller measured population.

**Wall clock is not free even though env steps are.** The oracle takes `--oracle_train_steps` (2)
Adam steps per update on 256 levels, and the BFS features run on every batch it observes, every
mutation candidate and every proposal population. All of it is small next to 5 PPO epochs over
32x256 LSTM steps, but "small" is a claim about a GPU, not a theorem: check `steps_per_second` in
the CSV against the ACCEL run's before trusting the comparison, and fall back to
`--oracle_features level` (no solver in the loop) if the BFS while-loop turns out to dominate.

## The arms

| preset | what it changes | what it isolates |
|---|---|---|
| `accel` | - | the baseline to beat |
| `sfl_accel` | learnability, 224-level phase | the previous experiment |
| `sfl_accel_cheap` | 64-level phase | the oracle's budget twin: is the win the cascade or the shorter phase? |
| `sfl_oracle` | the full cascade | the method |
| `sfl_oracle_noverify` | no verification rollouts | can the oracle be trusted alone? |
| `sfl_oracle_bfs` | solver features only, no convnet | is a learned view of the map worth anything over "how long is the shortest path"? |
| `sfl_oracle_level` | convnet only, no solver | can it learn difficulty without being handed the path? |
| `sfl_oracle_nomut` | blind mutation | how much of the effect is the guided mutation rather than the phase? |

```bash
# the comparison
python -m tlab_ued.sweep --jobs sfl_oracle:0 sfl_accel_cheap:0 --out_dir /workspace/tlab_ued --wait
# the follow-ups, in the order they are worth running
python -m tlab_ued.sweep --jobs sfl_oracle_bfs:0 sfl_oracle_noverify:0 sfl_oracle_nomut:0 --wait
```

## Gate probe: 3000 updates, RTX 4090

Run before committing GPU time, as `--run_name oracle_probe --num_updates 3000` (warm-up crossed at
1000, so nine phases used the ranking). Full suite green, all three baselines still bit-identical to
upstream, 48,065 oracle parameters, 454s wall - **1.26h projected for a full run, against ACCEL's
1.3h**, so the oracle costs ~27% throughput and buys it back in gradient updates. Dropping the BFS
features (`--oracle_features level`) changed the wall clock by 4s in 260, so the solver in the loop
is *not* the cost - the Adam steps are.

| | value | reading |
|---|---|---|
| `selection_gain`, median over 9 phases | **2.77** | the oracle's picks are ~3x more learnable than levels nobody picked |
| `selected` / `control` learnability | 0.099 / 0.058 | out of sample: it commits to a ranking, *then* the rollouts run |
| `predicted` → `selected` learnability | 0.229 → 0.099 | the winner's curse, and it is large |
| `rank_corr` on the shortlist | 0.03 | **uninformative by construction** - see below |
| `control_brier`, late phases | 0.004-0.077 | it predicts uniformly drawn levels well |
| buffer `p`: measured vs oracle | 0.565 vs 0.902 | the staleness the re-scoring exists to fix |

Three things this changed:

1. **`rank_corr` on the verified shortlist is a broken measurement, not a failing model.** After
   warm-up the oracle picks those 64 levels *because* it predicts them all at the same `p`; a rank
   correlation over a range-restricted sample carries no information about the ranking that produced
   it. The gain against the controls is positive over the same phases, which it could not be if the
   ranking were noise. Fixed by adding `oracle/control_rank_corr`, computed on the uniform controls
   only, and by raising `--oracle_control_levels` from 8 to 16 so that estimate is usable per phase.
2. **`selection_gain` needed a floor.** Late phases produced control groups scoring exactly zero
   learnability, and one phase logged a gain of 88,169 - enough to destroy any average over the
   column. The denominator is now floored at the smallest non-zero value it can take.
3. **The winner's curse is real and large** (0.229 predicted vs 0.099 realised). This is exactly
   what stage 2's verification absorbs, and it is the strongest single argument against the
   `sfl_oracle_noverify` arm working. Worth predicting in advance: if noverify loses, this is why.

A caveat to carry into the write-up: 2.8x is *lower* than the 5-10x that `sfl_accel`'s phase logs as
top-32-over-population. Those numbers are not comparable - SFL selects and evaluates on the same
4-sample statistic, so its ratio is inflated by regression to the mean, while this one is measured
out of sample. Saying so is part of the result, not a hedge around it.

## What to look at, and what would falsify it

Every one of these is per-run and seed-robust, which is what makes a two-run comparison worth
anything before there is budget for three seeds:

* **`oracle/selection_gain`** - the headline. Measured learnability of the oracle's 56 picks over
  the 8 uniform controls, once per phase. **1.0 means the oracle is choosing at chance and the
  method is dead**; the interesting question is whether it climbs above ~2 and stays there as the
  policy moves. Note that for this teacher `sfl/population_learnability` is the *shortlist* mean,
  not a random population - the control group is the population baseline.
* **`oracle/control_rank_corr`** - Spearman between predicted and measured `p` over the 16 uniform
  controls. This, not `oracle/rank_corr`, is the quantity the whole method rests on: the shortlist
  version is range-restricted by the selection that produced it and sits near zero even when the
  ranking works (see the gate probe above).
* **`oracle/control_brier` vs `oracle/brier`** - calibration on levels nobody chose vs on the whole
  verified set. Most of the oracle's training data comes from buffer levels, so its predictions on
  fresh random levels are extrapolation; if `control_brier` is much worse, the model is describing
  the curriculum rather than the level space, and the cascade is selecting on a distribution it has
  no business ranking.
* **`oracle/predicted_learnability` vs `oracle/selected_learnability`** - the winner's curse. An
  argmax over a prediction selects partly for prediction error; a large gap between what the oracle
  expected and what the rollouts found is that error, and it is the specific failure the
  verification stage exists to absorb.
* **`oracle/buffer_mean_p` vs `level_sampler/mean_p`** - the oracle's current view of the buffer
  against the last measurement of it. These diverging is the staleness the re-scoring is there to
  fix, and the size of the gap says how much that mechanism can matter.
* **`train/success_rate`** - unchanged in meaning from the SFL run: how hard the levels the student
  is actually training on are. SFL's claim is that this sits near 0.5.

Three ways this fails, all of which the run should be able to tell apart:

1. **The oracle cannot predict `p` at all** (`rank_corr` near 0). Then `sfl_oracle` degenerates to
   `sfl_accel_cheap` with extra steps, and should match it - if it is *worse*, the ranking is
   actively harmful, which would mean it is selecting on prediction error.
2. **It predicts `p` but learnability does not transfer.** `selection_gain` high, held-out solve
   rate flat. That would be a result about learnability rather than about the oracle, and the
   `sfl_accel` run is the reference for it.
3. **It only learns "is the goal reachable".** Then `sfl_oracle_bfs` matches `sfl_oracle` and the
   convnet is decoration. That is a cheap thing to find out and worth knowing either way: it would
   say the useful signal in this domain is solvability plus path length, which is a fact about the
   generator, not about the method.

## Checks before launching

```bash
python -m pytest tests/ -q                                          # CPU runs of every branch
python -m tlab_ued.parity --presets dr plr accel --num_updates 500  # must still be bit-identical
python -m tlab_ued.train --preset sfl_oracle --smoke                # every path, two eval steps
python -m tlab_ued.train --preset sfl_oracle_noverify --smoke       # the other phase shape
```

Parity matters because this work touched `teachers/sfl_accel.py` - two hooks (`after_rollout`,
`propose_children`) that are no-ops there. The baselines do not go through that file at all, but
the check is cheap and the whole comparison is worthless if they have drifted.

One thing the smoke run will not tell you: whether the oracle's wall-clock cost is acceptable at
full scale. Compare `steps_per_second` from the first few eval steps of a real `sfl_oracle` run
against the `accel` run's before committing to a whole sweep.
