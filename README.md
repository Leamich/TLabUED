# tlab_ued - Unsupervised Environment Design on JaxUED mazes

A frozen PPO+LSTM student, a **pluggable teacher**. This repo reproduces the three reference
curricula from [`docs/TASK.md`](docs/TASK.md) - DR, PLR-perp and ACCEL - and provides the
scaffolding to try better ones.

Built on [JaxUED](https://github.com/DramaCow/jaxued), pinned at commit `0f8f128` and vendored
unmodified into `third_party/`.

## Quickstart (RunPod A100, or any Linux box with a CUDA 12 driver)

```bash
git clone <this repo> && cd tlab_ued
bash scripts/bootstrap.sh          # clones jaxued @ pinned SHA, builds a 3.11 venv, installs the pins
```

Then open [`notebooks/baseline.ipynb`](notebooks/baseline.ipynb), which walks the whole path:
probe -> bootstrap -> smoke -> parity -> checkpoint compatibility -> full sweep -> plots.

Or drive it from the shell:

```bash
PY=/workspace/venvs/jaxued/bin/python

# one run
$PY -m tlab_ued.train --preset accel --seed 0 --out_dir /workspace/tlab_ued

# the baseline sweep, detached and resumable
$PY -m tlab_ued.sweep --presets dr plr accel --seeds 0 1 2 --out_dir /workspace/tlab_ued --wait

# what the curriculum is made of: solvable %, optimal-step distribution (no GPU needed)
$PY -m tlab_ued.level_diagnostics --preset accel

# evaluate a checkpoint on the held-out levels
$PY -m tlab_ued.train --mode eval --checkpoint_directory /workspace/tlab_ued/checkpoints/accel_maxmc/0
```

A full run is 30000 updates (~245M env steps), roughly 1-3 hours on an A100.

## Layout

```
src/tlab_ued/
  student.py      FROZEN - verbatim copy of upstream's PPO+LSTM student
  teachers/       level-selection strategies: dr.py, plr.py, accel.py, <yours>.py
  scoring.py      score-function registry - what "worth training on" means
  levels.py       level generator / mutator registry
  level_diagnostics.py  BFS over generated levels: how many are solvable, and how hard
  train.py        teacher-agnostic training loop, checkpointing, eval protocol
  evaluate.py     checkpoint -> held-out solve rates
  sweep.py        detached, resumable job queue
  parity.py       our trainer vs upstream, same seed, diffed
  analysis.py     metrics.csv -> tables and plots
notebooks/baseline.ipynb
scripts/bootstrap.sh
```

Outputs (all gitignored, all under `--out_dir`):

```
runs/<run_name>/<seed>/metrics.csv    one row per eval step - what the plots read
runs/<run_name>/<seed>/meta.json      git SHAs, JAX version, GPU, full config
runs/<run_name>/<seed>/train.log      stdout
runs/<run_name>/<seed>/level_diagnostics.json   the launch-time BFS report
checkpoints/<run_name>/<seed>/        orbax checkpoints + config.json - what gets submitted
results/<run_name>/<seed>/            eval-mode outputs
```

## The rules this code enforces

Three constraints from `docs/TASK.md` are load-bearing, so they are enforced rather than
remembered:

1. **The student is frozen.** `assert_student_frozen` runs at the top of every training run and
   refuses any deviation from upstream's PPO hyperparameters, budget or `agent_view_size`.
   `--allow_student_changes` exists for deliberate ablations and taints the run.
2. **The held-out levels are for measurement only.** They are never used for training, tuning, or
   as templates for generation. If you need a validation set, build a separate one.
3. **Checkpoints must load in the graders' harness**, which restores untargeted and reads
   `loaded["params"]` into the original `ActorCritic`. Teacher state may ride along in the
   checkpoint; the policy architecture may not change. The notebook proves this each run by
   evaluating our checkpoint with upstream's unmodified `maze_plr.py --mode eval`.

## Baselines

| preset | upstream equivalent |
|---|---|
| `dr` | `python examples/maze_dr.py` |
| `plr` | `python examples/maze_plr.py` (PLR-perp: no exploratory gradient updates) |
| `accel` | `python examples/maze_plr.py --use_accel` |

Plus our own, which has no upstream counterpart:

| preset | what it is |
|---|---|
| `sfl_accel` | ACCEL with SFL's learnability score in place of MaxMC ([docs/experiments/sfl_accel.md](docs/experiments/sfl_accel.md)) |
| `sfl_accel_nophase` | the ablation: learnability scoring, no evaluation phase |

DR is a separate teacher rather than "PLR with replay off" because upstream's DR genuinely differs:
it wraps the env in `AutoResetWrapper` and carries env state across updates, where PLR/ACCEL use
`AutoReplayWrapper` and reset onto a chosen batch of levels each update.

Fidelity is checked, not assumed:

```bash
$PY -m tlab_ued.parity --presets dr plr accel --num_updates 500
```

runs both implementations in-process on the same seed and diffs every logged scalar. Expected
output: identical, to the last bit.

## SFL-ACCEL

Our method. `p(1-p)` over a level's success rate (Rutherford et al., 2024) decides what enters the
level buffer; ACCEL's mutation operator evolves it from there. Because a single rollout gives one
episode per level - and `p(1-p)` of a single Bernoulli sample is exactly 0 either way - levels are
scored with several attempts each (the 32 envs carry `32/k` distinct levels), replayed levels carry
a decaying running estimate of `p`, and every `--sfl_period` updates a phase evaluates a large batch
of fresh random levels at a frozen policy and keeps the most learnable.

The phase is paid for out of the frozen budget, not added to it: it costs what ACCEL spends on its
DR branch, and `--replay_prob 1.0` keeps the count of *gradient* updates within 0.1% of ACCEL's.
Every run prints its expected budget split before the first update and logs the actual branch
counts to `metrics.csv`:

```bash
$PY -m tlab_ued.sweep --jobs accel:1 sfl_accel:0 --out_dir /workspace/tlab_ued --wait
```

Curriculum diagnostics land in the CSV alongside the solve rates: `train/success_rate` (how hard
the levels being trained on actually are), `sfl/topk_learnability` vs `sfl/population_learnability`
(how much the phase's selection is buying), `level_sampler/mean_p`, and `branch/num_*_updates`.

## Level diagnostics

Every training run starts by BFS-ing a sample of the levels its teacher will generate, and prints
what it found before the first update:

* **solvable %** - split into "the goal is walled off" and "the optimal solution is longer than
  the 250-step episode". Both are levels the student cannot learn from, but they have different
  causes.
* **optimal steps** - the true minimum number of *env* steps, searched over `(position,
  direction)` with `left` / `right` / `forward`, ending the way the env ends an episode: by
  stepping into the goal cell. Turning costs a step, and the goal is not walkable, so this is not
  the grid distance.
* **best return** - what that optimal play is worth under the env's time penalty
  (`1 - 0.9 * steps / 250`), i.e. the ceiling on `return/mean` for that distribution.
* **difficulty buckets, reachable area, wall count** - the shape of the distribution, not just its
  mean. A curriculum drifting into "trivial" or into "unsolvable" is visible here long before the
  eval curve reacts.

For a mutation teacher (ACCEL) the same report is printed for the mutated children, which is how
you tell a mutation operator that ratchets difficulty up from one that mostly seals goals off.

The report goes to stdout and to `runs/<run_name>/<seed>/level_diagnostics.json`. It draws from
its own PRNG key, so `--diagnose_levels 0` (off) and `--diagnose_levels 8192` produce
bit-identical training runs. Run it standalone to compare generator settings without training:

```bash
$PY -m tlab_ued.level_diagnostics --preset accel --n_walls 60 --diagnose_levels 8192
```

## Adding an idea

1. **A new score function** - `scoring.py`, `@register_score_fn("my_score")`, then
   `--score_function my_score`. It receives a `RolloutSignals` bundle (obs, actions, log-probs,
   values, advantages, targets, returns, levels), not just the two arrays MaxMC and PVL use.
2. **A new curriculum** - `teachers/my_idea.py`, subclass `Teacher`, register in
   `teachers/__init__.py`, then `--teacher my_idea`. Whatever memory it needs goes in
   `teacher_state`, an opaque pytree carried through the loop and the checkpoint.
3. **A new level generator or mutator** - `levels.py`, `@register_generator` / `@register_mutator`.

Add a preset in `config.py` and the sweep, parity, evaluation and plotting machinery all apply
unchanged.

## Tests

```bash
$PY -m pytest tests -q
```

`tests/test_sfl_accel.py` covers the learnability score, the budget arithmetic against ACCEL's,
and a CPU run through every SFL branch; `tests/test_config.py` covers the freeze guard and config
round-trips;
`tests/test_level_diagnostics.py` checks the BFS against a brute-force search over the real
`Maze.step_env` transitions; `tests/test_teachers.py`
runs each teacher for a few updates on a tiny config and checks that a written checkpoint reads
back with the expected parameter shapes.

## Notes

- Logging defaults to `WANDB_MODE=offline`: no API key needed mid-run, and every scalar is
  mirrored to `metrics.csv`. `wandb sync runs/.../wandb/offline-*` uploads later if wanted.
- Notebook outputs should be stripped before committing:
  `jupyter nbconvert --clear-output --inplace notebooks/baseline.ipynb`.
