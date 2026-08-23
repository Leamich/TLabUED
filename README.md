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

DR is a separate teacher rather than "PLR with replay off" because upstream's DR genuinely differs:
it wraps the env in `AutoResetWrapper` and carries env state across updates, where PLR/ACCEL use
`AutoReplayWrapper` and reset onto a chosen batch of levels each update.

Fidelity is checked, not assumed:

```bash
$PY -m tlab_ued.parity --presets dr plr accel --num_updates 500
```

runs both implementations in-process on the same seed and diffs every logged scalar. Expected
output: identical, to the last bit.

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

`tests/test_config.py` covers the freeze guard and config round-trips; `tests/test_teachers.py`
runs each teacher for a few updates on a tiny config and checks that a written checkpoint reads
back with the expected parameter shapes.

## Notes

- Logging defaults to `WANDB_MODE=offline`: no API key needed mid-run, and every scalar is
  mirrored to `metrics.csv`. `wandb sync runs/.../wandb/offline-*` uploads later if wanted.
- Notebook outputs should be stripped before committing:
  `jupyter nbconvert --clear-output --inplace notebooks/baseline.ipynb`.
