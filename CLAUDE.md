# Working in this repo

UED research code: a frozen student, a pluggable teacher, on JaxUED mazes. Read
[`docs/TASK.md`](docs/TASK.md) first - it is the assignment this code answers, and several of its
constraints are hard.

## Non-negotiables

1. **Never change the student.** `src/tlab_ued/student.py` is a verbatim copy of upstream's PPO+LSTM
   code and the `STUDENT_DEFAULTS` in `config.py` are its hyperparameters. Changing either turns a
   curriculum comparison into hyperparameter tuning and invalidates every result. `assert_student_frozen`
   enforces it; do not reach for `--allow_student_changes` to make a run go green.
2. **Never train, tune or generate against the eval levels.** The eight prefab levels
   (`SixteenRooms`, `Labyrinth`, `StandardMaze`, ...) are held out. The graders re-score our
   checkpoints on a secret set of the same format, so overfitting the dev set buys nothing.
3. **Never change the policy architecture.** Submitted checkpoints are loaded by the graders'
   harness into the original `ActorCritic`. Extra state in the checkpoint is fine (they read
   `loaded["params"]`); a changed `params` tree is a failed submission.
4. **Never edit `third_party/jaxued`.** It is vendored at a pinned SHA and is the parity reference.

## Where things go

| Change | File |
|---|---|
| new score function | `scoring.py` + `@register_score_fn` |
| new curriculum logic | `teachers/<name>.py` + registry in `teachers/__init__.py` |
| new level generation/mutation | `levels.py` + `@register_generator` / `@register_mutator` |
| new flag | `config.py` (`EXTRA_DEFAULTS` if it has no upstream counterpart) |

`train.py` should stay teacher-agnostic. If an idea seems to need a change there, it probably
belongs behind a teacher method instead.

## Conventions

- PRNG discipline: branch bodies keep upstream's exact `jax.random.split` arity and order. An
  equivalent-looking rearrangement silently changes every result and breaks `parity.py`.
- Long runs are launched detached via `sweep.py`, never in a notebook kernel.
- Results are read from `runs/*/*/metrics.csv` by `analysis.py` - no GPU, no wandb needed.
- After touching anything on the training path, re-run
  `python -m tlab_ued.parity --presets dr plr accel --num_updates 500`; the baselines must still
  match upstream exactly.
