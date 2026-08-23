"""UED experiments on JaxUED mazes: frozen student, pluggable teacher.

Layout:
    student.py     frozen PPO+LSTM student (verbatim from upstream)
    teachers/      level-selection strategies: dr, plr, accel, and yours
    scoring.py     score-function registry (what "worth training on" means)
    levels.py      level generator / mutator registry
    level_diagnostics.py  BFS over generated levels: solvable fraction, difficulty
    train.py       teacher-agnostic training loop
    evaluate.py    checkpoint -> held-out solve rates
    sweep.py       resumable, detached job queue
    parity.py      our trainer vs upstream, same seed, diffed
    analysis.py    metrics.csv -> tables and plots
"""

__version__ = "0.1.0"
