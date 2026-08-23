# Results

Final numbers average the last 3 evaluations of each seed.

## Runs

| run_name               |   seed |   updates |
|:-----------------------|-------:|----------:|
| accel_maxmc            |      0 |     30000 |
| accel_maxmc            |      1 |     30000 |
| dr                     |      0 |     30000 |
| plr_maxmc              |      0 |     30000 |
| sfl_accel_learnability |      0 |     30000 |

## Held-out solve rate

| run_name               |   mean |      std |   count |      sem |
|:-----------------------|-------:|---------:|--------:|---------:|
| sfl_accel_learnability | 0.8708 | nan      |       1 | nan      |
| accel_maxmc            | 0.6792 |   0.0825 |       2 |   0.0583 |
| plr_maxmc              | 0.4542 | nan      |       1 | nan      |
| dr                     | 0.275  | nan      |       1 | nan      |

## Per level

| run_name               |   SixteenRooms |   SixteenRooms2 |   Labyrinth |   LabyrinthFlipped |   Labyrinth2 |   StandardMaze |   StandardMaze2 |   StandardMaze3 |
|:-----------------------|---------------:|----------------:|------------:|-------------------:|-------------:|---------------:|----------------:|----------------:|
| accel_maxmc            |          1     |           0.283 |       0.8   |              0.767 |        0.783 |          0.583 |           0.783 |           0.433 |
| dr                     |          0.733 |           0.367 |       0     |              0.3   |        0     |          0.5   |           0     |           0.3   |
| plr_maxmc              |          0.967 |           0.5   |       0.967 |              0.033 |        0.733 |          0.133 |           0.233 |           0.067 |
| sfl_accel_learnability |          0.7   |           0.633 |       0.867 |              1     |        0.8   |          0.967 |           1     |           1     |

## Budget actually spent

|                               |   num_env_steps |   num_updates |   branch/num_dr_updates |   branch/num_replay_updates |   branch/num_mutation_updates |   branch/num_sfl_updates |
|:------------------------------|----------------:|--------------:|------------------------:|----------------------------:|------------------------------:|-------------------------:|
| ('sfl_accel_learnability', 0) |      2.4576e+08 |         30000 |                     242 |                       13199 |                         13199 |                     3360 |
| ('dr', 0)                     |      2.4576e+08 |         30000 |                     nan |                         nan |                           nan |                      nan |
| ('accel_maxmc', 0)            |      2.4576e+08 |         30000 |                     nan |                         nan |                           nan |                      nan |
| ('plr_maxmc', 0)              |      2.4576e+08 |         30000 |                     nan |                         nan |                           nan |                      nan |
| ('accel_maxmc', 1)            |      2.4576e+08 |         30000 |                     nan |                         nan |                           nan |                      nan |

## Throughput

| run_name               |   time_delta |   steps_per_second |   hours_per_30k_updates |
|:-----------------------|-------------:|-------------------:|------------------------:|
| accel_maxmc            |         37.5 |            56411.4 |                     1.3 |
| dr                     |         74.4 |            27519.4 |                     2.5 |
| plr_maxmc              |         64.2 |            31881.2 |                     2.1 |
| sfl_accel_learnability |         26.6 |            76956.8 |                     0.9 |

## Curriculum diagnostics

See `results/figs/curriculum.png`. Columns present: train/success_rate, train/learnability, level_sampler/mean_p, level/mean_num_blocks.
