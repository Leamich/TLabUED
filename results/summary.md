# Results

Final numbers average the last 3 evaluations of each seed.

## Runs

| run_name                                   |   seed |   updates |
|:-------------------------------------------|-------:|----------:|
| accel_maxmc                                |      0 |     30000 |
| accel_maxmc                                |      1 |     30000 |
| accel_maxmc                                |      2 |     30000 |
| dr                                         |      0 |     30000 |
| plr_maxmc                                  |      0 |     30000 |
| sfl_accel_learnability                     |      0 |     30000 |
| sfl_accel_learnability_n64                 |      0 |     30000 |
| sfl_oracle_learnability_bfs                |      0 |     30000 |
| sfl_oracle_learnability_level              |      0 |     30000 |
| sfl_oracle_learnability_level_bfs          |      0 |     30000 |
| sfl_oracle_learnability_level_bfs          |      1 |     30000 |
| sfl_oracle_learnability_level_bfs          |      2 |     30000 |
| sfl_oracle_learnability_level_bfs_nomut    |      0 |     30000 |
| sfl_oracle_learnability_level_bfs_noverify |      0 |     30000 |

## Held-out solve rate

| run_name                                   |   mean |      std |   count |      sem |
|:-------------------------------------------|-------:|---------:|--------:|---------:|
| sfl_oracle_learnability_level_bfs          | 0.9375 |   0.0617 |       3 |   0.0356 |
| sfl_oracle_learnability_bfs                | 0.925  | nan      |       1 | nan      |
| sfl_accel_learnability                     | 0.8708 | nan      |       1 | nan      |
| sfl_oracle_learnability_level              | 0.8583 | nan      |       1 | nan      |
| sfl_oracle_learnability_level_bfs_noverify | 0.8583 | nan      |       1 | nan      |
| sfl_oracle_learnability_level_bfs_nomut    | 0.8417 | nan      |       1 | nan      |
| accel_maxmc                                | 0.7181 |   0.0891 |       3 |   0.0514 |
| sfl_accel_learnability_n64                 | 0.7083 | nan      |       1 | nan      |
| plr_maxmc                                  | 0.4542 | nan      |       1 | nan      |
| dr                                         | 0.275  | nan      |       1 | nan      |

## Per level

| run_name                                   |   SixteenRooms |   SixteenRooms2 |   Labyrinth |   LabyrinthFlipped |   Labyrinth2 |   StandardMaze |   StandardMaze2 |   StandardMaze3 |
|:-------------------------------------------|---------------:|----------------:|------------:|-------------------:|-------------:|---------------:|----------------:|----------------:|
| accel_maxmc                                |          0.967 |           0.522 |       0.867 |              0.844 |        0.833 |          0.411 |           0.844 |           0.456 |
| dr                                         |          0.733 |           0.367 |       0     |              0.3   |        0     |          0.5   |           0     |           0.3   |
| plr_maxmc                                  |          0.967 |           0.5   |       0.967 |              0.033 |        0.733 |          0.133 |           0.233 |           0.067 |
| sfl_accel_learnability                     |          0.7   |           0.633 |       0.867 |              1     |        0.8   |          0.967 |           1     |           1     |
| sfl_accel_learnability_n64                 |          0.833 |           0.667 |       0.633 |              1     |        0.667 |          0.933 |           0.067 |           0.867 |
| sfl_oracle_learnability_bfs                |          1     |           0.733 |       1     |              1     |        0.767 |          1     |           0.9   |           1     |
| sfl_oracle_learnability_level              |          1     |           0.967 |       1     |              0.933 |        1     |          0.567 |           1     |           0.4   |
| sfl_oracle_learnability_level_bfs          |          0.989 |           0.944 |       0.967 |              1     |        0.9   |          0.911 |           0.833 |           0.956 |
| sfl_oracle_learnability_level_bfs_nomut    |          0.867 |           0.767 |       0.733 |              1     |        0.7   |          1     |           0.7   |           0.967 |
| sfl_oracle_learnability_level_bfs_noverify |          0.6   |           0.867 |       1     |              1     |        0.767 |          0.7   |           0.967 |           0.967 |

## Budget actually spent

|                                                   |   num_env_steps |   num_updates |   branch/num_dr_updates |   branch/num_replay_updates |   branch/num_mutation_updates |   branch/num_sfl_updates |   branch/num_oracle_inserts |
|:--------------------------------------------------|----------------:|--------------:|------------------------:|----------------------------:|------------------------------:|-------------------------:|----------------------------:|
| ('sfl_accel_learnability_n64', 0)                 |      2.4576e+08 |         30000 |                     242 |                       14399 |                         14399 |                      960 |                         nan |
| ('sfl_oracle_learnability_level_bfs', 2)          |      2.4576e+08 |         30000 |                     242 |                       14399 |                         14399 |                      960 |                         120 |
| ('sfl_oracle_learnability_level_bfs', 1)          |      2.4576e+08 |         30000 |                     242 |                       14399 |                         14399 |                      960 |                         120 |
| ('sfl_oracle_learnability_level_bfs', 0)          |      2.4576e+08 |         30000 |                     242 |                       14399 |                         14399 |                      960 |                         120 |
| ('sfl_oracle_learnability_level', 0)              |      2.4576e+08 |         30000 |                     242 |                       14399 |                         14399 |                      960 |                         120 |
| ('sfl_oracle_learnability_bfs', 0)                |      2.4576e+08 |         30000 |                     242 |                       14399 |                         14399 |                      960 |                         120 |
| ('sfl_accel_learnability', 0)                     |      2.4576e+08 |         30000 |                     242 |                       13199 |                         13199 |                     3360 |                         nan |
| ('plr_maxmc', 0)                                  |      2.4576e+08 |         30000 |                     nan |                         nan |                           nan |                      nan |                         nan |
| ('dr', 0)                                         |      2.4576e+08 |         30000 |                     nan |                         nan |                           nan |                      nan |                         nan |
| ('accel_maxmc', 2)                                |      2.4576e+08 |         30000 |                     nan |                         nan |                           nan |                      nan |                         nan |
| ('accel_maxmc', 1)                                |      2.4576e+08 |         30000 |                     nan |                         nan |                           nan |                      nan |                         nan |
| ('accel_maxmc', 0)                                |      2.4576e+08 |         30000 |                     nan |                         nan |                           nan |                      nan |                         nan |
| ('sfl_oracle_learnability_level_bfs_nomut', 0)    |      2.4576e+08 |         30000 |                     242 |                       14399 |                         14399 |                      960 |                         120 |
| ('sfl_oracle_learnability_level_bfs_noverify', 0) |      2.4576e+08 |         30000 |                     250 |                       14875 |                         14875 |                        0 |                         119 |

## Throughput

| run_name                                   |   time_delta |   steps_per_second |   hours_per_30k_updates |
|:-------------------------------------------|-------------:|-------------------:|------------------------:|
| accel_maxmc                                |         71.7 |            37593.6 |                     2.4 |
| dr                                         |         74.4 |            27519.4 |                     2.5 |
| plr_maxmc                                  |         64.2 |            31881.2 |                     2.1 |
| sfl_accel_learnability                     |         26.6 |            76956.8 |                     0.9 |
| sfl_accel_learnability_n64                 |         75.9 |            26999.2 |                     2.5 |
| sfl_oracle_learnability_bfs                |         76.3 |            26840.4 |                     2.5 |
| sfl_oracle_learnability_level              |         74   |            27688   |                     2.5 |
| sfl_oracle_learnability_level_bfs          |         76.8 |            26678.3 |                     2.6 |
| sfl_oracle_learnability_level_bfs_nomut    |         74.5 |            27487   |                     2.5 |
| sfl_oracle_learnability_level_bfs_noverify |         78.1 |            26235.5 |                     2.6 |

## Oracle diagnostics

Averaged over the phases after warm-up. `gain` is measured learnability of the oracle's picks over the uniform controls beside them (1.0 = chance); `predicted` against `picked` is the winner's curse; `rank corr` and `brier` are on the controls, the only sample the selection did not restrict.

|                                                |   gain |   picked |   control |   predicted |   rank corr |   brier |   loss |
|:-----------------------------------------------|-------:|---------:|----------:|------------:|------------:|--------:|-------:|
| ('sfl_oracle_learnability_bfs', 0)             |  1.158 |    0.018 |     0.01  |       0.131 |       0.015 |   0.023 |  0.547 |
| ('sfl_oracle_learnability_level', 0)           |  1.274 |    0.02  |     0.009 |       0.122 |       0.014 |   0.016 |  0.417 |
| ('sfl_oracle_learnability_level_bfs', 0)       |  1.142 |    0.019 |     0.009 |       0.104 |      -0.023 |   0.011 |  0.409 |
| ('sfl_oracle_learnability_level_bfs', 1)       |  1.08  |    0.018 |     0.009 |       0.096 |       0.04  |   0.011 |  0.396 |
| ('sfl_oracle_learnability_level_bfs', 2)       |  1.041 |    0.018 |     0.009 |       0.107 |       0.102 |   0.011 |  0.405 |
| ('sfl_oracle_learnability_level_bfs_nomut', 0) |  1.01  |    0.016 |     0.008 |       0.115 |       0.013 |   0.012 |  0.352 |

## Curriculum diagnostics

See `results/figs/curriculum.png`. Columns present: train/success_rate, train/learnability, level_sampler/mean_p, level/mean_num_blocks, oracle/selection_gain, oracle/control_rank_corr, oracle/control_brier, sfl/topk_learnability vs sfl/population_learnability, oracle/predicted_learnability vs oracle/selected_learnability, oracle/buffer_mean_p vs level_sampler/mean_p.
