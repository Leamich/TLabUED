# Results

The final number of a seed is an EMA (gamma = 0.8) over all evaluations of each seed; a method's is the mean over seeds.

## Runs

| run_name                          |   seed |   updates |
|:----------------------------------|-------:|----------:|
| accel_maxmc                       |      0 |     30000 |
| accel_maxmc                       |      1 |     30000 |
| accel_maxmc                       |      2 |     30000 |
| dr                                |      0 |     30000 |
| plr_maxmc                         |      0 |     30000 |
| sfl_accel_learnability            |      0 |     30000 |
| sfl_accel_learnability_n64        |      0 |     30000 |
| sfl_oracle_learnability_level     |      0 |     30000 |
| sfl_oracle_learnability_level_bfs |      0 |     30000 |
| sfl_oracle_learnability_level_bfs |      1 |     30000 |
| sfl_oracle_learnability_level_bfs |      2 |     30000 |

## Held-out solve rate

| run_name                          |   mean |      std |   count |      sem |
|:----------------------------------|-------:|---------:|--------:|---------:|
| sfl_oracle_learnability_level_bfs | 0.8739 |   0.0841 |       3 |   0.0486 |
| sfl_accel_learnability            | 0.8541 | nan      |       1 | nan      |
| sfl_oracle_learnability_level     | 0.7655 | nan      |       1 | nan      |
| sfl_accel_learnability_n64        | 0.6899 | nan      |       1 | nan      |
| accel_maxmc                       | 0.6646 |   0.0716 |       3 |   0.0414 |
| plr_maxmc                         | 0.4337 | nan      |       1 | nan      |
| dr                                | 0.2613 | nan      |       1 | nan      |

## Per level

| run_name                          |   SixteenRooms |   SixteenRooms2 |   Labyrinth |   LabyrinthFlipped |   Labyrinth2 |   StandardMaze |   StandardMaze2 |   StandardMaze3 |
|:----------------------------------|---------------:|----------------:|------------:|-------------------:|-------------:|---------------:|----------------:|----------------:|
| accel_maxmc                       |          0.946 |           0.409 |       0.876 |              0.768 |        0.809 |          0.408 |           0.649 |           0.451 |
| dr                                |          0.832 |           0.26  |       0     |              0.25  |        0     |          0.516 |           0.008 |           0.225 |
| plr_maxmc                         |          0.954 |           0.46  |       0.965 |              0.161 |        0.514 |          0.055 |           0.255 |           0.105 |
| sfl_accel_learnability            |          0.835 |           0.627 |       0.765 |              0.971 |        0.776 |          0.942 |           0.925 |           0.991 |
| sfl_accel_learnability_n64        |          0.849 |           0.754 |       0.654 |              0.877 |        0.504 |          0.857 |           0.156 |           0.87  |
| sfl_oracle_learnability_level     |          1     |           0.912 |       0.923 |              0.931 |        0.736 |          0.435 |           0.849 |           0.337 |
| sfl_oracle_learnability_level_bfs |          0.94  |           0.838 |       0.919 |              0.885 |        0.787 |          0.913 |           0.79  |           0.919 |

## Budget actually spent

|                                          |   num_env_steps |   num_updates |   branch/num_dr_updates |   branch/num_replay_updates |   branch/num_mutation_updates |   branch/num_sfl_updates |   branch/num_oracle_inserts |
|:-----------------------------------------|----------------:|--------------:|------------------------:|----------------------------:|------------------------------:|-------------------------:|----------------------------:|
| ('dr', 0)                                |      2.4576e+08 |         30000 |                     nan |                         nan |                           nan |                      nan |                         nan |
| ('sfl_oracle_learnability_level', 0)     |      2.4576e+08 |         30000 |                     242 |                       14399 |                         14399 |                      960 |                         120 |
| ('sfl_accel_learnability_n64', 0)        |      2.4576e+08 |         30000 |                     242 |                       14399 |                         14399 |                      960 |                         nan |
| ('accel_maxmc', 1)                       |      2.4576e+08 |         30000 |                     nan |                         nan |                           nan |                      nan |                         nan |
| ('accel_maxmc', 2)                       |      2.4576e+08 |         30000 |                     nan |                         nan |                           nan |                      nan |                         nan |
| ('sfl_oracle_learnability_level_bfs', 2) |      2.4576e+08 |         30000 |                     242 |                       14399 |                         14399 |                      960 |                         120 |
| ('sfl_oracle_learnability_level_bfs', 1) |      2.4576e+08 |         30000 |                     242 |                       14399 |                         14399 |                      960 |                         120 |
| ('sfl_accel_learnability', 0)            |      2.4576e+08 |         30000 |                     242 |                       13199 |                         13199 |                     3360 |                         nan |
| ('accel_maxmc', 0)                       |      2.4576e+08 |         30000 |                     nan |                         nan |                           nan |                      nan |                         nan |
| ('plr_maxmc', 0)                         |      2.4576e+08 |         30000 |                     nan |                         nan |                           nan |                      nan |                         nan |
| ('sfl_oracle_learnability_level_bfs', 0) |      2.4576e+08 |         30000 |                     242 |                       14399 |                         14399 |                      960 |                         120 |

## Throughput

| run_name                          |   time_delta |   steps_per_second |   hours_per_30k_updates |
|:----------------------------------|-------------:|-------------------:|------------------------:|
| accel_maxmc                       |         71.7 |            37593.6 |                     2.4 |
| dr                                |         74.4 |            27519.4 |                     2.5 |
| plr_maxmc                         |         64.2 |            31881.2 |                     2.1 |
| sfl_accel_learnability            |         26.6 |            76956.8 |                     0.9 |
| sfl_accel_learnability_n64        |         75.9 |            26999.2 |                     2.5 |
| sfl_oracle_learnability_level     |         74   |            27688   |                     2.5 |
| sfl_oracle_learnability_level_bfs |         76.8 |            26678.3 |                     2.6 |

## Oracle selection

Mean over all SFL phases of the measured learnability of the levels the oracle picked (`picked`) and of the uniformly drawn controls measured in the same rollouts (`control`); `gain` is the ratio of those means (1.0 = chance).

|                                          |   picked |   control |   gain |
|:-----------------------------------------|---------:|----------:|-------:|
| ('sfl_oracle_learnability_level', 0)     |   0.0214 |    0.0109 | 1.9602 |
| ('sfl_oracle_learnability_level_bfs', 0) |   0.0207 |    0.0105 | 1.9619 |
| ('sfl_oracle_learnability_level_bfs', 1) |   0.0194 |    0.0102 | 1.8938 |
| ('sfl_oracle_learnability_level_bfs', 2) |   0.0193 |    0.0106 | 1.8094 |

## Curriculum diagnostics

See `results/figs/curriculum.png`. Columns present: train/success_rate, train/learnability, level_sampler/mean_p, level/mean_num_blocks, sfl/topk_learnability vs sfl/population_learnability, oracle/selected_learnability vs oracle/control_learnability, oracle/buffer_mean_p vs level_sampler/mean_p.
