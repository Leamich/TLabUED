# Handoff: SFL-ORACLE experiment in flight

State as of 2026-08-24 ~06:40 UTC. Written so this can be picked up from another machine.

## Where things are

| | |
|---|---|
| Pod (A100 80GB, 252 vCPU) | `ssh root@38.128.232.45 -p 34475 -i ~/.ssh/id_ed25519` |
| Repo on the pod | `/workspace/TLabUED` |
| Outputs on the pod | `/workspace/tlab_ued/{runs,checkpoints,results}` |
| venv | `/workspace/venvs/jaxued/bin/python` |
| Logs | `/workspace/logs/{sweep2,preflight_cpu,restore,bootstrap}.log` |
| Old pod's data (recovered) | was network volume `3k9j2s39ir`, already pulled onto this pod |

The previous pod (RTX 4090) is gone. Its results - `dr/0`, `plr_maxmc/0`, `accel_maxmc/0`,
`accel_maxmc/1`, `sfl_accel_learnability/0`, all at 120/120 evals, plus their checkpoints - were
recovered from the old network volume over RunPod's S3 endpoint and now live on this pod. The
puller is `/workspace/restore/pull.py` (resumable) if any of it needs repeating.

## What is running

Nine runs, launched `2026-08-24T06:08Z`, ~2.4h to completion (so ~08:45Z):

```
sfl_oracle:0  sfl_oracle:1  sfl_oracle:2        the method, three seeds
sfl_accel_cheap:0                               its budget twin - the control that matters
sfl_oracle_noverify:0                           insert on prediction alone
sfl_oracle_nomut:0                              blind mutation
sfl_oracle_bfs:0  sfl_oracle_level:0            feature ablations
accel:2                                         third baseline seed + same-hardware timing
```

Check progress:

```bash
ssh root@38.128.232.45 -p 34475 'tail -3 /workspace/logs/sweep2.log; ls /workspace/tlab_ued/runs'
ssh root@38.128.232.45 -p 34475 'for d in /workspace/tlab_ued/runs/*/*/metrics.csv; do echo "$(( $(wc -l < $d) - 1 ))/120 $d"; done'
```

A run is finished when `runs/<name>/<seed>/DONE` exists. `python -m tlab_ued.sweep --jobs ... --status`
prints the same thing. Interrupted runs resume from their last checkpoint - re-running the sweep
command is safe and idempotent.

## What remains

1. Wait for the nine runs, then `python scripts/report.py --out_dir /workspace/tlab_ued`
   (writes `results/summary.md`, `results/figs/*.png`, `results/*.csv`).
2. Re-run parity **on the GPU** once it frees up - the in-flight CPU parity is a weaker check:
   `python -m tlab_ued.parity --presets dr plr accel --num_updates 500`.
3. Write the report into `README.md` **in Russian** (the task asks for it there).
4. Commit everything including weights, and push to `origin/master`. `.gitignore` excludes
   `checkpoints/`, `runs/`, `results/` - the previous commit force-added them, so use `git add -f`.
5. **Stop** the pod (do not delete it, and do not touch the network volumes). RunPod REST API:
   `curl -X POST -H "Authorization: Bearer $RUNPOD_API_TOKEN" https://rest.runpod.io/v1/pods/<id>/stop`.
   This pod is `q1l0lrmww4bqvw`. The token was pasted into a chat log and should be rotated.

## Things that cost time here, so you do not repeat them

- **CUDA MPS is worth 3x on this workload.** Nine trainers on one A100 time-slice badly: 8.5k
  steps/s each, 81k aggregate. Under MPS they run concurrently: 27k each, 243k aggregate. Start it
  with `nvidia-cuda-mps-control -d` (with `CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps`) *before*
  launching, and export the same variable in the launching shell so children inherit it. The tell
  that it is not working: `nvidia-smi` reporting 100% GPU utilisation at 1% memory utilisation.
- **SSH to this pod drops after ~10 minutes.** Anything long must be `setsid nohup ... &` with
  stdout redirected and `< /dev/null`, then polled with short connections. A non-detached run dies
  with the connection.
- **Parity and pytest must run with `CUDA_VISIBLE_DEVICES=""`.** `JAX_PLATFORMS=cpu` alone still
  let the parity process map 26 GB of device memory, which starves the trainers; and a GPU parity
  run alongside a full card fails with `gpusolverDnCreate ... cuSolver internal error`.
- **`pytest` is not in `requirements.txt`** - `pip install pytest` into the venv first.
- Files copied from Windows arrive with CRLF, which breaks shell scripts (`set: pipefail: invalid
  option name`). `find . -type f -name "*.sh" -exec sed -i 's/\r$//' {} +`.

## Where the experiment stands scientifically

The gate probe (3000 updates on the 4090, before this sweep) passed: selection gain 2.77 median
over nine post-warm-up phases, measured out of sample. Full detail, including the two measurement
bugs it caught, is in [`docs/experiments/sfl_oracle.md`](experiments/sfl_oracle.md#gate-probe-3000-updates-rtx-4090).

At ~2000 updates into the real runs the signal is holding: `oracle/selection_gain` between 1.6 and
8.0 across arms, and `train/success_rate` at 0.60-0.68 for the oracle arms against 0.726 for
`sfl_accel_cheap` - which is the direction predicted, since the free buffer re-scoring is supposed
to pull the frontier back toward `p = 0.5` from the 0.80 the SFL-ACCEL run settled at.

Read before writing the report: [`docs/experiments/sfl_oracle.md`](experiments/sfl_oracle.md) has
the arms, what each isolates, and the three ways the method can fail with the metric that
distinguishes them. [`docs/experiments/sfl_accel.md`](experiments/sfl_accel.md) is the previous
experiment and the source of the baseline numbers.
