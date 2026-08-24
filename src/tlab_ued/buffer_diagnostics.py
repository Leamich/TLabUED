"""What the curriculum actually became: BFS over the level buffer, checkpoint by checkpoint.

`level_diagnostics` measures the *operator* - what the generator produces and
what one mutation does to it. That is a property of the code, not of a run: no
student is involved, so it cannot answer "did the levels the student trains on
get harder over time". Only the run can answer that, because the buffer's
contents are selected by the score the student's own rollouts produced.

Nothing has to be re-run to find out. Every checkpoint carries the whole teacher
state, and for PLR/ACCEL/SFL that includes the level buffer:

    teacher_state.sampler.levels        the levels themselves
    teacher_state.sampler.scores        the PLR score each one was kept for
    teacher_state.sampler.size          how many slots are filled
    teacher_state.sampler.levels_extra  our bookkeeping: measured `p`, max return

So this module loads each checkpoint of a run, slices the buffer to its filled
part, and runs the same exact BFS `level_diagnostics` uses. The result is the
difficulty of the training distribution as a function of training progress, with
the real, student-driven selection in the loop.

Two populations are reported per checkpoint, because they answer different
questions:

    buffer  - everything in the buffer. What the curriculum has accumulated.
    top32   - the 32 highest-scored entries. Closer to what actually gets
              replayed, since PLR samples by score rank.

    python -m tlab_ued.buffer_diagnostics --run sfl_accel_learnability --seed 0
    python -m tlab_ued.buffer_diagnostics --run accel_maxmc --seed 0 --out results/buffer.json

Read alongside `level_diagnostics --ladder`, which brackets the same question
from the other side: what the mutation operator can produce with no selection at
all (`random`) and with perfect selection on difficulty (`hardest_of_m`). If the
buffer curve sits inside that bracket and rises, the ACCEL ladder is real in this
domain; if it is flat while the ladder's ceiling rises, the operator can build
harder levels but the score is not asking it to.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from tlab_ued.level_diagnostics import summarize

# Buffer fields we need to rebuild a `Level`. `width`/`height` ride along in the
# checkpoint and are restored as-is rather than assumed to be 13.
LEVEL_FIELDS: Tuple[str, ...] = ("wall_map", "goal_pos", "agent_pos", "agent_dir", "width", "height")


def checkpoint_steps(models_dir: str) -> List[int]:
    """The eval steps a run left behind, in order."""
    if not os.path.isdir(models_dir):
        raise FileNotFoundError(f"no checkpoints at {models_dir}")
    steps = [int(d) for d in os.listdir(models_dir) if d.isdigit()]
    return sorted(steps)


def _plain(x):
    """Orbax metadata containers -> plain dict/list/tuple, so `tree_map` accepts them."""
    if isinstance(x, Mapping):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_plain(v) for v in x)
    return x


def restore_tree(step_dir: str) -> Dict[str, Any]:
    """Restore one checkpoint as a plain nested dict of numpy arrays.

    Two things make this trickier than `restore(path)`:

    * these checkpoints were written on a GPU, so every leaf's metadata carries
      `SingleDeviceShardingMetadata(cuda:0)`. A target-less restore on a machine
      with no GPU asks jax to honour that sharding and fails. Restoring each leaf
      as `np.ndarray` sidesteps devices entirely - which is the point, since this
      analysis is CPU-only BFS and never touches the student.
    * the metadata tree is orbax's own mapping type, not a dict, so it has to be
      flattened into plain containers before `tree_map` will walk it.
    """
    import jax
    import orbax.checkpoint as ocp

    path = os.path.abspath(step_dir)
    item = os.path.join(path, "default")
    if os.path.isdir(item):
        path = item

    checkpointer = ocp.PyTreeCheckpointer()
    structure = _plain(checkpointer.metadata(path).item_metadata)
    restore_args = jax.tree_util.tree_map(
        lambda _: ocp.RestoreArgs(restore_type=np.ndarray), structure
    )
    return checkpointer.restore(path, restore_args=restore_args)


def _sampler(tree: Dict[str, Any]) -> Dict[str, Any]:
    teacher_state = tree.get("teacher_state") or {}
    sampler = teacher_state.get("sampler")
    if sampler is None:
        raise KeyError(
            "this checkpoint has no level buffer - `dr` keeps no sampler, "
            "so there is no curriculum to diagnose"
        )
    return sampler


def buffer_levels(sampler: Dict[str, Any], limit_to_size: bool = True):
    """The filled part of the buffer as a `Level` pytree.

    Slots past `size` hold whatever was there at initialisation; BFS over them
    would report the difficulty of placeholder levels as if the teacher had
    chosen them.
    """
    from jaxued.environments.maze import Level

    levels = sampler["levels"]
    fields = {k: np.asarray(levels[k]) for k in LEVEL_FIELDS if k in levels}
    size = int(np.asarray(sampler["size"])) if limit_to_size else len(fields["wall_map"])
    size = max(0, min(size, len(fields["wall_map"])))
    return Level(**{k: v[:size] for k, v in fields.items()}), size


def diagnose_checkpoint(
    step_dir: str, max_steps_in_episode: int = 250, top_k: int = 32
) -> Dict[str, Any]:
    """Difficulty of one checkpoint's buffer, whole and top-scored."""
    sampler = _sampler(restore_tree(step_dir))
    levels, size = buffer_levels(sampler)
    if size == 0:
        return {"size": 0}

    scores = np.asarray(sampler["scores"])[:size]
    out: Dict[str, Any] = {
        "size": size,
        "score_mean": float(scores.mean()),
        "buffer": summarize(levels, max_steps_in_episode),
    }

    extra = sampler.get("levels_extra") or {}
    if "p" in extra:
        out["p_mean"] = float(np.asarray(extra["p"])[:size].mean())

    if size > top_k:
        keep = np.argsort(-scores)[:top_k]
        import jax

        top = jax.tree_util.tree_map(lambda a: a[keep], levels)
        out[f"top{top_k}"] = summarize(top, max_steps_in_episode)
    return out


def diagnose_run(
    run: str,
    seed: int = 0,
    out_dir: str = ".",
    max_steps_in_episode: int = 250,
    top_k: int = 32,
    steps: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Every checkpoint of one run, oldest first."""
    models = os.path.join(out_dir, "checkpoints", run, str(seed), "models")
    wanted = checkpoint_steps(models) if steps is None else sorted(steps)
    per_step: Dict[str, Any] = {}
    for step in wanted:
        per_step[str(step)] = diagnose_checkpoint(
            os.path.join(models, str(step)), max_steps_in_episode, top_k
        )
    return {"meta": {"run": run, "seed": seed, "steps": wanted, "top_k": top_k}, "steps": per_step}


def format_run(report: Dict[str, Any]) -> str:
    """One row per checkpoint, meant to be read in a terminal."""
    meta = report["meta"]
    lines = [
        f"Level buffer over training  ({meta['run']}, seed {meta['seed']})",
        "",
        "  ckpt   size   median  p90   max   walls   mean_p   "
        f"top{meta['top_k']}_median  top{meta['top_k']}_max",
    ]
    for step in meta["steps"]:
        st = report["steps"][str(step)]
        if not st.get("size"):
            lines.append(f"  {step:>4}   empty")
            continue
        buf = st["buffer"]
        top = st.get(f"top{meta['top_k']}")
        p = st.get("p_mean")
        lines.append(
            f"  {step:>4}  {st['size']:>5}   {buf['steps_median']:>6.0f}  {buf['steps_p90']:>4.0f}  "
            f"{buf['steps_max']:>4}  {buf['mean_num_walls']:>5.1f}   "
            f"{('%.3f' % p) if p is not None else '    -'}   "
            f"{(('%.0f' % top['steps_median']) if top else '-'):>14}  "
            f"{((str(top['steps_max'])) if top else '-'):>11}"
        )
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    p = argparse.ArgumentParser(description="BFS the level buffer stored in a run's checkpoints")
    p.add_argument("--run", required=True, help="run name, i.e. the directory under checkpoints/")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", type=str, default=".", help="where checkpoints/ lives")
    p.add_argument("--top_k", type=int, default=32, help="size of the high-score subset reported")
    p.add_argument("--max_steps_in_episode", type=int, default=250)
    p.add_argument("--out", type=str, default="", help="also write the report to this JSON path")
    args = p.parse_args(argv)

    report = diagnose_run(
        args.run, args.seed, args.out_dir, args.max_steps_in_episode, args.top_k
    )
    print("")
    print(format_run(report), flush=True)
    print("")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
    return report


if __name__ == "__main__":
    main()
