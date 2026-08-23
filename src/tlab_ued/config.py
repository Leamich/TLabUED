"""Configuration: upstream-compatible flags, presets, and the student-freeze guard.

Field names and defaults mirror the argparse blocks of JaxUED's
`examples/maze_plr.py` and `examples/maze_dr.py` exactly, so a config dict from
here is a drop-in for upstream `main()` (this is what the parity check in
`tlab_ued.parity` relies on). Everything we add lives in `EXTRA_DEFAULTS` and is
teacher-side or infrastructure only.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from typing import Any, Dict, List, Optional

EVAL_LEVELS: List[str] = [
    "SixteenRooms",
    "SixteenRooms2",
    "Labyrinth",
    "LabyrinthFlipped",
    "Labyrinth2",
    "StandardMaze",
    "StandardMaze2",
    "StandardMaze3",
]

# === Frozen by the assignment ===
# Upstream's "Training params" argparse group holds both student and teacher
# knobs. These are the student ones: architecture-adjacent settings plus the
# budget. Changing any of them turns a curriculum comparison into
# hyperparameter tuning, so assert_student_frozen refuses it.
STUDENT_DEFAULTS: Dict[str, Any] = {
    "lr": 1e-4,
    "max_grad_norm": 0.5,
    "num_updates": 30000,
    "num_steps": 256,
    "num_train_envs": 32,
    "num_minibatches": 1,
    "gamma": 0.995,
    "epoch_ppo": 5,
    "clip_eps": 0.2,
    "gae_lambda": 0.98,
    "entropy_coeff": 1e-3,
    "critic_coeff": 0.5,
    "agent_view_size": 5,
}

# Teacher-side knobs from that same upstream group - free to change.
TEACHER_KNOBS: Dict[str, Any] = {
    "score_function": "MaxMC",
    "exploratory_grad_updates": False,
    "level_buffer_capacity": 4000,
    "replay_prob": 0.8,
    "staleness_coeff": 0.3,
    "temperature": 0.3,
    "topk_k": 4,
    "minimum_fill_ratio": 0.5,
    "prioritization": "rank",
    "buffer_duplicate_check": True,
    "use_accel": False,
    "num_edits": 5,
    "n_walls": 25,
}

# Upstream flags outside the "Training params" group.
RUN_DEFAULTS: Dict[str, Any] = {
    "project": "tlab-ued",
    "run_name": None,
    "seed": 0,
    "mode": "train",
    "checkpoint_directory": None,
    "checkpoint_to_eval": -1,
    # The assignment mandates 17. Note the unit: EVAL steps, not updates.
    "checkpoint_save_interval": 17,
    "max_number_of_checkpoints": 60,
    "eval_freq": 250,
    "eval_num_attempts": 10,
    "eval_levels": EVAL_LEVELS,
    "num_env_steps": None,
}

# === Ours (no upstream counterpart) ===
EXTRA_DEFAULTS: Dict[str, Any] = {
    # which teacher plugin drives level selection
    "teacher": "accel",
    # which random level generator / mutator to use (see levels.py)
    "level_generator": "minigrid_walls",
    "level_mutator": "minimax",
    # "all" reproduces upstream logging exactly (needs moviepy: wandb encodes the
    # eval animations from raw numpy); "levels" keeps the rendered level images
    # but drops the videos; "none" drops all rendering. Media never touches the
    # training math or the PRNG stream - only wall time.
    "log_media": "levels",
    # where checkpoints/, runs/ and results/ are written (upstream: cwd)
    "out_dir": ".",
    # continue from the latest checkpoint of this run, if one exists
    "resume": False,
    # bypass the student-freeze guard (deliberate ablations only)
    "allow_student_changes": False,
}

DEFAULTS: Dict[str, Any] = {
    **RUN_DEFAULTS,
    **STUDENT_DEFAULTS,
    **TEACHER_KNOBS,
    **EXTRA_DEFAULTS,
}

# === Presets ===
# Each baseline preset is one upstream command expressed as overrides.
PRESETS: Dict[str, Dict[str, Any]] = {
    # python examples/maze_dr.py
    "dr": {"teacher": "dr"},
    # python examples/maze_plr.py --use_accel   (the assignment's ACCEL baseline)
    "accel": {"teacher": "accel", "use_accel": True, "exploratory_grad_updates": False},
    # PLR-perp: robust PLR, i.e. no gradient updates on exploratory rollouts.
    # This is the assignment's PLR baseline and upstream's default for maze_plr.py.
    "plr": {"teacher": "plr", "exploratory_grad_updates": False},
    # python examples/maze_plr.py --exploratory_grad_updates (non-robust PLR)
    "plr_exploratory": {"teacher": "plr", "exploratory_grad_updates": True},
    "accel_exploratory": {"teacher": "accel", "use_accel": True, "exploratory_grad_updates": True},
}

BASELINE_PRESETS = ("dr", "plr", "accel")

# Appended to the run name of any --smoke run.
SMOKE_SUFFIX = "_smoke"

# Applied on top of a preset by --smoke: exercises every code path (two eval
# steps, a checkpoint write) in a couple of minutes. The run name gets a
# "_smoke" suffix so a throwaway run can never collide with - or worse, be
# mistaken for finished work by the sweep runner - a real one.
SMOKE_OVERRIDES: Dict[str, Any] = {
    "num_updates": 500,
    "eval_freq": 250,
    "eval_num_attempts": 2,
    "checkpoint_save_interval": 1,
    "log_media": "levels",
}


class StudentConfigError(RuntimeError):
    """Raised when a run would silently change the frozen student."""


def assert_student_frozen(config: Dict[str, Any]) -> None:
    """Guard: the student config must match upstream defaults exactly.

    docs/TASK.md freezes the architecture, the PPO hyperparameters and the
    budget (30000 updates). Called at the top of every training run.
    """
    if config.get("allow_student_changes"):
        return
    deviations = {
        k: (config[k], v)
        for k, v in STUDENT_DEFAULTS.items()
        if k in config and config[k] != v
    }
    if deviations:
        lines = "\n".join(
            f"  {k}: {got!r} (frozen value: {want!r})" for k, (got, want) in deviations.items()
        )
        raise StudentConfigError(
            "The student config is frozen by the assignment; these fields deviate:\n"
            f"{lines}\n"
            "Only teacher-side settings may change. Pass --allow_student_changes for a "
            "deliberate ablation that will NOT be submitted as a comparison."
        )


def _bool_flag(parser, name: str, default: bool, help_text: str = "") -> None:
    parser.add_argument(
        f"--{name}", action=argparse.BooleanOptionalAction, default=default, help=help_text
    )


def build_parser() -> argparse.ArgumentParser:
    """argparse mirroring upstream flag-for-flag, plus our teacher-side extras."""
    from tlab_ued.scoring import SCORE_FUNCTIONS
    from tlab_ued.teachers import TEACHERS

    p = argparse.ArgumentParser(
        description="UED baselines and custom teachers on JaxUED mazes"
    )
    p.add_argument("--project", type=str, default=DEFAULTS["project"])
    p.add_argument("--run_name", type=str, default=None, help="default: derived from the teacher")
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    p.add_argument(
        "--preset", type=str, default=None, choices=sorted(PRESETS), help="baseline shorthand"
    )
    p.add_argument("--smoke", action="store_true", help="short run to validate plumbing and timing")
    # === Train vs Eval ===
    p.add_argument("--mode", type=str, default="train", choices=["train", "eval"])
    p.add_argument("--checkpoint_directory", type=str, default=None)
    p.add_argument("--checkpoint_to_eval", type=int, default=-1)
    # === CHECKPOINTING ===
    p.add_argument(
        "--checkpoint_save_interval",
        type=int,
        default=DEFAULTS["checkpoint_save_interval"],
        help="in EVAL steps, not updates (one eval step = eval_freq updates)",
    )
    p.add_argument(
        "--max_number_of_checkpoints", type=int, default=DEFAULTS["max_number_of_checkpoints"]
    )
    # === EVAL ===
    p.add_argument("--eval_freq", type=int, default=DEFAULTS["eval_freq"])
    p.add_argument("--eval_num_attempts", type=int, default=DEFAULTS["eval_num_attempts"])
    p.add_argument("--eval_levels", nargs="+", default=list(EVAL_LEVELS))
    # === INFRA (ours) ===
    p.add_argument("--out_dir", type=str, default=DEFAULTS["out_dir"])
    p.add_argument(
        "--log_media", type=str, default=DEFAULTS["log_media"], choices=["all", "levels", "none"]
    )
    _bool_flag(p, "resume", DEFAULTS["resume"], "continue from the latest checkpoint of this run")
    _bool_flag(p, "allow_student_changes", DEFAULTS["allow_student_changes"])

    group = p.add_argument_group("Training params")
    # === PPO (frozen) ===
    group.add_argument("--lr", type=float, default=STUDENT_DEFAULTS["lr"])
    group.add_argument("--max_grad_norm", type=float, default=STUDENT_DEFAULTS["max_grad_norm"])
    mut = group.add_mutually_exclusive_group()
    mut.add_argument("--num_updates", type=int, default=STUDENT_DEFAULTS["num_updates"])
    mut.add_argument("--num_env_steps", type=int, default=None)
    group.add_argument("--num_steps", type=int, default=STUDENT_DEFAULTS["num_steps"])
    group.add_argument("--num_train_envs", type=int, default=STUDENT_DEFAULTS["num_train_envs"])
    group.add_argument("--num_minibatches", type=int, default=STUDENT_DEFAULTS["num_minibatches"])
    group.add_argument("--gamma", type=float, default=STUDENT_DEFAULTS["gamma"])
    group.add_argument("--epoch_ppo", type=int, default=STUDENT_DEFAULTS["epoch_ppo"])
    group.add_argument("--clip_eps", type=float, default=STUDENT_DEFAULTS["clip_eps"])
    group.add_argument("--gae_lambda", type=float, default=STUDENT_DEFAULTS["gae_lambda"])
    group.add_argument("--entropy_coeff", type=float, default=STUDENT_DEFAULTS["entropy_coeff"])
    group.add_argument("--critic_coeff", type=float, default=STUDENT_DEFAULTS["critic_coeff"])
    group.add_argument("--agent_view_size", type=int, default=STUDENT_DEFAULTS["agent_view_size"])
    # === TEACHER ===
    group.add_argument("--teacher", type=str, default=DEFAULTS["teacher"], choices=sorted(TEACHERS))
    group.add_argument(
        "--score_function",
        type=str,
        default=TEACHER_KNOBS["score_function"],
        choices=sorted(SCORE_FUNCTIONS),
    )
    group.add_argument("--level_generator", type=str, default=DEFAULTS["level_generator"])
    group.add_argument("--level_mutator", type=str, default=DEFAULTS["level_mutator"])
    _bool_flag(group, "exploratory_grad_updates", TEACHER_KNOBS["exploratory_grad_updates"])
    group.add_argument(
        "--level_buffer_capacity", type=int, default=TEACHER_KNOBS["level_buffer_capacity"]
    )
    group.add_argument("--replay_prob", type=float, default=TEACHER_KNOBS["replay_prob"])
    group.add_argument("--staleness_coeff", type=float, default=TEACHER_KNOBS["staleness_coeff"])
    group.add_argument("--temperature", type=float, default=TEACHER_KNOBS["temperature"])
    group.add_argument("--topk_k", type=int, default=TEACHER_KNOBS["topk_k"])
    group.add_argument(
        "--minimum_fill_ratio", type=float, default=TEACHER_KNOBS["minimum_fill_ratio"]
    )
    group.add_argument(
        "--prioritization",
        type=str,
        default=TEACHER_KNOBS["prioritization"],
        choices=["rank", "topk"],
    )
    _bool_flag(group, "buffer_duplicate_check", TEACHER_KNOBS["buffer_duplicate_check"])
    _bool_flag(group, "use_accel", TEACHER_KNOBS["use_accel"])
    group.add_argument("--num_edits", type=int, default=TEACHER_KNOBS["num_edits"])
    group.add_argument("--n_walls", type=int, default=TEACHER_KNOBS["n_walls"])
    return p


def default_run_name(config: Dict[str, Any]) -> str:
    """Run name from the teacher-side settings that actually distinguish runs."""
    name = str(config["teacher"])
    if config["teacher"] != "dr":
        name += f"_{str(config['score_function']).lower()}"
        if config.get("exploratory_grad_updates"):
            name += "_expl"
    return name


def finalize(config: Dict[str, Any]) -> Dict[str, Any]:
    """Fill derived fields. Mirrors upstream's __main__ post-processing."""
    config = copy.deepcopy(config)
    if config.get("num_env_steps"):
        config["num_updates"] = config["num_env_steps"] // (
            config["num_train_envs"] * config["num_steps"]
        )
    if not config.get("run_name"):
        config["run_name"] = default_run_name(config)
    config["group_name"] = "_".join(
        f"{k}={config[k]}" for k in sorted(TEACHER_KNOBS) if k in config
    )
    config["out_dir"] = os.path.abspath(os.path.expanduser(str(config["out_dir"])))
    if config["num_updates"] % config["eval_freq"] != 0:
        raise ValueError(
            f"num_updates ({config['num_updates']}) must be a multiple of eval_freq "
            f"({config['eval_freq']}): the loop runs num_updates // eval_freq times."
        )
    return config


def _explicitly_passed(parser: argparse.ArgumentParser, argv: Optional[List[str]]) -> set:
    """Which dests appeared on the command line, so presets never override them."""
    tokens = list(sys.argv[1:] if argv is None else argv)
    dests = set()
    for action in parser._actions:
        for opt in action.option_strings:
            if any(t == opt or t.startswith(opt + "=") for t in tokens):
                dests.add(action.dest)
    return dests


def from_args(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    """Parse argv into a finalized config dict, applying --preset and --smoke."""
    parser = build_parser()
    args = vars(parser.parse_args(argv))
    explicit = _explicitly_passed(parser, argv)
    preset = args.pop("preset", None)
    smoke = args.pop("smoke", False)
    if preset:
        for k, v in PRESETS[preset].items():
            if k not in explicit:  # explicit flags win over the preset
                args[k] = v
    if smoke:
        for k, v in SMOKE_OVERRIDES.items():
            if k not in explicit:
                args[k] = v
        # a smoke run shortens num_updates on purpose
        args["allow_student_changes"] = True
        if "run_name" not in explicit:
            args["run_name"] = default_run_name(args) + SMOKE_SUFFIX
    return finalize(args)


def make_config(
    preset: Optional[str] = None, smoke: bool = False, **overrides: Any
) -> Dict[str, Any]:
    """Programmatic equivalent of from_args - used by notebooks and sweeps."""
    config = copy.deepcopy(DEFAULTS)
    config["eval_levels"] = list(EVAL_LEVELS)
    if preset:
        config.update(PRESETS[preset])
    if smoke:
        config.update(SMOKE_OVERRIDES)
        config["allow_student_changes"] = True
        config.setdefault("run_name", None)
    config.update(overrides)
    if smoke and not config.get("run_name"):
        config["run_name"] = default_run_name(config) + SMOKE_SUFFIX
    return finalize(config)


def to_argv(config: Dict[str, Any]) -> List[str]:
    """Render a config dict back into CLI flags (used by the sweep runner)."""
    argv: List[str] = []
    for key, value in config.items():
        if key in ("group_name", "preset", "smoke") or value is None:
            continue
        if isinstance(value, bool):
            argv.append(f"--{key}" if value else f"--no-{key}")
        elif isinstance(value, (list, tuple)):
            argv.append(f"--{key}")
            argv.extend(str(v) for v in value)
        else:
            argv.extend([f"--{key}", str(value)])
    return argv


def upstream_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Strip our extras, leaving a dict upstream main() accepts unchanged.

    Used by the parity check to run the same settings through
    third_party/jaxued/examples/maze_plr.py.
    """
    ours = set(EXTRA_DEFAULTS) | {"preset", "smoke", "level_generator", "level_mutator"}
    return {k: v for k, v in config.items() if k not in ours}
