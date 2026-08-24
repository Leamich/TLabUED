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
    # BFS sanity check on the training distribution, printed before the run and
    # written to <run_dir>/level_diagnostics.json: how many generated levels are
    # solvable at all, and how hard the solvable ones are. 0 turns it off. It
    # draws from its own PRNG key, so it never shifts the training stream.
    "diagnose_levels": 1024,
    # how many rounds of the mutator to apply before diagnosing the children
    # (mutation teachers only)
    "diagnose_mutation_rounds": 1,
    # bypass the student-freeze guard (deliberate ablations only)
    "allow_student_changes": False,
    # === SFL (teacher `sfl_accel`; see teachers/sfl_accel.py) ===
    # How many attempts each level gets when it is scored. The 32 envs of an
    # update carry num_train_envs/k distinct levels, so this costs nothing extra
    # - it trades levels-per-update for a success rate that is not 0 or 1.
    "sfl_num_attempts": 4,
    # Updates between the start of one SFL evaluation phase and the next; 0
    # disables the phase and leaves learnability scoring on the ACCEL branches
    # (the ablation that separates "learnability" from "SFL").
    "sfl_period": 250,
    # Levels evaluated per phase, and how many of them are kept. 224 * 4 / 32 =
    # 28 updates per phase, 120 phases over a full run: 11.2% of the budget,
    # which is what ACCEL spends on its DR branch.
    "sfl_num_levels": 224,
    "sfl_topk": 32,
    # Weight of the newest observation in a replayed level's running success
    # rate. 0.25 forgets over ~10 replays, which is what keeps a buffer entry's
    # score tracking the policy instead of the policy it was measured against.
    "sfl_p_decay": 0.25,
    # === Learnability oracle (teacher `sfl_oracle`; see oracle.py) ===
    # What the oracle is allowed to look at: "level" is the raw map through a
    # convnet, "bfs" is the exact solver's summary (optimal steps, solvable,
    # reachable area, wall density) with no vision at all, "level_bfs" is both.
    # The three are an ablation ladder: what does predicting `p` actually need?
    "oracle_features": "level_bfs",
    "oracle_hidden": 64,
    "oracle_lr": 3e-4,
    # Adam steps per training update, and the minibatch each one draws from the
    # ring of recent observations. Small on purpose: the oracle has to chase a
    # moving policy, and it must not become a noticeable share of the wall clock.
    "oracle_train_steps": 2,
    "oracle_batch_size": 256,
    # How many (level, solved) observations the oracle remembers. 8192 is roughly
    # the last 300 updates - old enough to be plentiful, recent enough that the
    # policy that produced it is still approximately this one.
    "oracle_buffer_capacity": 8192,
    # Levels ranked per phase before any of them is played. This is the number
    # that measurement could never afford: 8192 rollout-scored levels would cost
    # 1024 updates, ranking them costs one forward pass.
    "oracle_num_proposals": 8192,
    # Of the `sfl_num_levels` the phase verifies, how many are drawn uniformly
    # rather than from the top of the ranking. They are the control group that
    # makes `oracle/selection_gain` a measurement rather than a hope - and the
    # only sample on which the oracle's ranking can be scored without range
    # restriction, since it picked the other 48 to all look alike. 8 was too few:
    # in the 3000-update probe whole control groups scored zero learnability,
    # which makes the gain ratio undefined for that phase.
    "oracle_control_levels": 16,
    # Children generated per parent on a mutation update; the one predicted most
    # learnable is the one played. 1 restores ACCEL's blind single edit.
    "oracle_mutation_proposals": 8,
    # Re-score the whole level buffer with the oracle at each phase, so an entry's
    # priority stops depending on how long ago it was last replayed.
    "oracle_rescore_buffer": True,
    # Play the oracle's shortlist before inserting it (the cascade). Turning this
    # off inserts on prediction alone and returns the phase's updates to the
    # student - the aggressive arm.
    "oracle_verify": True,
    # Updates before the oracle's ranking is used at all; until then selections
    # are uniform random, which is what an uninformative model should do.
    "oracle_warmup_updates": 1000,
}

DEFAULTS: Dict[str, Any] = {
    **RUN_DEFAULTS,
    **STUDENT_DEFAULTS,
    **TEACHER_KNOBS,
    **EXTRA_DEFAULTS,
}

def _oracle_presets() -> Dict[str, Dict[str, Any]]:
    """`sfl_oracle` and the four ablations that take it apart.

    The base arm is the cascade: 8192 proposals ranked by the oracle, 64 played,
    32 kept. Each ablation removes exactly one mechanism, so a difference in the
    final solve rate has one candidate explanation rather than five.
    """
    base = {
        "teacher": "sfl_oracle",
        "use_accel": True,
        "exploratory_grad_updates": False,
        "score_function": "learnability",
        "replay_prob": 1.0,
        # The verified shortlist: a seventh of what `sfl_accel` plays per phase.
        "sfl_num_levels": 64,
    }
    return {
        "sfl_oracle": base,
        # No verification: insert on prediction alone and give the phase's updates
        # back to the student.
        "sfl_oracle_noverify": {**base, "oracle_verify": False},
        # Feature ablations: is a convnet over the map needed, or is the solver's
        # summary of it enough - or vice versa?
        "sfl_oracle_level": {**base, "oracle_features": "level"},
        "sfl_oracle_bfs": {**base, "oracle_features": "bfs"},
        # ACCEL's blind mutation, with the oracle still driving the phase.
        "sfl_oracle_nomut": {**base, "oracle_mutation_proposals": 1},
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
    # Ours: ACCEL with SFL's learnability score (Rutherford et al., 2024) in
    # place of MaxMC. `replay_prob=1.0` is what keeps the budget matched to
    # ACCEL's - the SFL phase replaces the DR branch rather than adding to it, so
    # outside a phase every update is a replay or its mutation, and the count of
    # *gradient* updates lands within 0.1% of ACCEL's. See teachers/sfl_accel.py.
    "sfl_accel": {
        "teacher": "sfl_accel",
        "use_accel": True,
        "exploratory_grad_updates": False,
        "score_function": "learnability",
        "replay_prob": 1.0,
    },
    # The ablation: learnability everywhere, but no evaluation phase.
    "sfl_accel_nophase": {
        "teacher": "sfl_accel",
        "use_accel": True,
        "exploratory_grad_updates": False,
        "score_function": "learnability",
        "sfl_period": 0,
    },
    # SFL with the oracle's phase *cost* but none of its machinery: 64 measured
    # candidates instead of 224. The budget twin of `sfl_oracle`, and the control
    # that separates "the cascade selects better" from "the shorter phase leaves
    # more updates for the student".
    "sfl_accel_cheap": {
        "teacher": "sfl_accel",
        "use_accel": True,
        "exploratory_grad_updates": False,
        "score_function": "learnability",
        "replay_prob": 1.0,
        "sfl_num_levels": 64,
    },
    **_oracle_presets(),
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
    p.add_argument(
        "--diagnose_levels",
        type=int,
        default=DEFAULTS["diagnose_levels"],
        help="BFS-diagnose this many generated levels before training (0 = off)",
    )
    p.add_argument(
        "--diagnose_mutation_rounds",
        type=int,
        default=DEFAULTS["diagnose_mutation_rounds"],
        help="rounds of the mutator to apply before diagnosing the children",
    )
    _bool_flag(p, "allow_student_changes", DEFAULTS["allow_student_changes"])
    # === SFL (ours) ===
    sfl = p.add_argument_group("SFL teacher")
    sfl.add_argument(
        "--sfl_num_attempts",
        type=int,
        default=DEFAULTS["sfl_num_attempts"],
        help="attempts per level when scoring; num_train_envs must be divisible by it",
    )
    sfl.add_argument(
        "--sfl_period",
        type=int,
        default=DEFAULTS["sfl_period"],
        help="updates between SFL evaluation phases (0 = no phase)",
    )
    sfl.add_argument(
        "--sfl_num_levels",
        type=int,
        default=DEFAULTS["sfl_num_levels"],
        help="levels evaluated per phase (for sfl_oracle: the shortlist it verifies)",
    )
    sfl.add_argument("--sfl_topk", type=int, default=DEFAULTS["sfl_topk"])
    sfl.add_argument("--sfl_p_decay", type=float, default=DEFAULTS["sfl_p_decay"])

    # === Learnability oracle (ours) ===
    from tlab_ued.oracle import FEATURE_SETS

    orc = p.add_argument_group("Learnability oracle")
    orc.add_argument(
        "--oracle_features",
        type=str,
        default=DEFAULTS["oracle_features"],
        choices=sorted(FEATURE_SETS),
        help="what the oracle sees: the map, the BFS solution, or both",
    )
    orc.add_argument("--oracle_hidden", type=int, default=DEFAULTS["oracle_hidden"])
    orc.add_argument("--oracle_lr", type=float, default=DEFAULTS["oracle_lr"])
    orc.add_argument("--oracle_train_steps", type=int, default=DEFAULTS["oracle_train_steps"])
    orc.add_argument("--oracle_batch_size", type=int, default=DEFAULTS["oracle_batch_size"])
    orc.add_argument(
        "--oracle_buffer_capacity", type=int, default=DEFAULTS["oracle_buffer_capacity"]
    )
    orc.add_argument(
        "--oracle_num_proposals",
        type=int,
        default=DEFAULTS["oracle_num_proposals"],
        help="levels ranked per phase before any is played",
    )
    orc.add_argument(
        "--oracle_control_levels",
        type=int,
        default=DEFAULTS["oracle_control_levels"],
        help="uniformly drawn levels inside the verified shortlist (the control group)",
    )
    orc.add_argument(
        "--oracle_mutation_proposals",
        type=int,
        default=DEFAULTS["oracle_mutation_proposals"],
        help="children generated per parent; 1 is ACCEL's blind mutation",
    )
    _bool_flag(orc, "oracle_rescore_buffer", DEFAULTS["oracle_rescore_buffer"])
    _bool_flag(
        orc, "oracle_verify", DEFAULTS["oracle_verify"], "play the shortlist before inserting it"
    )
    orc.add_argument(
        "--oracle_warmup_updates", type=int, default=DEFAULTS["oracle_warmup_updates"]
    )

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
    """Run name from the teacher-side settings that actually distinguish runs.

    Ablations of the same teacher must not collide: two runs sharing a name share
    a directory, and the sweep runner would call the second one already finished.
    So anything a preset varies has to show up here.
    """
    name = str(config["teacher"])
    if config["teacher"] != "dr":
        name += f"_{str(config['score_function']).lower()}"
        if config.get("exploratory_grad_updates"):
            name += "_expl"
    if config["teacher"] in ("sfl_accel", "sfl_oracle") and not config.get("sfl_period"):
        name += "_nophase"
    elif config["teacher"] == "sfl_accel":
        if config.get("sfl_num_levels") != EXTRA_DEFAULTS["sfl_num_levels"]:
            name += f"_n{config['sfl_num_levels']}"
    elif config["teacher"] == "sfl_oracle":
        name += f"_{config.get('oracle_features', '')}"
        if not config.get("oracle_verify", True):
            name += "_noverify"
        if int(config.get("oracle_mutation_proposals", 1)) <= 1:
            name += "_nomut"
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
    # Sizing errors here would otherwise surface as a shape mismatch inside a
    # jitted branch, minutes into a run.
    if config.get("teacher") == "sfl_accel":
        from tlab_ued.teachers.sfl_accel import validate as validate_sfl

        validate_sfl(config)
    if config.get("teacher") == "sfl_oracle":
        from tlab_ued.teachers.sfl_accel import validate as validate_sfl
        from tlab_ued.teachers.sfl_oracle import validate as validate_oracle

        validate_sfl(config)
        validate_oracle(config)
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
