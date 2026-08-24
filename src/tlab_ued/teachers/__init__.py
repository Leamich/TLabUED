"""Teacher registry.

Adding an idea: write `teachers/my_idea.py` with a `Teacher` subclass, register
it below, and it is selectable as `--teacher my_idea`. Nothing in `train.py`,
`student.py` or the notebook needs to change.
"""

from __future__ import annotations

from typing import Any, Dict, Type

from tlab_ued.teachers.accel import ACCELTeacher
from tlab_ued.teachers.base import Teacher, TrainContext, TrainState
from tlab_ued.teachers.dr import DRTeacher
from tlab_ued.teachers.plr import PLRTeacher
from tlab_ued.teachers.sfl_accel import SFLACCELTeacher
from tlab_ued.teachers.sfl_oracle import SFLOracleTeacher

TEACHERS: Dict[str, Type[Teacher]] = {
    "dr": DRTeacher,
    "plr": PLRTeacher,
    "accel": ACCELTeacher,
    "sfl_accel": SFLACCELTeacher,
    "sfl_oracle": SFLOracleTeacher,
}


def get_teacher_cls(config: Dict[str, Any]) -> Type[Teacher]:
    name = config["teacher"]
    if name not in TEACHERS:
        raise ValueError(f"Unknown teacher {name!r}. Registered: {sorted(TEACHERS)}")
    return TEACHERS[name]


__all__ = [
    "TEACHERS",
    "get_teacher_cls",
    "Teacher",
    "TrainContext",
    "TrainState",
    "DRTeacher",
    "PLRTeacher",
    "ACCELTeacher",
    "SFLACCELTeacher",
    "SFLOracleTeacher",
]
