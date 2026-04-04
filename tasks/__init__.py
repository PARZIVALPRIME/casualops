from .base import BaseTask
from .task_easy import EasyTask
from .task_medium import MediumTask
from .task_hard import HardTask

TASK_REGISTRY = {
    "easy_smoking_gun": EasyTask,
    "medium_web_of_lies": MediumTask,
    "hard_shape_shifter": HardTask,
}

__all__ = ["BaseTask", "TASK_REGISTRY"]
