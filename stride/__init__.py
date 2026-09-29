"""STRIDE: Spatial-Temporal Reasoning In Driving Environments."""

__version__ = "0.1.0"

from .mcq import extract_answer_key, score_mcq_tasks
from .trajectory import parse_points, score_trajectory_tasks
from .aggregate import aggregate_scores, load_tasks

__all__ = [
    "__version__",
    "extract_answer_key",
    "score_mcq_tasks",
    "parse_points",
    "score_trajectory_tasks",
    "aggregate_scores",
    "load_tasks",
]
