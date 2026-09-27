"""
Evaluation suite for UAV Multi-Agent Pipeline.
Includes dataset loaders (VisDrone, UAVDT), quantitative metrics computation,
and evaluation orchestration CLI.
"""
from .dataset_loader import SequenceDataset, GroundTruthBox, FrameItem
from .metrics import BenchmarkEvaluator, EvaluationReport

__all__ = [
    "SequenceDataset",
    "GroundTruthBox",
    "FrameItem",
    "BenchmarkEvaluator",
    "EvaluationReport",
]

