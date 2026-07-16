"""Core modules for MOGAT multi-omics classification."""

from .data_preprocessing import MultiOmicsDataProcessor
from .model_training import CrossValidator, ModelTrainer, MultiOmicsGATModel

__all__ = [
    "MultiOmicsDataProcessor",
    "MultiOmicsGATModel",
    "ModelTrainer",
    "CrossValidator",
]
