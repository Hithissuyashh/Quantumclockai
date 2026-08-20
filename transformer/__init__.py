"""
Transformer Package for QuantumClockAI
========================================
Attention-based sequence model for clock state prediction and
frequency error forecasting.
"""

from .model import ClockTransformer
from .dataset import SequenceDataset
from .trainer import TransformerTrainer

__all__ = ["ClockTransformer", "SequenceDataset", "TransformerTrainer"]
