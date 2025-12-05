"""
Detectors module - Módulos de detecção de deepfakes.

Contém implementações de detectores determinísticos e baseados em IA.
"""

from src.detectors.ai_model import DeepFakeClassifier
from src.detectors.deterministic import DeterministicDetector

__all__ = ["DeterministicDetector", "DeepFakeClassifier"]

