"""
Módulo de treinamento e validação cruzada
"""
from .trainer import ModelTrainer
from .cross_validation import CrossValidator, cross_validate_model

__all__ = ['ModelTrainer', 'CrossValidator', 'cross_validate_model']

