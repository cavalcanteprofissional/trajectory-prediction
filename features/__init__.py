"""
Módulo de engenharia de features, detecção de outliers e clusterização
"""
from .engineering import FeatureEngineer
from .outlier_detection import OutlierDetector
from .clustering import DataClusterer

__all__ = ['FeatureEngineer', 'OutlierDetector', 'DataClusterer']

