import sys
from pathlib import Path
# Ensure repo root is in path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.model_factory import ModelFactory
from training.cross_validation import CrossValidator
import numpy as np

np.random.seed(42)
X = np.random.randn(200, 10)
y = np.random.randn(200, 2) + np.array([40.0, -74.0])

factory = ModelFactory(n_samples=200)
models = factory.create_all_models(priority_only=True, include_ensemble=True, n_features=X.shape[1])
print('Models keys:', list(models.keys()))
validator = CrossValidator(n_splits=3, random_state=42)
results = validator.validate_multiple_models(models, X, y, verbose=True)
print('Validation finished. Ensemble present in results:', 'EnsembleVoting' in results)
