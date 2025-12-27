"""Quick experiment: 3-fold CV using FeatureEngineer new features + RobustScaler."""

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import pandas as pd
from data.loader import DataLoader
from features.engineering import FeatureEngineer
from models.model_factory import ModelFactory
from training.trainer import ModelTrainer


def run():
    ROOT = Path(__file__).parent.parent
    logger.info("Loading data...")
    loader = DataLoader()
    train_data, test_data = loader.load_data(use_sample_if_missing=True)

    fe = FeatureEngineer()
    logger.info("Extracting features...")
    train_features = fe.extract_all_features(train_data)
    test_features = fe.extract_all_features(test_data)

    # ensure dest present
    if 'dest_lat' in train_data.columns:
        train_features['dest_lat'] = train_data['dest_lat'].values
        train_features['dest_lon'] = train_data['dest_lon'].values
        # test may not have dests
        if isinstance(test_features, pd.DataFrame):
            test_features['dest_lat'] = test_data.get('dest_lat', pd.NA)
            test_features['dest_lon'] = test_data.get('dest_lon', pd.NA)

    logger.info("Preparing features (RobustScaler)...")
    prepared = fe.prepare_features_for_training(train_features, test_features, scaler_type='robust')

    X = prepared['X_train']
    y = prepared['y_train']
    groups = prepared.get('groups', None)

    logger.info(f"X shape: {X.shape}, y shape: {y.shape}, groups: {'present' if groups is not None else 'none'}")

    factory = ModelFactory(n_samples=len(X))
    models = factory.create_all_models(priority_only=True, include_ensemble=False, n_features=X.shape[1])

    trainer = ModelTrainer()
    logger.info("Running 3-fold CV on priority models...")
    results = trainer.train_all_models(X, y, models, cv_folds=3, groups=groups, y_unit='degrees')

    # Save quick report
    rpt = ROOT / 'reports' / 'quick_experiment_report.txt'
    rpt.parent.mkdir(exist_ok=True)
    with open(rpt, 'w', encoding='utf-8') as f:
        f.write('Quick experiment report\n')
        f.write('=======================\n')
        for name, res in results.items():
            f.write(f"{name}: {res['mean_error']:.4f} ± {res['std_error']:.4f} km\n")

    logger.info(f"Quick report saved: {rpt}")


if __name__ == '__main__':
    import pandas as pd
    run()
