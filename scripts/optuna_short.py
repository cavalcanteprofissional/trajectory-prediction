"""Short Optuna tuning run (20 trials) for LightGBM and XGBoost.

Prepares features using `FeatureEngineer` (RobustScaler) and runs
`ModelFactory.tune_with_optuna` for each target model. Saves results
to `reports/optuna_short_results.json`.
"""
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from data.loader import DataLoader
from features.engineering import FeatureEngineer
from models.model_factory import ModelFactory


def run():
    loader = DataLoader()
    train_df, _ = loader.load_data(use_sample_if_missing=True)

    fe = FeatureEngineer()
    X_df = fe.extract_all_features(train_df)

    # Ensure targets exist in the original train
    if 'dest_lat' in train_df.columns and 'dest_lon' in train_df.columns:
        X_df['dest_lat'] = train_df['dest_lat'].values
        X_df['dest_lon'] = train_df['dest_lon'].values

    # Prepare a dummy test_features DataFrame with same columns (required by prepare_features_for_training)
    # Use a single copy of the first row so scalers can `transform` without empty arrays
    test_df = X_df.iloc[[0]].copy() if len(X_df) > 0 else X_df.iloc[0:0].copy()
    prepared = fe.prepare_features_for_training(X_df, test_df, scaler_type='robust')
    X = prepared['X_train']
    y = prepared['y_train']
    groups = prepared.get('groups', None)

    factory = ModelFactory(n_samples=len(X))

    results = {}
    out_file = ROOT / 'reports' / 'optuna_short_results.json'
    out_file.parent.mkdir(exist_ok=True)

    for model_name in ['LightGBM', 'XGBoost']:
        logger.info(f"Starting Optuna tuning for {model_name} (20 trials)...")
        try:
            res = factory.tune_with_optuna(model_name, X, y, n_trials=20, cv_folds=3, groups=groups, y_unit='degrees')
            if res is None:
                logger.warning(f"No results for {model_name} (optuna missing or error)")
                results[model_name] = None
            else:
                # study object is not JSON serializable; save best params and value
                results[model_name] = {
                    'best_params': res.get('best_params'),
                    'best_value': float(res.get('best_value')) if res.get('best_value') is not None else None
                }
        except Exception as e:
            logger.exception(f"Optuna tuning failed for {model_name}: {e}")
            results[model_name] = {'error': str(e)}

    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Optuna short results saved: {out_file}")


if __name__ == '__main__':
    run()
