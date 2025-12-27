"""
Short script to tune GradientBoostingRegressor using Optuna.
Saves best params to reports/optuna_gb_results.json
"""
import json
from pathlib import Path

import sys
from pathlib import Path

# garantir que o diretório raiz está no path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from data.loader import load_trajectory_data
from features.cleaning import clean_train_test
from features.engineering import FeatureEngineer
from models.model_factory import ModelFactory

REPORTS_DIR = Path(__file__).parent.parent / 'reports'
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def main(n_trials=12, cv_folds=3):
    print("Loading data...")
    train, test = load_trajectory_data()

    print("Applying conservative cleaning...")
    from features.cleaning import clean_train_test
    train, test = clean_train_test(train, test)

    fe = FeatureEngineer()
    print("Extracting features (train)...")
    train_feat = fe.extract_all_features(train)
    print("Extracting features (test)...")
    test_feat = fe.extract_all_features(test)

    # Garantir que os alvos estão presentes no DataFrame de features
    if 'dest_lat' in train.columns and 'dest_lon' in train.columns:
        train_feat['dest_lat'] = train['dest_lat'].values
        train_feat['dest_lon'] = train['dest_lon'].values

    prep = FeatureEngineer.prepare_features_for_training(train_feat, test_feat)
    X = prep['X_train']
    y = prep['y_train']

    factory = ModelFactory(n_samples=len(train))
    print(f"Running Optuna tuning for GradientBoosting ({n_trials} trials, {cv_folds} folds)...")
    result = factory.tune_with_optuna('GradientBoosting', X, y, n_trials=n_trials, cv_folds=cv_folds)

    out = {
        'best_value': result.get('best_value') if result else None,
        'best_params': result.get('best_params') if result else None
    }

    out_path = REPORTS_DIR / 'optuna_gb_results.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)

    print(f"Saved results to {out_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=12)
    parser.add_argument('--folds', type=int, default=3)
    args = parser.parse_args()
    main(n_trials=args.trials, cv_folds=args.folds)
