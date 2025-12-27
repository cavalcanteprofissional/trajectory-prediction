#!/usr/bin/env python3
"""
Script para tunar GradientBoosting com dados limpos (com detecção de outliers)
"""
import sys
from pathlib import Path
import json
import optuna
import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

# Adicionar diretório raiz ao path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from data.loader import DataLoader
from features.engineering import FeatureEngineer
from features.outlier_detection import OutlierDetector
from features.cleaning import clean_train_test

# Setup logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_and_clean_data():
    """Carrega e limpa dados com detecção de outliers"""
    loader = DataLoader()
    train_df, test_df = loader.load_data()

    # Limpeza inicial
    train_df, test_df = clean_train_test(train_df, test_df)

    # Detecção de outliers (usando parâmetros atuais do main.py)
    outlier_detector = OutlierDetector(
        max_jump_distance_km=200.0,
        max_speed_kmh=500.0,
        contamination=0.02,
        max_outlier_percentage=0.10
    )

    outliers = outlier_detector.detect_all_outliers(
        train_df,
        use_geographic=True,
        use_trajectory=True,
        use_target=True,
        use_features=False
    )

    # Combinar outliers
    outliers_combined = outlier_detector.get_combined_outliers(outliers, method='any')

    # Aplicar proteção: remover apenas coordenadas geográficas inválidas se >2%
    outlier_percentage = outliers_combined.sum() / len(train_df) if len(train_df) > 0 else 0

    if outlier_percentage > 0.02:
        # Usar apenas outliers geográficos e de target (mais seguros)
        safe_outliers = pd.Series(False, index=train_df.index)
        if 'geographic' in outliers:
            safe_outliers = safe_outliers | outliers['geographic']
        if 'target' in outliers:
            safe_outliers = safe_outliers | outliers['target']
        outliers_combined = safe_outliers

    logger.info(f"Outliers detectados: {outliers_combined.sum()} de {len(train_df)}")

    # Remover outliers
    if outliers_combined.sum() > 0:
        train_df = outlier_detector.remove_outliers(train_df, outliers_combined, inplace=False)
        logger.info(f"Dados após remoção de outliers: {len(train_df)} amostras")

    # Engenharia de features
    feature_engineer = FeatureEngineer()
    train_features = feature_engineer.extract_all_features(train_df)
    test_features = feature_engineer.extract_all_features(test_df)

    # Adicionar targets se existirem
    if 'dest_lat' in train_df.columns and 'dest_lon' in train_df.columns:
        train_features['dest_lat'] = train_df['dest_lat'].values
        train_features['dest_lon'] = train_df['dest_lon'].values

    # Preparar features para treinamento
    prepared_data = feature_engineer.prepare_features_for_training(
        train_features, test_features, scaler_type='robust', use_local_target=True
    )

    X_train = prepared_data['X_train']
    y_train = prepared_data['y_train']

    return X_train, y_train

def objective(trial):
    """Função objetivo para Optuna"""
    # Definir espaço de busca
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'random_state': 42
    }

    # Criar modelo
    base_model = GradientBoostingRegressor(**params)
    model = MultiOutputRegressor(base_model)

    # Carregar dados (uma vez por estudo)
    if not hasattr(objective, 'X_train'):
        objective.X_train, objective.y_train = load_and_clean_data()

    # Calcular erro de validação cruzada
    scores = cross_val_score(
        model, objective.X_train, objective.y_train,
        cv=3, scoring='neg_mean_squared_error', n_jobs=-1
    )

    # Converter MSE para RMSE em km (aproximado)
    rmse = np.sqrt(-scores.mean())

    return rmse

def main():
    """Função principal"""
    logger.info("Iniciando tuning de GradientBoosting com dados limpos...")

    # Criar estudo Optuna
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20, timeout=3600)  # 20 trials ou 1 hora

    # Salvar melhores parâmetros
    results = {
        'best_value': study.best_value,
        'best_params': study.best_params,
        'n_trials': len(study.trials)
    }

    # Salvar em JSON
    reports_dir = ROOT_DIR / 'reports'
    reports_dir.mkdir(exist_ok=True)

    output_file = reports_dir / 'optuna_gb_clean_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Melhores parâmetros salvos em: {output_file}")
    logger.info(f"Melhor erro CV: {study.best_value:.2f} km")
    logger.info(f"Parâmetros: {study.best_params}")

if __name__ == '__main__':
    main()