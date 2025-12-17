# training/trainer.py
import numpy as np
import time
from typing import Dict, Any
from sklearn.model_selection import KFold

class ModelTrainer:
    """Classe para treinamento e avaliação de modelos"""
    
    def __init__(self):
        self.results = {}
        self.best_model_info = None
        self.logger = self._get_logger()
    
    def _get_logger(self):
        """Obtém logger de forma segura"""
        try:
            from utils.logger import get_logger
            return get_logger(__name__)
        except ImportError:
            import logging
            logging.basicConfig(level=logging.INFO)
            return logging.getLogger(__name__)
    
    def train_with_kfold(self, model, X, y, model_name, n_splits=5):
        """Treina e avalia modelo com K-Fold Cross Validation"""
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=config.SEED)
        fold_results = {
            'errors': [],
            'times': [],
            'predictions': [],
            'true_values': []
        }
        
        logger.info(f"Treinando {model_name} com {n_splits}-fold CV")
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            logger.info(f"  Fold {fold + 1}/{n_splits}")
            
            # Separar dados
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Treinar modelo
            start_time = time.time()
            
            try:
                model.fit(X_train, y_train)
                training_time = time.time() - start_time
                
                # Previsões
                y_pred = model.predict(X_val)
                
                # Calcular métrica
                error = haversine_distance_km(y_val, y_pred).mean()
                
                fold_results['errors'].append(error)
                fold_results['times'].append(training_time)
                fold_results['predictions'].append(y_pred)
                fold_results['true_values'].append(y_val)
                
                logger.info(f"    Erro: {error:.4f} km, Tempo: {training_time:.2f}s")
                
            except Exception as e:
                logger.error(f"    Erro no fold {fold + 1}: {e}")
                fold_results['errors'].append(np.inf)
                fold_results['times'].append(0)
        
        # Calcular estatísticas
        if fold_results['errors']:
            mean_error = np.mean(fold_results['errors'])
            std_error = np.std(fold_results['errors'])
            mean_time = np.mean(fold_results['times'])
            
            result = {
                'mean_error': mean_error,
                'std_error': std_error,
                'mean_time': mean_time,
                'fold_errors': fold_results['errors'],
                'fold_times': fold_results['times'],
                'model': model,
                'model_name': model_name
            }
            
            logger.info(f"Resultado {model_name}: {mean_error:.4f} ± {std_error:.4f} km")
            
            return result
        
        return None
    
    def train_all_models(self, X, y, models):
        """Treina todos os modelos"""
        
        logger.info(f"Iniciando treinamento de {len(models)} modelos")
        
        for model_name, model in models.items():
            try:
                result = self.train_with_kfold(
                    model, X, y, model_name, 
                    n_splits=config.KFOLD_SPLITS
                )
                
                if result:
                    self.results[model_name] = result
                    
                    # Atualizar melhor modelo
                    if (self.best_model_info is None or 
                        result['mean_error'] < self.best_model_info['mean_error']):
                        self.best_model_info = result
                        
            except Exception as e:
                logger.error(f"Erro ao treinar {model_name}: {e}")
        
        logger.info(f"Treinamento concluído. Melhor modelo: {self.best_model_info['model_name']}")
        return self.results
    
    def train_final_model(self, X, y, model_name=None, model=None):
        """Treina o modelo final com todos os dados"""
        
        if model_name is None and model is None:
            if self.best_model_info:
                model_name = self.best_model_info['model_name']
                model = self.best_model_info['model']
            else:
                raise ValueError("Nenhum modelo disponível para treino final")
        
        logger.info(f"Treinando modelo final: {model_name}")
        
        start_time = time.time()
        model.fit(X, y)
        training_time = time.time() - start_time
        
        logger.info(f"Modelo final treinado em {training_time:.2f}s")
        
        return {
            'model': model,
            'model_name': model_name,
            'training_time': training_time
        }