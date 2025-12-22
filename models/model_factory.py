# models/model_factory.py
import sys
from pathlib import Path

# Adicionar diretório raiz ao path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from sklearn.ensemble import (
    RandomForestRegressor, 
    GradientBoostingRegressor, 
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    AdaBoostRegressor
)
from sklearn.linear_model import (
    LinearRegression, 
    Ridge, 
    Lasso, 
    ElasticNet,
    BayesianRidge
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import BaggingRegressor
import numpy as np

class ModelFactory:
    """Fábrica para criação de modelos aprimorada para predição de trajetórias"""
    
    # Modelos para teste
    DEFAULT_MODELS = [
        'RandomForest',
        'XGBoost',
        'LightGBM',
        'GradientBoosting',
        'HistGradientBoosting',
        'ExtraTrees',
        'CatBoost',
        'Ridge',
        'Lasso',
        'ElasticNet',
        'BayesianRidge',
        'KNN',
        'SVR',
        'MLP',
        'AdaBoost'
    ]
    
    # Adicionar Bagged Gradient Boosting como opção
    DEFAULT_MODELS = DEFAULT_MODELS + ['BaggedGB']
    
    # Modelos prioritários (melhores para coordenadas)
    PRIORITY_MODELS = [
        'RandomForest',
        'XGBoost', 
        'LightGBM',
        'GradientBoosting',
        'BaggedGB',
        'HistGradientBoosting'
    ]
    
    DEFAULT_SEED = 42
    
    def __init__(self, n_samples=None, lgbm_force_col_wise: bool = True):
        self.logger = self._get_logger()
        self.n_samples = n_samples
        # Recomenda-se forçar col-wise para reduzir warnings em alguns datasets
        self.lgbm_force_col_wise = lgbm_force_col_wise
    
    def _get_logger(self):
        """Obtém logger"""
        try:
            from utils.logger import get_logger
            return get_logger(__name__)
        except ImportError:
            import logging
            logging.basicConfig(level=logging.INFO)
            return logging.getLogger(__name__)
    
    def _get_model_params(self, model_name, n_features=None):
        """Retorna parâmetros otimizados para cada modelo"""
        
        # Ajustar n_estimators baseado no tamanho dos dados
        if self.n_samples:
            if self.n_samples < 1000:
                n_estimators = 50
            elif self.n_samples < 5000:
                n_estimators = 100
            else:
                n_estimators = 200
        else:
            n_estimators = 100
        
        # Parâmetros específicos para cada modelo
        model_params = {}
        
        # Random Forest - Otimizado para reduzir erro
        model_params['RandomForest'] = {
            'n_estimators': max(n_estimators, 200),  # Mais árvores para melhor performance
            'max_depth': 20,  # Profundidade limitada para evitar overfitting
            'min_samples_split': 10,  # Aumentado para reduzir overfitting
            'min_samples_leaf': 4,  # Aumentado para reduzir overfitting
            'max_features': 'sqrt' if n_features and n_features > 10 else None,
            'bootstrap': True,
            'oob_score': True,
            'random_state': self.DEFAULT_SEED,
            'n_jobs': -1
        }
        
        # XGBoost - Otimizado para reduzir erro
        model_params['XGBoost'] = {
            'n_estimators': max(n_estimators, 200),
            'max_depth': 7,  # Aumentado para capturar mais complexidade
            'learning_rate': 0.05,  # Reduzido para melhor generalização
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'reg_alpha': 0.5,  # Aumentado para regularização
            'reg_lambda': 1.5,  # Aumentado para regularização
            'gamma': 0.1,  # Adicionado para regularização
            'min_child_weight': 3,  # Adicionado para regularização
            'random_state': self.DEFAULT_SEED,
            'n_jobs': -1
        }
        
        # LightGBM - Otimizado para reduzir erro
        # LightGBM - Otimizado para reduzir erro e evitar warnings
        model_params['LightGBM'] = {
            'n_estimators': max(n_estimators, 200),
            'max_depth': 8,
            'learning_rate': 0.05,
            # num_leaves controlado para evitar splits sem ganho
            'num_leaves': 31,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'reg_alpha': 0.5,
            'reg_lambda': 0.5,
            # Aumentar min_child_samples reduz overfitting e evita mensagens de ganho -inf
            'min_child_samples': 100,
            # Forçar col-wise para estabilidade em determinados formatos de dados
            'force_col_wise': True if getattr(self, 'lgbm_force_col_wise', True) else False,
            # reduzir mensagens verbosas internas
            'verbosity': -1,
            'random_state': self.DEFAULT_SEED,
            'n_jobs': -1
        }
        
        # Gradient Boosting (scikit-learn) - Otimizado
        # Use Optuna-tuned defaults from prior tuning run (improves CV)
        model_params['GradientBoosting'] = {
            'n_estimators': 461,
            'learning_rate': 0.07347845538730073,
            'max_depth': 8,
            'min_samples_split': 7,
            'min_samples_leaf': 3,
            'subsample': 0.5138539907283792,
            'max_features': None,
            'random_state': self.DEFAULT_SEED
            # Nota: GradientBoostingRegressor não tem parâmetro 'bootstrap'
        }
        
        # Histogram Gradient Boosting
        model_params['HistGradientBoosting'] = {
            'max_iter': n_estimators,
            'learning_rate': 0.1,
            'max_depth': None,
            'min_samples_leaf': 20,
            'l2_regularization': 0.0,
            'max_bins': 255,
            'random_state': self.DEFAULT_SEED
        }
        
        # Extra Trees
        model_params['ExtraTrees'] = {
            'n_estimators': n_estimators,
            'max_depth': None,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'bootstrap': False,
            'random_state': self.DEFAULT_SEED,
            'n_jobs': -1
        }
        
        # CatBoost
        model_params['CatBoost'] = {
            'iterations': n_estimators,
            'depth': 6,
            'learning_rate': 0.1,
            'l2_leaf_reg': 3.0,
            'border_count': 254,
            'random_strength': 1.0,
            'bagging_temperature': 1.0,
            'od_type': 'Iter',
            'od_wait': 50,
            'random_state': self.DEFAULT_SEED,
            'verbose': 0
        }
        
        # Ridge Regression
        model_params['Ridge'] = {
            'alpha': 1.0,
            'random_state': self.DEFAULT_SEED
        }
        
        # Lasso
        model_params['Lasso'] = {
            'alpha': 1.0,
            'random_state': self.DEFAULT_SEED,
            'max_iter': 1000
        }
        
        # ElasticNet
        model_params['ElasticNet'] = {
            'alpha': 1.0,
            'l1_ratio': 0.5,
            'random_state': self.DEFAULT_SEED,
            'max_iter': 1000
        }
        
        # Bayesian Ridge
        model_params['BayesianRidge'] = {
            'n_iter': 300,
            'tol': 0.001,
            'alpha_1': 1e-06,
            'alpha_2': 1e-06,
            'lambda_1': 1e-06,
            'lambda_2': 1e-06,
            'compute_score': False
        }
        
        # K-Nearest Neighbors
        model_params['KNN'] = {
            'n_neighbors': 5,
            'weights': 'uniform',
            'algorithm': 'auto',
            'leaf_size': 30,
            'p': 2,
            'metric': 'minkowski'
        }
        
        # Support Vector Regression
        model_params['SVR'] = {
            'kernel': 'rbf',
            'C': 1.0,
            'epsilon': 0.1,
            'gamma': 'scale',
            'max_iter': 1000
        }
        
        # Multi-layer Perceptron
        model_params['MLP'] = {
            'hidden_layer_sizes': (100, 50),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.0001,
            'batch_size': 'auto',
            'learning_rate': 'constant',
            'learning_rate_init': 0.001,
            'max_iter': 200,
            'shuffle': True,
            'random_state': self.DEFAULT_SEED,
            'tol': 0.0001,
            'verbose': False,
            'warm_start': False,
            'momentum': 0.9,
            'nesterovs_momentum': True,
            'early_stopping': False,
            'validation_fraction': 0.1,
            'beta_1': 0.9,
            'beta_2': 0.999,
            'epsilon': 1e-08
        }
        
        # AdaBoost
        model_params['AdaBoost'] = {
            'n_estimators': n_estimators,
            'learning_rate': 1.0,
            'random_state': self.DEFAULT_SEED,
            'loss': 'linear'
        }
        
        return model_params.get(model_name, {})
    
    def create_model(self, model_name, params=None, n_features=None):
        """Cria um modelo baseado no nome com parâmetros otimizados"""
        
        # Obter parâmetros padrão otimizados
        default_params = self._get_model_params(model_name, n_features)
        
        # Sobrescrever com parâmetros fornecidos pelo usuário
        if params:
            default_params.update(params)
        
        # Mapeamento de modelos - IMPORTANTE: Cada modelo tem seus próprios parâmetros
        try:
            if model_name == 'RandomForest':
                model = RandomForestRegressor(**default_params)
            elif model_name == 'XGBoost':
                model = XGBRegressor(**default_params)
            elif model_name == 'LightGBM':
                model = LGBMRegressor(**default_params)
            elif model_name == 'GradientBoosting':
                model = GradientBoostingRegressor(**default_params)
            elif model_name == 'BaggedGB':
                # Criar base estimator com parâmetros otimizados
                base_params = default_params.copy()
                # remover parâmetros irrlevantes
                try:
                    base_est = GradientBoostingRegressor(**base_params)
                except Exception:
                    base_est = GradientBoostingRegressor()
                # BaggingRegressor API mudou entre versões (base_estimator -> estimator)
                try:
                    bag = BaggingRegressor(estimator=base_est, n_estimators=5, n_jobs=-1, random_state=self.DEFAULT_SEED)
                except TypeError:
                    bag = BaggingRegressor(base_estimator=base_est, n_estimators=5, n_jobs=-1, random_state=self.DEFAULT_SEED)
                model = bag
            elif model_name == 'HistGradientBoosting':
                model = HistGradientBoostingRegressor(**default_params)
            elif model_name == 'ExtraTrees':
                model = ExtraTreesRegressor(**default_params)
            elif model_name == 'CatBoost':
                model = CatBoostRegressor(**default_params)
            elif model_name == 'Ridge':
                model = Ridge(**default_params)
            elif model_name == 'Lasso':
                model = Lasso(**default_params)
            elif model_name == 'ElasticNet':
                model = ElasticNet(**default_params)
            elif model_name == 'BayesianRidge':
                model = BayesianRidge(**default_params)
            elif model_name == 'KNN':
                model = KNeighborsRegressor(**default_params)
            elif model_name == 'SVR':
                model = SVR(**default_params)
            elif model_name == 'MLP':
                model = MLPRegressor(**default_params)
            elif model_name == 'AdaBoost':
                model = AdaBoostRegressor(**default_params)
            else:
                raise ValueError(f"Modelo desconhecido: {model_name}. "
                               f"Modelos disponíveis: {self.DEFAULT_MODELS}")
        except TypeError as e:
            # Remover parâmetros que não existem para o modelo específico
            self.logger.warning(f"Ajustando parâmetros para {model_name}: {e}")
            # Tentar criar modelo sem parâmetros inválidos
            model = self._create_model_safe(model_name, default_params)
        
        # Criar modelo multi-output para prever latitude e longitude
        # Apenas RandomForest, XGBoost, ExtraTrees e CatBoost suportam multi-output nativamente
        # LightGBM, GradientBoosting e HistGradientBoosting precisam de wrapper
        if model_name in ['RandomForest', 'XGBoost', 'ExtraTrees', 'CatBoost']:
            # Esses modelos já suportam multi-output nativamente
            pass
        else:
            # Usar wrapper MultiOutputRegressor para outros modelos
            model = MultiOutputRegressor(model)
        
        # Adicionar nome do modelo como atributo
        model.model_name = model_name
        
        return model
    
    def _create_model_safe(self, model_name, params):
        """Cria modelo removendo parâmetros inválidos"""
        
        # Mapear parâmetros válidos para cada modelo
        valid_params_map = {
            'RandomForest': ['n_estimators', 'max_depth', 'min_samples_split', 
                           'min_samples_leaf', 'max_features', 'bootstrap', 
                           'oob_score', 'random_state', 'n_jobs'],
            'XGBoost': ['n_estimators', 'max_depth', 'learning_rate', 'subsample',
                       'colsample_bytree', 'reg_alpha', 'reg_lambda', 'gamma',
                       'random_state', 'n_jobs'],
            'LightGBM': ['n_estimators', 'max_depth', 'learning_rate', 'num_leaves',
                        'subsample', 'colsample_bytree', 'reg_alpha', 'reg_lambda',
                        'min_child_samples', 'force_col_wise', 'random_state', 'n_jobs'],
            'GradientBoosting': ['n_estimators', 'learning_rate', 'max_depth',
                                'min_samples_split', 'min_samples_leaf', 'subsample',
                                'max_features', 'random_state'],
            'HistGradientBoosting': ['max_iter', 'learning_rate', 'max_depth',
                                    'min_samples_leaf', 'l2_regularization',
                                    'max_bins', 'random_state'],
            'ExtraTrees': ['n_estimators', 'max_depth', 'min_samples_split',
                          'min_samples_leaf', 'max_features', 'bootstrap',
                          'random_state', 'n_jobs'],
            'CatBoost': ['iterations', 'depth', 'learning_rate', 'l2_leaf_reg',
                        'border_count', 'random_strength', 'bagging_temperature',
                        'od_type', 'od_wait', 'random_state', 'verbose'],
            'Ridge': ['alpha', 'random_state'],
            'Lasso': ['alpha', 'random_state', 'max_iter'],
            'ElasticNet': ['alpha', 'l1_ratio', 'random_state', 'max_iter'],
            'BayesianRidge': ['n_iter', 'tol', 'alpha_1', 'alpha_2', 'lambda_1',
                             'lambda_2', 'compute_score'],
            'KNN': ['n_neighbors', 'weights', 'algorithm', 'leaf_size', 'p', 'metric'],
            'SVR': ['kernel', 'C', 'epsilon', 'gamma', 'max_iter'],
            'MLP': ['hidden_layer_sizes', 'activation', 'solver', 'alpha', 'batch_size',
                   'learning_rate', 'learning_rate_init', 'max_iter', 'shuffle',
                   'random_state', 'tol', 'verbose', 'warm_start', 'momentum',
                   'nesterovs_momentum', 'early_stopping', 'validation_fraction',
                   'beta_1', 'beta_2', 'epsilon'],
            'AdaBoost': ['n_estimators', 'learning_rate', 'random_state', 'loss']
        }
        
        # Filtrar apenas parâmetros válidos
        valid_params = valid_params_map.get(model_name, [])
        filtered_params = {k: v for k, v in params.items() if k in valid_params}
        
        # Criar modelo com parâmetros filtrados
        if model_name == 'RandomForest':
            return RandomForestRegressor(**filtered_params)
        elif model_name == 'XGBoost':
            return XGBRegressor(**filtered_params)
        elif model_name == 'LightGBM':
            return LGBMRegressor(**filtered_params)
        elif model_name == 'GradientBoosting':
            return GradientBoostingRegressor(**filtered_params)
        elif model_name == 'HistGradientBoosting':
            return HistGradientBoostingRegressor(**filtered_params)
        elif model_name == 'ExtraTrees':
            return ExtraTreesRegressor(**filtered_params)
        elif model_name == 'CatBoost':
            return CatBoostRegressor(**filtered_params)
        elif model_name == 'Ridge':
            return Ridge(**filtered_params)
        elif model_name == 'Lasso':
            return Lasso(**filtered_params)
        elif model_name == 'ElasticNet':
            return ElasticNet(**filtered_params)
        elif model_name == 'BayesianRidge':
            return BayesianRidge(**filtered_params)
        elif model_name == 'KNN':
            return KNeighborsRegressor(**filtered_params)
        elif model_name == 'SVR':
            return SVR(**filtered_params)
        elif model_name == 'MLP':
            return MLPRegressor(**filtered_params)
        elif model_name == 'AdaBoost':
            return AdaBoostRegressor(**filtered_params)
        elif model_name == 'BaggedGB':
            # Filtrar parâmetros válidos para GradientBoosting
            gb_valid = valid_params_map.get('GradientBoosting', [])
            gb_params = {k: v for k, v in params.items() if k in gb_valid}
            try:
                base = GradientBoostingRegressor(**gb_params) if gb_params else GradientBoostingRegressor()
            except Exception:
                base = GradientBoostingRegressor()
            try:
                return BaggingRegressor(estimator=base, n_estimators=5, n_jobs=-1, random_state=self.DEFAULT_SEED)
            except TypeError:
                return BaggingRegressor(base_estimator=base, n_estimators=5, n_jobs=-1, random_state=self.DEFAULT_SEED)
        else:
            raise ValueError(f"Modelo desconhecido: {model_name}")
    
    def create_ensemble_model(self, base_models=None, voting='soft'):
        """Cria um modelo ensemble"""
        try:
            from sklearn.ensemble import VotingRegressor
            from sklearn.multioutput import MultiOutputRegressor
            
            if base_models is None:
                # Modelos base para ensemble
                base_models = [
                    ('rf', self.create_model('RandomForest')),
                    ('xgb', self.create_model('XGBoost')),
                    ('lgb', self.create_model('LightGBM'))
                ]
            
            # Garantir que os estimadores passados ao VotingRegressor sejam
            # single-output (desembrulhar MultiOutputRegressor se necessário)
            unwrapped = []
            for name, est in base_models:
                if hasattr(est, 'estimator'):
                    # estimador foi envolvido por MultiOutputRegressor
                    base_est = est.estimator
                else:
                    base_est = est
                unwrapped.append((name, base_est))

            # VotingRegressor não suporta multi-output diretamente
            voting_regressor = VotingRegressor(estimators=unwrapped)
            ensemble = MultiOutputRegressor(voting_regressor)
            ensemble.model_name = 'EnsembleVoting'
            return ensemble
            
        except Exception as e:
            self.logger.warning(f"Erro ao criar ensemble: {e}")
            return None
    
    def create_stacking_model(self, base_models=None, final_estimator=None):
        """Cria um modelo stacking"""
        try:
            from sklearn.ensemble import StackingRegressor
            
            if base_models is None:
                base_models = [
                    ('rf', self.create_model('RandomForest')),
                    ('xgb', self.create_model('XGBoost')),
                    ('lgb', self.create_model('LightGBM'))
                ]
            
            if final_estimator is None:
                final_estimator = Ridge(alpha=1.0)
            
            stacking = StackingRegressor(
                estimators=base_models,
                final_estimator=final_estimator,
                cv=5,
                n_jobs=-1
            )
            stacking.model_name = 'Stacking'
            return stacking
            
        except Exception as e:
            self.logger.warning(f"Erro ao criar stacking: {e}")
            return None
    
    def create_all_models(self, model_names=None, priority_only=False, 
                          include_ensemble=False, n_features=None):
        """Cria todos os modelos especificados"""
        if model_names is None:
            if priority_only:
                model_names = self.PRIORITY_MODELS
            else:
                model_names = self.DEFAULT_MODELS
        
        models = {}
        
        for model_name in model_names:
            try:
                models[model_name] = self.create_model(
                    model_name, 
                    n_features=n_features
                )
                self.logger.info(f"✅ Modelo criado: {model_name}")
            except Exception as e:
                self.logger.warning(f"⚠️ Erro ao criar modelo {model_name}: {e}")
                # Tentar criar com parâmetros mínimos
                try:
                    models[model_name] = self.create_model(
                        model_name, 
                        params={'random_state': self.DEFAULT_SEED},
                        n_features=n_features
                    )
                    self.logger.info(f"✅ Modelo criado (parâmetros mínimos): {model_name}")
                except:
                    self.logger.error(f"❌ Falha ao criar {model_name} mesmo com parâmetros mínimos")
        
        # Adicionar modelos ensemble se solicitado
        if include_ensemble and len(models) >= 2:
            try:
                from sklearn.ensemble import VotingRegressor
                from sklearn.multioutput import MultiOutputRegressor
                
                # Usar os modelos já criados como base
                base_models = []
                for name, model in list(models.items())[:3]:
                    # Se o modelo for um MultiOutputRegressor, desembrulhar o estimador
                    if hasattr(model, 'estimator'):
                        base_est = model.estimator
                    else:
                        base_est = model
                    base_models.append((name, base_est))

                # VotingRegressor não suporta multi-output diretamente
                # Precisa envolver em MultiOutputRegressor
                voting_regressor = VotingRegressor(estimators=base_models)
                ensemble = MultiOutputRegressor(voting_regressor)
                ensemble.model_name = 'EnsembleVoting'
                models['EnsembleVoting'] = ensemble
                self.logger.info("✅ Modelo criado: EnsembleVoting")
                
            except Exception as e:
                self.logger.warning(f"⚠️ Erro ao criar modelos ensemble: {e}")
        
        self.logger.info(f"📊 Total: {len(models)} modelos criados")
        return models
    
    def get_model_info(self, model_name):
        """Retorna informações sobre um modelo específico"""
        model_info = {
            'RandomForest': {
                'description': 'Random Forest - Ensemble de árvores de decisão',
                'strengths': 'Robusto, lida bem com features não-lineares',
                'weaknesses': 'Pode ser lento com muitos dados',
                'best_for': 'Problemas complexos com interações não-lineares'
            },
            'XGBoost': {
                'description': 'Gradient Boosting otimizado',
                'strengths': 'Alta performance, regularização embutida',
                'weaknesses': 'Sensível a hiperparâmetros',
                'best_for': 'Competições Kaggle, dados estruturados'
            },
            'LightGBM': {
                'description': 'Gradient Boosting com histogramas',
                'strengths': 'Muito rápido, eficiente com grandes dados',
                'weaknesses': 'Pode overfit com dados pequenos',
                'best_for': 'Grandes datasets, necessidade de velocidade'
            },
            'GradientBoosting': {
                'description': 'Gradient Boosting tradicional',
                'strengths': 'Bom poder preditivo',
                'weaknesses': 'Mais lento que XGBoost/LightGBM',
                'best_for': 'Problemas gerais de regressão'
            },
            'HistGradientBoosting': {
                'description': 'Gradient Boosting com histogramas (scikit-learn)',
                'strengths': 'Rápido, eficiente com memória',
                'weaknesses': 'Menos features que LightGBM',
                'best_for': 'Dados grandes, dentro do ecossistema scikit-learn'
            }
        }
        
        return model_info.get(model_name, {'description': 'Modelo não documentado'})

    def tune_with_optuna(self, model_name: str, X, y, n_trials: int = 20, cv_folds: int = 3, groups=None, y_unit='degrees'):
        """Rotina simples de tuning com Optuna para LightGBM/XGBoost.

        Retorna: melhores parâmetros e estudo (se optuna estiver disponível).
        """
        try:
            import optuna
        except ImportError:
            self.logger.warning("Optuna não instalado. Instale com `pip install optuna` para usar tuning.")
            return None

        from . import model_factory as mf  # evitar conflitos
        from training.cross_validation import CrossValidator

        def objective(trial):
            if model_name == 'LightGBM':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.2),
                    'num_leaves': trial.suggest_int('num_leaves', 16, 128),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                }
                model = LGBMRegressor(**params)
            elif model_name == 'XGBoost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.2),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                }
                model = XGBRegressor(**params)
            elif model_name == 'GradientBoosting':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 800),
                    'learning_rate': trial.suggest_loguniform('learning_rate', 0.005, 0.2),
                    'max_depth': trial.suggest_int('max_depth', 2, 8),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                }
                # max_features pode ser 'sqrt' ou a fração
                mf_choice = trial.suggest_categorical('max_features_choice', ['sqrt', 'log2', 'frac', 'none'])
                if mf_choice == 'frac':
                    params['max_features'] = trial.suggest_float('max_features_frac', 0.2, 1.0)
                elif mf_choice == 'none':
                    params['max_features'] = None
                else:
                    params['max_features'] = mf_choice
                model = GradientBoostingRegressor(**params)
            else:
                raise ValueError('Optuna tuning only supported for LightGBM and XGBoost in this helper')

            # envolver em MultiOutput se necessário
            try:
                from sklearn.multioutput import MultiOutputRegressor
                model = MultiOutputRegressor(model)
            except Exception:
                pass

            validator = CrossValidator(n_splits=cv_folds, random_state=42, shuffle=True)
            res = validator.cross_validate(model, X, y, model_name=model_name, verbose=False, groups=groups, y_unit=y_unit)
            return res['mean_error']

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)

        self.logger.info(f"Optuna best value: {study.best_value}, params: {study.best_params}")
        return {'study': study, 'best_params': study.best_params, 'best_value': study.best_value}

# Teste da classe
if __name__ == "__main__":
    factory = ModelFactory(n_samples=1000)
    
    print("🧪 Testando ModelFactory...")
    
    # Testar criação de modelos individuais
    test_models = ['RandomForest', 'XGBoost', 'LightGBM', 'GradientBoosting']
    
    for model_name in test_models:
        try:
            model = factory.create_model(model_name)
            print(f"✅ {model_name}: {type(model).__name__}")
            
            # Verificar parâmetros
            if hasattr(model, 'get_params'):
                params = model.get_params()
                print(f"   Parâmetros: {list(params.keys())[:5]}...")
        except Exception as e:
            print(f"❌ {model_name}: {e}")
    
    # Testar criação de todos os modelos
    print(f"\n📊 Criando modelos prioritários...")
    models = factory.create_all_models(priority_only=True)
    print(f"   Modelos criados: {len(models)}")
    print(f"   Nomes: {list(models.keys())}")