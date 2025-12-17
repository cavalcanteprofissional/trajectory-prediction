# models/model_factory.py
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

class ModelFactory:
    """Fábrica para criação de modelos"""
    
    def __init__(self):
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
        
    @staticmethod
    def create_model(model_name, params=None):
        """Cria um modelo baseado no nome"""
        
        default_params = {
            'random_state': config.SEED,
            'n_jobs': -1
        }
        
        if params:
            default_params.update(params)
        
        models_map = {
            'RandomForest': RandomForestRegressor(
                n_estimators=100,
                **default_params
            ),
            'XGBoost': XGBRegressor(
                n_estimators=100,
                **default_params
            ),
            'LightGBM': LGBMRegressor(
                n_estimators=100,
                **default_params
            ),
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=100,
                random_state=config.SEED
            ),
            'CatBoost': CatBoostRegressor(
                iterations=100,
                random_state=config.SEED,
                verbose=0
            ),
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(
                alpha=1.0,
                random_state=config.SEED
            )
        }
        
        if model_name not in models_map:
            raise ValueError(f"Modelo desconhecido: {model_name}. "
                           f"Modelos disponíveis: {list(models_map.keys())}")
        
        # Criar modelo multi-output
        model = MultiOutputRegressor(models_map[model_name])
        logger.info(f"Modelo criado: {model_name}")
        
        return model
    
    @classmethod
    def create_all_models(cls, model_names=None):
        """Cria todos os modelos especificados"""
        if model_names is None:
            model_names = config.DEFAULT_MODELS
        
        models = {}
        for model_name in model_names:
            try:
                models[model_name] = cls.create_model(model_name)
            except Exception as e:
                logger.warning(f"Erro ao criar modelo {model_name}: {e}")
        
        logger.info(f"{len(models)} modelos criados")
        return models