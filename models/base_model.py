# models/base_model.py
"""
Classe base para modelos de predição de trajetórias
"""
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import logging


class BaseTrajectoryModel(ABC):
    """
    Classe base abstrata para modelos de predição de trajetórias.
    
    Esta classe define a interface comum para todos os modelos de predição
    de coordenadas geográficas (latitude e longitude).
    """
    
    def __init__(self, model_name: str, random_state: int = 42):
        """
        Inicializa o modelo base
        
        Args:
            model_name: Nome do modelo
            random_state: Seed para reprodutibilidade
        """
        self.model_name = model_name
        self.random_state = random_state
        self.model = None
        self.is_trained = False
        self.logger = self._get_logger()
    
    def _get_logger(self):
        """Obtém logger"""
        try:
            from utils.logger import get_logger
            return get_logger(__name__)
        except ImportError:
            import logging
            logging.basicConfig(level=logging.INFO)
            return logging.getLogger(__name__)
    
    @abstractmethod
    def _create_model(self) -> Any:
        """
        Cria a instância do modelo específico.
        Deve ser implementado por classes filhas.
        
        Returns:
            Instância do modelo scikit-learn compatível
        """
        pass
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseTrajectoryModel':
        """
        Treina o modelo
        
        Args:
            X: Features de treino (n_samples, n_features)
            y: Targets de treino (n_samples, 2) - [lat, lon]
            
        Returns:
            self (para permitir method chaining)
        """
        if self.model is None:
            self.model = self._create_model()
        
        self.logger.info(f"Treinando {self.model_name} em {len(X)} amostras...")
        self.model.fit(X, y)
        self.is_trained = True
        self.logger.info(f"✅ {self.model_name} treinado com sucesso")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Faz predições
        
        Args:
            X: Features de teste (n_samples, n_features)
            
        Returns:
            Predições (n_samples, 2) - [lat, lon]
        """
        if not self.is_trained:
            raise ValueError(f"Modelo {self.model_name} não foi treinado. Chame fit() primeiro.")
        
        return self.model.predict(X)
    
    def get_params(self) -> Dict[str, Any]:
        """
        Retorna parâmetros do modelo
        
        Returns:
            Dicionário com parâmetros
        """
        if self.model is None:
            return {}
        
        if hasattr(self.model, 'get_params'):
            return self.model.get_params()
        
        return {}
    
    def set_params(self, **params) -> 'BaseTrajectoryModel':
        """
        Define parâmetros do modelo
        
        Args:
            **params: Parâmetros a serem definidos
            
        Returns:
            self
        """
        if self.model is None:
            self.model = self._create_model()
        
        if hasattr(self.model, 'set_params'):
            self.model.set_params(**params)
        
        return self
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calcula score do modelo (R² por padrão)
        
        Args:
            X: Features
            y: Targets verdadeiros
            
        Returns:
            Score do modelo
        """
        if not self.is_trained:
            raise ValueError(f"Modelo {self.model_name} não foi treinado.")
        
        if hasattr(self.model, 'score'):
            return self.model.score(X, y)
        
        # Fallback: calcular R² manualmente
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return np.mean(r2)
    
    def __repr__(self) -> str:
        """Representação string do modelo"""
        status = "treinado" if self.is_trained else "não treinado"
        return f"{self.__class__.__name__}(model_name='{self.model_name}', status='{status}')"


class WrappedModel(BaseTrajectoryModel):
    """
    Wrapper para modelos scikit-learn existentes.
    Permite usar qualquer modelo scikit-learn como BaseTrajectoryModel.
    """
    
    def __init__(self, model: Any, model_name: Optional[str] = None):
        """
        Inicializa wrapper com modelo existente
        
        Args:
            model: Modelo scikit-learn
            model_name: Nome do modelo (opcional)
        """
        if model_name is None:
            model_name = getattr(model, 'model_name', type(model).__name__)
        
        super().__init__(model_name)
        self.model = model
        
        # Se o modelo já foi treinado, marcar como tal
        if hasattr(model, 'fit') and hasattr(model, 'predict'):
            # Verificar se tem atributos que indicam treinamento
            if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_'):
                self.is_trained = True
    
    def _create_model(self) -> Any:
        """Retorna o modelo já existente"""
        return self.model
