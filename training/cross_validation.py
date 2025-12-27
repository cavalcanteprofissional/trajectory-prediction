# training/cross_validation.py
"""
Módulo para validação cruzada com métrica Haversine customizada
"""
import numpy as np
import time
from sklearn.model_selection import KFold, GroupKFold
from typing import Dict, List, Tuple, Optional, Any
import logging


class CrossValidator:
    """Classe para validação cruzada com métrica Haversine"""
    
    def __init__(self, n_splits: int = 10, random_state: int = 42, shuffle: bool = True):
        """
        Inicializa o validador cruzado
        
        Args:
            n_splits: Número de folds (padrão: 10)
            random_state: Seed para reprodutibilidade
            shuffle: Se True, embaralha os dados antes de dividir
        """
        self.n_splits = n_splits
        self.random_state = random_state
        self.shuffle = shuffle
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
    
    @staticmethod
    def haversine_distance_vectorized(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calcula a distância Haversine média (em km) entre coordenadas reais e previstas.
        Versão vetorizada para melhor performance.
        
        Args:
            y_true: Array de shape (n_samples, 2) com [lat, lon] verdadeiros
            y_pred: Array de shape (n_samples, 2) com [lat, lon] previstos
            
        Returns:
            Distância Haversine média em quilômetros
        """
        # Raio da Terra em quilômetros
        R = 6371.0
        
        # Converter para radianos
        lat1 = np.radians(y_true[:, 0])
        lon1 = np.radians(y_true[:, 1])
        lat2 = np.radians(y_pred[:, 0])
        lon2 = np.radians(y_pred[:, 1])
        
        # Diferenças
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        # Fórmula Haversine vetorizada
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        distances = R * c
        
        return np.mean(distances)
    
    @staticmethod
    def _local_xy_to_latlon(lat_ref, lon_ref, dx, dy):
        """Converte coordenadas locais (m) de volta para lat/lon (graus)."""
        R = 6371000.0
        lat0 = np.radians(lat_ref)
        lon0 = np.radians(lon_ref)

        lat = lat0 + dy / R
        lon = lon0 + dx / (R * np.cos((lat + lat0) / 2.0))
        return np.degrees(lat), np.degrees(lon)
    
    def cross_validate(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        model_name: Optional[str] = None,
        verbose: bool = True,
        groups: Optional[np.ndarray] = None,
        y_unit: str = 'degrees',  # 'degrees' (lat/lon) or 'meters' (dx,dy)
        refs_lat: Optional[np.ndarray] = None,
        refs_lon: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Executa validação cruzada com métrica Haversine
        
        Args:
            model: Modelo scikit-learn compatível
            X: Features de treino (n_samples, n_features)
            y: Targets de treino (n_samples, 2) - [lat, lon]
            model_name: Nome do modelo para logging
            verbose: Se True, imprime progresso
            
        Returns:
            Dicionário com resultados da validação cruzada:
            {
                'mean_error': float,  # Erro médio (km)
                'std_error': float,    # Desvio padrão do erro
                'fold_scores': List[float],  # Erros por fold
                'fold_times': List[float],   # Tempos por fold
                'mean_time': float,   # Tempo médio por fold
                'model': model,       # Modelo treinado
                'model_name': str     # Nome do modelo
            }
        """
        if model_name is None:
            model_name = getattr(model, 'model_name', type(model).__name__)
        
        if verbose:
            self.logger.info(f"Treinando {model_name} com {self.n_splits}-fold CV")
        
        # Escolher estratégia de split: GroupKFold se groups fornecidos
        if groups is not None:
            if len(groups) != len(X):
                raise ValueError("Length of groups must match number of samples")
            kf = GroupKFold(n_splits=self.n_splits)
            splitter = kf.split(X, groups=groups)
        else:
            kf = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
            splitter = kf.split(X)
        
        fold_scores = []
        fold_times = []
        
        for fold, (train_idx, val_idx) in enumerate(splitter, 1):
            start_time = time.time()
            
            # Dividir dados
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Criar nova instância do modelo para cada fold
            # (alguns modelos não podem ser reutilizados)
            model_copy = self._clone_model(model)
            
            # Garantir que X está no formato ndarray para evitar warnings de feature names
            X_train = np.asarray(X_train)
            X_val = np.asarray(X_val)

            # Treinar modelo
            import warnings as _warnings
            with _warnings.catch_warnings():
                # Ignorar avisos de incompatibilidade de nomes de features do sklearn/LightGBM
                _warnings.filterwarnings("ignore", message="X does not have valid feature names")
                model_copy.fit(X_train, y_train)
            
            # Fazer predições (usar ndarray consistente)
            with _warnings.catch_warnings():
                _warnings.filterwarnings("ignore", message="X does not have valid feature names")
                y_pred = model_copy.predict(X_val)
            
            # Se y_unit for 'meters', converter dx,dy de volta para lat/lon
            if y_unit == 'meters' and refs_lat is not None and refs_lon is not None:
                refs_val_lat = refs_lat[val_idx]
                refs_val_lon = refs_lon[val_idx]
                
                y_val_latlon = []
                y_pred_latlon = []
                
                for i in range(len(y_val)):
                    lat_true, lon_true = self._local_xy_to_latlon(
                        refs_val_lat[i], refs_val_lon[i], y_val[i, 0], y_val[i, 1]
                    )
                    lat_pred, lon_pred = self._local_xy_to_latlon(
                        refs_val_lat[i], refs_val_lon[i], y_pred[i, 0], y_pred[i, 1]
                    )
                    y_val_latlon.append([lat_true, lon_true])
                    y_pred_latlon.append([lat_pred, lon_pred])
                
                y_val = np.array(y_val_latlon)
                y_pred = np.array(y_pred_latlon)
            
            # Calcular erro com Haversine (sempre em lat/lon após conversão)
            error = self.haversine_distance_vectorized(y_val, y_pred)
            fold_time = time.time() - start_time
            
            fold_scores.append(error)
            fold_times.append(fold_time)
            
            if verbose:
                self.logger.info(f"  Fold {fold}/{self.n_splits}: "
                               f"Erro: {error:.4f} km, Tempo: {fold_time:.2f}s")
        
        # Calcular estatísticas
        mean_error = np.mean(fold_scores)
        std_error = np.std(fold_scores)
        mean_time = np.mean(fold_times)
        
        if verbose:
            self.logger.info(f"Resultado {model_name}: {mean_error:.4f} ± {std_error:.4f} km")
        
        return {
            'model': model,
            'mean_error': mean_error,
            'std_error': std_error,
            'fold_scores': fold_scores,
            'fold_times': fold_times,
            'mean_time': mean_time,
            'model_name': model_name
        }
    
    def _clone_model(self, model: Any) -> Any:
        """
        Cria uma cópia do modelo para usar em cada fold
        
        Args:
            model: Modelo original
            
        Returns:
            Nova instância do modelo com mesmos parâmetros
        """
        # Tentar usar clone do sklearn
        try:
            from sklearn.base import clone
            return clone(model)
        except:
            pass
        
        # Se não funcionar, tentar recriar baseado no tipo
        model_name = getattr(model, 'model_name', None)
        if model_name:
            try:
                from models.model_factory import ModelFactory
                factory = ModelFactory()
                return factory.create_model(model_name)
            except:
                pass
        
        # Último recurso: retornar o modelo original
        # (pode causar problemas com alguns modelos)
        return model
    
    def validate_multiple_models(
        self,
        models: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        verbose: bool = True,
        groups: Optional[np.ndarray] = None,
        y_unit: str = 'degrees',
        refs_lat: Optional[np.ndarray] = None,
        refs_lon: Optional[np.ndarray] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Valida múltiplos modelos com validação cruzada
        
        Args:
            models: Dicionário {nome: modelo}
            X: Features de treino
            y: Targets de treino
            verbose: Se True, imprime progresso
            
        Returns:
            Dicionário com resultados de cada modelo
        """
        results = {}
        
        if verbose:
            self.logger.info(f"Iniciando validação cruzada de {len(models)} modelos")
        
        for name, model in models.items():
            try:
                if verbose:
                    self.logger.info(f"--- {name} ---")
                result = self.cross_validate(model, X, y, model_name=name, verbose=verbose, groups=groups, y_unit=y_unit, refs_lat=refs_lat, refs_lon=refs_lon)
                results[name] = result
                
            except Exception as e:
                self.logger.error(f"❌ Erro ao validar {name}: {e}")
                # Log mais detalhado para debug
                import traceback
                self.logger.debug(f"Traceback: {traceback.format_exc()}")
        
        # Ordenar resultados por erro
        if results:
            sorted_results = dict(sorted(results.items(), 
                                        key=lambda x: x[1]['mean_error']))
            
            if verbose:
                self.logger.info("\n📊 RANKING DE MODELOS (menor erro é melhor):")
                for i, (name, result) in enumerate(sorted_results.items(), 1):
                    self.logger.info(f"  {i:2d}. {name:20s}: "
                                   f"{result['mean_error']:.4f} ± {result['std_error']:.4f} km")
        
        return results


# Função auxiliar para uso direto
def cross_validate_model(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 10,
    model_name: Optional[str] = None,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Função auxiliar para validação cruzada rápida
    
    Args:
        model: Modelo scikit-learn
        X: Features
        y: Targets
        n_splits: Número de folds
        model_name: Nome do modelo
        random_state: Seed
        
    Returns:
        Resultados da validação cruzada
    """
    validator = CrossValidator(n_splits=n_splits, random_state=random_state)
    return validator.cross_validate(model, X, y, model_name=model_name)


if __name__ == "__main__":
    # Teste do módulo
    print("🧪 Testando CrossValidator...")
    
    # Dados dummy
    np.random.seed(42)
    X_test = np.random.randn(100, 10)
    y_test = np.random.randn(100, 2) * 0.1 + np.array([-23.5, -46.6])
    
    from sklearn.ensemble import RandomForestRegressor
    
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    
    validator = CrossValidator(n_splits=5, random_state=42)
    results = validator.cross_validate(model, X_test, y_test, model_name="RandomForest")
    
    print(f"\n✅ Teste concluído!")
    print(f"   Erro médio: {results['mean_error']:.4f} km")
    print(f"   Desvio padrão: {results['std_error']:.4f} km")
