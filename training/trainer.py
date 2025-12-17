# training/trainer.py
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
import logging
import math

class ModelTrainer:
    def __init__(self):
        self.logger = self._get_logger()
        self.best_model_info = None
        self.results = {}
    
    def _get_logger(self):
        import logging
        return logging.getLogger(__name__)
    
    def haversine_distance(self, y_true, y_pred):
        """
        Calcula a distância Haversine (em km) entre coordenadas reais e previstas.
        
        Fórmula:
        a = sin²(Δφ/2) + cos φ1 ⋅ cos φ2 ⋅ sin²(Δλ/2)
        c = 2 ⋅ atan2(√a, √(1−a))
        d = R ⋅ c
        
        Onde:
        φ = latitude em radianos
        λ = longitude em radianos
        R = raio da Terra (6371 km)
        """
        # Raio da Terra em quilômetros
        R = 6371.0
        
        distances = []
        
        for i in range(len(y_true)):
            # Converter graus decimais para radianos
            lat1 = math.radians(y_true[i, 0])
            lon1 = math.radians(y_true[i, 1])
            lat2 = math.radians(y_pred[i, 0])
            lon2 = math.radians(y_pred[i, 1])
            
            # Diferenças
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            
            # Fórmula Haversine
            a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            distance = R * c
            
            distances.append(distance)
        
        return np.mean(distances)
    
    def haversine_distance_vectorized(self, y_true, y_pred):
        """
        Versão vetorizada da distância Haversine (mais rápida).
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
    
    def calculate_error_metrics(self, y_true, y_pred):
        """
        Calcula várias métricas de erro.
        
        Retorna:
        - haversine_km: Distância Haversine média (km)
        - mse_lat: MSE latitude
        - mse_lon: MSE longitude  
        - mae_lat: MAE latitude
        - mae_lon: MAE longitude
        """
        # Distância Haversine
        haversine_km = self.haversine_distance_vectorized(y_true, y_pred)
        
        # Erros por coordenada
        mse_lat = mean_squared_error(y_true[:, 0], y_pred[:, 0])
        mse_lon = mean_squared_error(y_true[:, 1], y_pred[:, 1])
        
        mae_lat = mean_absolute_error(y_true[:, 0], y_pred[:, 0])
        mae_lon = mean_absolute_error(y_true[:, 1], y_pred[:, 1])
        
        return {
            'haversine_km': haversine_km,
            'mse_lat': mse_lat,
            'mse_lon': mse_lon,
            'mae_lat': mae_lat,
            'mae_lon': mae_lon
        }
    
    def train_model_cv(self, model, X, y, cv_folds=5, model_name=None):
        """Treina modelo com validação cruzada"""
        
        if model_name is None:
            model_name = getattr(model, 'model_name', type(model).__name__)
        
        self.logger.info(f"Treinando {model_name} com {cv_folds}-fold CV")
        
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        fold_scores = []
        fold_times = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            start_time = time.time()
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Treinar modelo
            model.fit(X_train, y_train)
            
            # Fazer predições
            y_pred = model.predict(X_val)
            
            # Calcular erro (distância Haversine)
            error = self.haversine_distance_vectorized(y_val, y_pred)
            fold_time = time.time() - start_time
            
            fold_scores.append(error)
            fold_times.append(fold_time)
            
            self.logger.info(f"  Fold {fold}/{cv_folds}: "
                           f"Erro: {error:.4f} km, Tempo: {fold_time:.2f}s")
        
        mean_error = np.mean(fold_scores)
        std_error = np.std(fold_scores)
        mean_time = np.mean(fold_times)
        
        self.logger.info(f"Resultado {model_name}: {mean_error:.4f} ± {std_error:.4f} km")
        
        return {
            'model': model,
            'mean_error': mean_error,
            'std_error': std_error,
            'fold_scores': fold_scores,
            'mean_time': mean_time,
            'model_name': model_name
        }
    
    def train_all_models(self, X, y, models, cv_folds=5):
        """Treina todos os modelos"""
        self.logger.info(f"Iniciando treinamento de {len(models)} modelos")
        
        results = {}
        
        for name, model in models.items():
            try:
                self.logger.info(f"--- {name} ---")
                result = self.train_model_cv(model, X, y, cv_folds, name)
                results[name] = result
                
                # Atualizar melhor modelo
                if self.best_model_info is None or result['mean_error'] < self.best_model_info['mean_error']:
                    self.best_model_info = result
                    self.logger.info(f"🏆 Novo melhor modelo: {name} ({result['mean_error']:.4f} km)")
                    
            except Exception as e:
                self.logger.error(f"❌ Erro ao treinar {name}: {e}")
                # Log mais detalhado para debug
                import traceback
                self.logger.debug(f"Traceback: {traceback.format_exc()}")
        
        self.results = results
        
        # Ordenar resultados por erro
        if results:
            sorted_results = dict(sorted(results.items(), 
                                       key=lambda x: x[1]['mean_error']))
            
            self.logger.info("\n📊 RANKING DE MODELOS (menor erro é melhor):")
            for i, (name, result) in enumerate(sorted_results.items(), 1):
                self.logger.info(f"  {i:2d}. {name:20s}: {result['mean_error']:.4f} ± {result['std_error']:.4f} km")
        else:
            sorted_results = {}
            self.logger.warning("Nenhum modelo foi treinado com sucesso")
        
        return sorted_results
    
    def train_final_model(self, X, y):
        """Treina o melhor modelo em todos os dados"""
        if self.best_model_info is None:
            self.logger.warning("Nenhum modelo treinado. Treinando RandomForest por padrão.")
            from models.model_factory import ModelFactory
            factory = ModelFactory()
            model = factory.create_model('RandomForest')
            model_name = 'RandomForest'
        else:
            model = self.best_model_info['model']
            model_name = self.best_model_info['model_name']
            # Criar nova instância para treinar em todos os dados
            # Alguns modelos (como CatBoost) não podem ser reutilizados
            # Então criamos um novo com os mesmos parâmetros
            try:
                from models.model_factory import ModelFactory
                factory = ModelFactory()
                model = factory.create_model(model_name)
            except:
                self.logger.warning(f"Não foi possível recriar {model_name}, reusando modelo existente")
        
        self.logger.info(f"Treinando modelo final: {model_name} em {len(X)} amostras")
        
        start_time = time.time()
        model.fit(X, y)
        train_time = time.time() - start_time
        
        self.logger.info(f"Modelo final treinado em {train_time:.2f}s")
        
        return {
            'model': model,
            'model_name': model_name,
            'train_time': train_time
        }
    
    def evaluate_model(self, model, X_test, y_test, model_name=None):
        """Avalia modelo em conjunto de teste"""
        if model_name is None:
            model_name = getattr(model, 'model_name', type(model).__name__)
        
        self.logger.info(f"Avaliando {model_name} em conjunto de teste...")
        
        start_time = time.time()
        y_pred = model.predict(X_test)
        predict_time = time.time() - start_time
        
        # Calcular métricas
        metrics = self.calculate_error_metrics(y_test, y_pred)
        
        self.logger.info(f"Resultados {model_name}:")
        self.logger.info(f"  Distância Haversine: {metrics['haversine_km']:.4f} km")
        self.logger.info(f"  MSE Latitude: {metrics['mse_lat']:.6f}")
        self.logger.info(f"  MSE Longitude: {metrics['mse_lon']:.6f}")
        self.logger.info(f"  MAE Latitude: {metrics['mae_lat']:.6f}")
        self.logger.info(f"  MAE Longitude: {metrics['mae_lon']:.6f}")
        self.logger.info(f"  Tempo predição: {predict_time:.4f}s")
        
        return {
            'model': model,
            'model_name': model_name,
            'y_pred': y_pred,
            'metrics': metrics,
            'predict_time': predict_time
        }

# Função auxiliar para testar o cálculo Haversine
def test_haversine():
    """Testa a função Haversine"""
    trainer = ModelTrainer()
    
    # Coordenadas conhecidas (São Paulo para Rio de Janeiro)
    # São Paulo: -23.550520, -46.633308
    # Rio de Janeiro: -22.906847, -43.172897
    # Distância real: ~358 km
    
    y_true = np.array([[-23.550520, -46.633308]])  # São Paulo
    y_pred = np.array([[-22.906847, -43.172897]])  # Rio de Janeiro
    
    distance = trainer.haversine_distance(y_true, y_pred)
    distance_vec = trainer.haversine_distance_vectorized(y_true, y_pred)
    
    print(f"Distância calculada (loop): {distance:.2f} km")
    print(f"Distância calculada (vetorizada): {distance_vec:.2f} km")
    print(f"Distância real aproximada: 358 km")
    
    return abs(distance - 358) < 20  # Aceita diferença de até 20km

if __name__ == "__main__":
    # Testar função Haversine
    print("🧪 Testando cálculo Haversine...")
    if test_haversine():
        print("✅ Teste passou!")
    else:
        print("❌ Teste falhou!")
    
    # Testar treinador
    print("\n🧪 Testando ModelTrainer...")
    
    # Dados dummy para teste
    np.random.seed(42)
    X_dummy = np.random.randn(100, 10)
    y_dummy = np.random.randn(100, 2) * 0.1 + np.array([-23.5, -46.6])
    
    from models.model_factory import ModelFactory
    
    factory = ModelFactory(n_samples=100)
    models = factory.create_all_models(
        model_names=['RandomForest', 'LinearRegression'],
        priority_only=False
    )
    
    trainer = ModelTrainer()
    results = trainer.train_all_models(X_dummy, y_dummy, models, cv_folds=3)
    
    print(f"\n✅ Teste concluído! Modelos treinados: {len(results)}")