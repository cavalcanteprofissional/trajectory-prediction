# training/trainer.py
import numpy as np
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
    
    def train_model_cv(self, model, X, y, cv_folds=10, model_name=None):
        """Treina modelo com validação cruzada usando o módulo cross_validation"""
        
        # Importar CrossValidator
        from .cross_validation import CrossValidator
        
        if model_name is None:
            model_name = getattr(model, 'model_name', type(model).__name__)
        
        # Usar CrossValidator
        validator = CrossValidator(n_splits=cv_folds, random_state=42, shuffle=True)
        result = validator.cross_validate(model, X, y, model_name=model_name, verbose=True)
        
        return result
    
    def train_all_models(self, X, y, models, cv_folds=10):
        """Treina todos os modelos usando validação cruzada"""
        
        # Usar CrossValidator para validar múltiplos modelos
        from .cross_validation import CrossValidator
        
        validator = CrossValidator(n_splits=cv_folds, random_state=42, shuffle=True)
        results = validator.validate_multiple_models(models, X, y, verbose=True)
                
                # Atualizar melhor modelo
        if results:
            best_name = min(results.keys(), key=lambda k: results[k]['mean_error'])
            self.best_model_info = results[best_name]
            self.logger.info(f"🏆 Melhor modelo: {best_name} ({self.best_model_info['mean_error']:.4f} km)")
        
        self.results = results
        
        return results
    
    def train_final_model(self, X, y):
        """Treina o melhor modelo em todos os dados"""
        if self.best_model_info is None:
            self.logger.warning("Nenhum modelo treinado. Treinando RandomForest por padrão.")
            from models import ModelFactory
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
                from models import ModelFactory
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