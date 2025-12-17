# features/engineering.py
import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2
from pathlib import Path

# NÃO inicialize logger aqui no nível do módulo
# Isso causa erro se get_logger não estiver disponível

class FeatureEngineer:
    """Classe para engenharia de features"""
    
    def __init__(self):
        # Inicializar logger apenas quando a classe for instanciada
        self.logger = self._get_logger()
    
    def _get_logger(self):
        """Obtém logger de forma segura"""
        try:
            from utils.logger import get_logger
            return get_logger(__name__)
        except ImportError:
            # Fallback básico
            import logging
            logging.basicConfig(level=logging.INFO)
            return logging.getLogger(__name__)
    
    @staticmethod
    def haversine_distance(lat1, lon1, lat2, lon2):
        """Calcula distância Haversine entre dois pontos em metros"""
        R = 6371000  # raio da Terra em metros
        
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        return R * c
    
    @classmethod
    def extract_basic_features(cls, row):
        """Extrai features básicas de uma trajetória"""
        features = {}
        
        lat_list = row['path_lat_parsed']
        lon_list = row['path_lon_parsed']
        
        if len(lat_list) < 2:
            return features
        
        # Features de posição
        features['start_lat'] = lat_list[0]
        features['start_lon'] = lon_list[0]
        features['end_lat'] = lat_list[-1]
        features['end_lon'] = lon_list[-1]
        
        # Estatísticas
        features['mean_lat'] = np.mean(lat_list)
        features['mean_lon'] = np.mean(lon_list)
        features['std_lat'] = np.std(lat_list)
        features['std_lon'] = np.std(lon_list)
        features['median_lat'] = np.median(lat_list)
        features['median_lon'] = np.median(lon_list)
        
        return features
    
    @classmethod
    def extract_distance_features(cls, row):
        """Extrai features relacionadas a distâncias"""
        features = {}
        
        lat_list = row['path_lat_parsed']
        lon_list = row['path_lon_parsed']
        
        if len(lat_list) < 2:
            return features
        
        # Calcular distâncias entre pontos consecutivos
        total_distance = 0
        distances = []
        
        for i in range(1, len(lat_list)):
            dist = cls.haversine_distance(
                lat_list[i-1], lon_list[i-1],
                lat_list[i], lon_list[i]
            )
            total_distance += dist
            distances.append(dist)
        
        features['total_distance'] = total_distance
        features['mean_distance'] = np.mean(distances) if distances else 0
        features['std_distance'] = np.std(distances) if distances else 0
        
        # Distância em linha reta do início ao fim
        straight_distance = cls.haversine_distance(
            lat_list[0], lon_list[0],
            lat_list[-1], lon_list[-1]
        )
        
        features['straight_distance'] = straight_distance
        features['straightness'] = straight_distance / total_distance if total_distance > 0 else 1
        
        return features
    
    @classmethod
    def extract_geometric_features(cls, row):
        """Extrai features geométricas"""
        features = {}
        
        lat_list = row['path_lat_parsed']
        lon_list = row['path_lon_parsed']
        
        if len(lat_list) < 2:
            return features
        
        # Features geométricas
        features['lat_range'] = np.max(lat_list) - np.min(lat_list)
        features['lon_range'] = np.max(lon_list) - np.min(lon_list)
        features['area_bbox'] = features['lat_range'] * features['lon_range']
        
        # Razão aspecto
        features['aspect_ratio'] = (
            features['lat_range'] / features['lon_range'] 
            if features['lon_range'] != 0 else 0
        )
        
        # Centroide
        features['centroid_lat'] = np.mean(lat_list)
        features['centroid_lon'] = np.mean(lon_list)
        
        return features
    
   # features/engineering.py - método extract_all_features corrigido
def extract_all_features(self, df):
    """Extrai todas as features para um DataFrame"""
    # Verificar se já foi processado
    if hasattr(df, '_features_extracted'):
        self.logger.warning(f"Features já extraídas para DataFrame com {len(df)} linhas")
        # Se já temos features extraídas, retornar cache
        if hasattr(df, '_cached_features'):
            return df._cached_features.copy()
    
    self.logger.info(f"Extraindo features de {len(df)} trajetórias...")
    
    features_list = []
    
    for idx, row in df.iterrows():
        features = {}
        
        # Combinar todas as features
        features.update(self.extract_basic_features(row))
        features.update(self.extract_distance_features(row))
        features.update(self.extract_geometric_features(row))
        
        # Features adicionais
        lat_list = row.get('path_lat_parsed', [])
        lon_list = row.get('path_lon_parsed', [])
        
        if len(lat_list) > 0:
            features['num_points'] = len(lat_list)
            if features.get('total_distance', 0) > 0:
                features['density'] = len(lat_list) / features['total_distance']
            else:
                features['density'] = 0
            
            # Hash do user_id para usar como feature
            features['user_id_hash'] = hash(row.get('user_id', 'unknown')) % 1000 if 'user_id' in row else 0
        
        features_list.append(features)
    
    features_df = pd.DataFrame(features_list)
    
    # Preencher valores nulos
    features_df = features_df.fillna(0)
    
    self.logger.info(f"Features extraídas: {features_df.shape}")
    
    # Cache para evitar reprocessamento
    df._features_extracted = True
    df._cached_features = features_df.copy()
    
    return features_df
    @staticmethod
    def prepare_features_for_training(train_features, test_features, target_cols=['dest_lat', 'dest_lon']):
        """Prepara features para treinamento"""
        from sklearn.preprocessing import StandardScaler
        
        # Separar features e target
        feature_cols = [col for col in train_features.columns 
                       if col not in target_cols]
        
        X_train = train_features[feature_cols].values
        y_train = train_features[target_cols].values
        X_test = test_features[feature_cols].values
        
        # Normalizar features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Log apenas se logger estiver disponível
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.info(f"Features preparadas - Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")
        except:
            pass
        
        return {
            'X_train': X_train_scaled,
            'y_train': y_train,
            'X_test': X_test_scaled,
            'feature_names': feature_cols,
            'scaler': scaler
        }