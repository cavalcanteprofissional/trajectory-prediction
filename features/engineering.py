# features/engineering.py
import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2
from pathlib import Path
import math

class FeatureEngineer:
    """Classe para engenharia de features"""
    
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
        
        lat_list = row.get('path_lat_parsed', [])
        lon_list = row.get('path_lon_parsed', [])
        
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
        
        lat_list = row.get('path_lat_parsed', [])
        lon_list = row.get('path_lon_parsed', [])
        
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
        
        lat_list = row.get('path_lat_parsed', [])
        lon_list = row.get('path_lon_parsed', [])
        
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
    
    def extract_all_features(self, df):
        """Extrai todas as features para um DataFrame"""
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
                
                # NOVAS FEATURES PARA MELHORAR PREDIÇÃO
                # Direção média (bearing) do início ao fim
                if len(lat_list) >= 2:
                    start_lat, start_lon = lat_list[0], lon_list[0]
                    end_lat, end_lon = lat_list[-1], lon_list[-1]
                    
                    # Calcular bearing (direção)
                    lat1, lon1 = radians(start_lat), radians(start_lon)
                    lat2, lon2 = radians(end_lat), radians(end_lon)
                    dlon = lon2 - lon1
                    
                    y = sin(dlon) * cos(lat2)
                    x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
                    bearing = atan2(y, x)
                    bearing_degrees = (bearing * 180 / 3.14159) % 360
                    
                    features['bearing'] = bearing_degrees
                    features['bearing_sin'] = sin(bearing)
                    features['bearing_cos'] = cos(bearing)
                
                # Velocidade estimada (assumindo 1 segundo entre pontos)
                if features.get('total_distance', 0) > 0 and len(lat_list) > 1:
                    # Velocidade média em m/s
                    time_seconds = len(lat_list) - 1  # Assumindo 1 segundo entre pontos
                    features['avg_speed_ms'] = features['total_distance'] / time_seconds if time_seconds > 0 else 0
                    features['avg_speed_kmh'] = features['avg_speed_ms'] * 3.6
                
                # Features de direção e mudança de direção
                if len(lat_list) >= 3:
                    # Calcular mudanças de direção
                    bearings = []
                    for i in range(1, len(lat_list)):
                        lat1, lon1 = radians(lat_list[i-1]), radians(lon_list[i-1])
                        lat2, lon2 = radians(lat_list[i]), radians(lon_list[i])
                        dlon = lon2 - lon1
                        
                        y = sin(dlon) * cos(lat2)
                        x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
                        bearing = atan2(y, x)
                        bearings.append(bearing)
                    
                    if bearings:
                        # Variabilidade de direção (quanto mais variável, mais curvas)
                        bearing_changes = [abs(bearings[i] - bearings[i-1]) for i in range(1, len(bearings))]
                        features['direction_variance'] = np.std(bearings) if len(bearings) > 1 else 0
                        features['avg_direction_change'] = np.mean(bearing_changes) if bearing_changes else 0
                
                # Distância do último ponto ao destino (se disponível)
                if 'dest_lat' in row and 'dest_lon' in row and not pd.isna(row['dest_lat']):
                    last_lat, last_lon = lat_list[-1], lon_list[-1]
                    dest_lat, dest_lon = row['dest_lat'], row['dest_lon']
                    remaining_distance = self.haversine_distance(last_lat, last_lon, dest_lat, dest_lon)
                    features['remaining_distance'] = remaining_distance
                else:
                    features['remaining_distance'] = 0
            
            features_list.append(features)
        
        features_df = pd.DataFrame(features_list)
        
        # Preencher valores nulos
        features_df = features_df.fillna(0)
        
        # Substituir infinitos por 0
        features_df = features_df.replace([np.inf, -np.inf], 0)
        
        self.logger.info(f"Features extraídas: {features_df.shape}")
        
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