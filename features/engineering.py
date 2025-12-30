# features/engineering.py
import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2
from pathlib import Path
import math
from scipy.stats import skew, kurtosis

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
            if 'trajectory_id' in row:
                features['trajectory_id'] = row['trajectory_id']

            # Combinar features básicas
            features.update(self.extract_basic_features(row))
            features.update(self.extract_distance_features(row))
            features.update(self.extract_geometric_features(row))

            lat_list = row.get('path_lat_parsed', [])
            lon_list = row.get('path_lon_parsed', [])

            if len(lat_list) > 0:
                features['num_points'] = len(lat_list)
                features['density'] = (len(lat_list) / features['total_distance']) if features.get('total_distance', 0) > 0 else 0

                # calcular segmentos entre pontos
                segment_distances = []
                segment_bearings = []
                speeds = []

                for i in range(1, len(lat_list)):
                    lat1, lon1 = lat_list[i-1], lon_list[i-1]
                    lat2, lon2 = lat_list[i], lon_list[i]
                    d = self.haversine_distance(lat1, lon1, lat2, lon2)
                    segment_distances.append(d)

                    # bearing entre segmentos
                    a1, b1 = radians(lat1), radians(lon1)
                    a2, b2 = radians(lat2), radians(lon2)
                    dlon = b2 - b1
                    y = sin(dlon) * cos(a2)
                    x = cos(a1) * sin(a2) - sin(a1) * cos(a2) * cos(dlon)
                    bearing = atan2(y, x)
                    segment_bearings.append(bearing)

                    # velocidade aproximada (1s entre pontos)
                    speeds.append(d / 1.0 if d is not None else 0.0)

                # velocidade média e máxima
                if speeds:
                    features['avg_speed_ms'] = float(np.mean(speeds))
                    features['max_speed_ms'] = float(np.max(speeds))
                    accs = np.diff(speeds) if len(speeds) > 1 else np.array([0.0])
                    features['avg_acc_ms2'] = float(np.mean(accs)) if len(accs) > 0 else 0.0
                    features['max_acc_ms2'] = float(np.max(accs)) if len(accs) > 0 else 0.0

                # bearing do início ao fim
                if len(lat_list) >= 2:
                    start_lat, start_lon = lat_list[0], lon_list[0]
                    end_lat, end_lon = lat_list[-1], lon_list[-1]
                    lat1, lon1 = radians(start_lat), radians(start_lon)
                    lat2, lon2 = radians(end_lat), radians(end_lon)
                    dlon = lon2 - lon1
                    y = sin(dlon) * cos(lat2)
                    x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
                    bearing = atan2(y, x)
                    features['bearing'] = (bearing * 180 / math.pi) % 360
                    features['bearing_sin'] = math.sin(bearing)
                    features['bearing_cos'] = math.cos(bearing)

                # direction changes
                if len(lat_list) >= 3:
                    bearings = []
                    for i in range(1, len(lat_list)):
                        a1, b1 = radians(lat_list[i-1]), radians(lon_list[i-1])
                        a2, b2 = radians(lat_list[i]), radians(lon_list[i])
                        dlon = b2 - b1
                        y = sin(dlon) * cos(a2)
                        x = cos(a1) * sin(a2) - sin(a1) * cos(a2) * cos(dlon)
                        bearing = atan2(y, x)
                        bearings.append(bearing)

                    if bearings:
                        bearing_changes = [abs(bearings[i] - bearings[i-1]) for i in range(1, len(bearings))]
                        features['direction_variance'] = float(np.std(bearings)) if len(bearings) > 1 else 0.0
                        features['avg_direction_change'] = float(np.mean(bearing_changes)) if bearing_changes else 0.0

                # Curvatura aproximada
                if len(segment_bearings) > 1 and len(segment_distances) > 1:
                    bearing_diffs = np.abs(np.diff(segment_bearings))
                    dist_array = np.array(segment_distances[1:])
                    curvature = np.mean(bearing_diffs / (dist_array + 1e-6))
                    features['curvature'] = float(curvature)

                # Percentis das distâncias entre pontos
                if segment_distances:
                    seg = np.array(segment_distances)
                    features['seg_p10'] = float(np.percentile(seg, 10))
                    features['seg_p25'] = float(np.percentile(seg, 25))
                    features['seg_p50'] = float(np.percentile(seg, 50))
                    features['seg_p75'] = float(np.percentile(seg, 75))
                    features['seg_p90'] = float(np.percentile(seg, 90))
                    features['seg_max'] = float(np.max(seg))
                    features['seg_mean'] = float(np.mean(seg))
                    features['seg_std'] = float(np.std(seg))

                # NOVAS FEATURES: Percentis dos bearings
                if bearings:
                    bearings_deg = np.array(bearings) * 180 / math.pi % 360
                    features['bearing_p10'] = float(np.percentile(bearings_deg, 10))
                    features['bearing_p50'] = float(np.percentile(bearings_deg, 50))
                    features['bearing_p90'] = float(np.percentile(bearings_deg, 90))

                # NOVAS FEATURES: Percentis das velocidades
                if speeds:
                    features['speed_p10'] = float(np.percentile(speeds, 10))
                    features['speed_p50'] = float(np.percentile(speeds, 50))
                    features['speed_p90'] = float(np.percentile(speeds, 90))

                # NOVAS FEATURES: Percentis das acelerações
                if len(speeds) > 1:
                    accs = np.diff(speeds)
                    features['acc_p10'] = float(np.percentile(accs, 10))
                    features['acc_p50'] = float(np.percentile(accs, 50))
                    features['acc_p90'] = float(np.percentile(accs, 90))

                # NOVAS FEATURES: Total turning angle
                if bearing_changes:
                    features['total_turning_angle'] = float(np.sum(bearing_changes) * 180 / math.pi)

                # NOVAS FEATURES: Number of significant turns (>30 degrees)
                if bearing_changes:
                    significant_turns = sum(1 for change in bearing_changes if change * 180 / math.pi > 30)
                    features['n_significant_turns'] = significant_turns

                # NOVAS FEATURES: Bearing of last segment
                if bearings:
                    features['last_bearing'] = float(bearings[-1] * 180 / math.pi % 360)

                # NOVAS FEATURES: Speed of last segment
                if speeds:
                    features['last_speed'] = float(speeds[-1])

                # NOVAS FEATURES: Last direction change
                if bearing_changes:
                    features['last_direction_change'] = float(bearing_changes[-1] * 180 / math.pi) if bearing_changes else 0.0

                # Distância do último ponto ao destino (se disponível)
                if 'dest_lat' in row and 'dest_lon' in row and not pd.isna(row['dest_lat']):
                    last_lat, last_lon = lat_list[-1], lon_list[-1]
                    dest_lat, dest_lon = row['dest_lat'], row['dest_lon']
                    remaining_distance = self.haversine_distance(last_lat, last_lon, dest_lat, dest_lon)
                    features['remaining_distance'] = remaining_distance
                else:
                    features['remaining_distance'] = 0

                # NOVAS FEATURES: Linear trend (slope and intercept for lat/lon over point index)
                if len(lat_list) >= 3:
                    indices = np.arange(len(lat_list))
                    try:
                        lat_slope, lat_intercept = np.polyfit(indices, lat_list, 1)
                        lon_slope, lon_intercept = np.polyfit(indices, lon_list, 1)
                        features['lat_slope'] = lat_slope
                        features['lat_intercept'] = lat_intercept
                        features['lon_slope'] = lon_slope
                        features['lon_intercept'] = lon_intercept
                        # Trend magnitude
                        features['trend_magnitude'] = np.sqrt(lat_slope**2 + lon_slope**2)
                        # Trend direction (bearing of slope vector)
                        features['trend_bearing'] = (np.arctan2(lon_slope, lat_slope) * 180 / math.pi) % 360
                    except:
                        features['lat_slope'] = 0
                        features['lat_intercept'] = lat_list[0] if lat_list else 0
                        features['lon_slope'] = 0
                        features['lon_intercept'] = lon_list[0] if lon_list else 0
                        features['trend_magnitude'] = 0
                        features['trend_bearing'] = 0

                # NOVAS FEATURES: Polynomial fit (degree 2) coefficients
                if len(lat_list) >= 5:
                    try:
                        lat_poly = np.polyfit(indices, lat_list, 2)
                        lon_poly = np.polyfit(indices, lon_list, 2)
                        features['lat_poly_a'] = lat_poly[0]  # quadratic term
                        features['lat_poly_b'] = lat_poly[1]  # linear term
                        features['lat_poly_c'] = lat_poly[2]  # constant term
                        features['lon_poly_a'] = lon_poly[0]
                        features['lon_poly_b'] = lon_poly[1]
                        features['lon_poly_c'] = lon_poly[2]
                        # Curvature magnitude
                        features['poly_curvature'] = np.sqrt(lat_poly[0]**2 + lon_poly[0]**2)
                    except:
                        features['lat_poly_a'] = 0
                        features['lat_poly_b'] = 0
                        features['lat_poly_c'] = lat_list[0] if lat_list else 0
                        features['lon_poly_a'] = 0
                        features['lon_poly_b'] = 0
                        features['lon_poly_c'] = lon_list[0] if lon_list else 0
                        features['poly_curvature'] = 0

                # NOVAS FEATURES: Last segment detailed features
                if len(lat_list) >= 3 and speeds and bearings:
                    features['last_segment_speed'] = speeds[-1]
                    features['last_segment_bearing'] = bearings[-1] * 180 / math.pi % 360
                    if len(speeds) >= 2:
                        features['last_acceleration'] = speeds[-1] - speeds[-2]
                    else:
                        features['last_acceleration'] = 0

                    # Direction change in last segments
                    if len(bearings) >= 3:
                        last_change = abs(bearings[-1] - bearings[-2])
                        prev_change = abs(bearings[-2] - bearings[-3])
                        features['last_direction_change'] = last_change * 180 / math.pi
                        features['prev_direction_change'] = prev_change * 180 / math.pi
                        features['direction_change_trend'] = last_change - prev_change

                # NOVAS FEATURES: Distance from centroid to end point
                if len(lat_list) >= 2:
                    centroid_lat = np.mean(lat_list)
                    centroid_lon = np.mean(lon_list)
                    end_lat, end_lon = lat_list[-1], lon_list[-1]
                    centroid_to_end = self.haversine_distance(centroid_lat, centroid_lon, end_lat, end_lon)
                    features['centroid_to_end_distance'] = centroid_to_end
                    features['centroid_to_end_ratio'] = centroid_to_end / features.get('total_distance', 1) if features.get('total_distance', 0) > 0 else 0

                # NOVAS FEATURES: Start to end vector vs last segment vector angle
                if len(lat_list) >= 3 and bearings:
                    start_lat, start_lon = lat_list[0], lon_list[0]
                    end_lat, end_lon = lat_list[-1], lon_list[-1]
                    start_to_end_bearing = math.atan2(end_lon - start_lon, end_lat - start_lat) * 180 / math.pi % 360
                    last_segment_bearing = bearings[-1] * 180 / math.pi % 360
                    angle_diff = abs(start_to_end_bearing - last_segment_bearing)
                    angle_diff = min(angle_diff, 360 - angle_diff)  # smallest angle
                    features['start_end_vs_last_angle'] = angle_diff

                # NOVAS FEATURES: Direction consistency (how aligned are segments with overall direction)
                if len(bearings) >= 2:
                    overall_bearing = math.atan2(lon_list[-1] - lon_list[0], lat_list[-1] - lat_list[0])
                    bearing_diffs = [abs(b - overall_bearing) for b in bearings]
                    features['direction_consistency'] = np.mean([min(d, 2*math.pi - d) for d in bearing_diffs]) * 180 / math.pi

                # NOVAS FEATURES: More percentiles for various features
                if segment_distances:
                    seg = np.array(segment_distances)
                    features['seg_p5'] = float(np.percentile(seg, 5))
                    features['seg_p15'] = float(np.percentile(seg, 15))
                    features['seg_p85'] = float(np.percentile(seg, 85))
                    features['seg_p95'] = float(np.percentile(seg, 95))

                if bearings:
                    bearings_deg = np.array(bearings) * 180 / math.pi % 360
                    features['bearing_p5'] = float(np.percentile(bearings_deg, 5))
                    features['bearing_p15'] = float(np.percentile(bearings_deg, 15))
                    features['bearing_p85'] = float(np.percentile(bearings_deg, 85))
                    features['bearing_p95'] = float(np.percentile(bearings_deg, 95))

                if speeds:
                    features['speed_p5'] = float(np.percentile(speeds, 5))
                    features['speed_p15'] = float(np.percentile(speeds, 15))
                    features['speed_p85'] = float(np.percentile(speeds, 85))
                    features['speed_p95'] = float(np.percentile(speeds, 95))

                # NOVAS FEATURES: Statistical moments (skewness, kurtosis)
                if len(segment_distances) >= 3:
                    from scipy.stats import skew, kurtosis
                    features['seg_skewness'] = float(skew(segment_distances))
                    features['seg_kurtosis'] = float(kurtosis(segment_distances))

                if len(speeds) >= 3:
                    features['speed_skewness'] = float(skew(speeds))
                    features['speed_kurtosis'] = float(kurtosis(speeds))

                if len(bearings) >= 3:
                    bearings_deg = np.array(bearings) * 180 / math.pi % 360
                    features['bearing_skewness'] = float(skew(bearings_deg))
                    features['bearing_kurtosis'] = float(kurtosis(bearings_deg))

            features_list.append(features)

        features_df = pd.DataFrame(features_list)
        
        # Preencher valores nulos
        features_df = features_df.fillna(0)
        
        # Substituir infinitos por 0
        features_df = features_df.replace([np.inf, -np.inf], 0)
        
        self.logger.info(f"Features extraídas: {features_df.shape}")
        
        return features_df
    def latlon_to_local_xy(lat_ref, lon_ref, lat, lon):
        """Converte lat/lon para coordenadas locais (metros) usando aproximação equiretangular.

        lat_ref, lon_ref: referência (ponto de origem) em graus
        lat, lon: ponto(s) alvo em graus (escala: float ou array-like)
        Retorna: (dx, dy) em metros
        """
        R = 6371000.0
        # suportar arrays ou escalares
        lat_r = np.radians(lat)
        lon_r = np.radians(lon)
        lat0 = np.radians(lat_ref)
        lon0 = np.radians(lon_ref)

        # Se inputs forem arrays, operar vetorialmente
        dx = (lon_r - lon0) * np.cos((lat_r + lat0) / 2.0) * R
        dy = (lat_r - lat0) * R
        return dx, dy

    def local_xy_to_latlon(lat_ref, lon_ref, dx, dy):
        """Converte coordenadas locais (m) de volta para lat/lon (graus)."""
        R = 6371000.0
        lat0 = np.radians(lat_ref)
        lon0 = np.radians(lon_ref)

        lat = lat0 + dy / R
        lon = lon0 + dx / (R * np.cos((lat + lat0) / 2.0))
        return np.degrees(lat), np.degrees(lon)

    @staticmethod
    def prepare_features_for_training(train_features, test_features, target_cols=['dest_lat', 'dest_lon'], use_local_target=False, scaler_type='robust'):
        """Prepara features para treinamento.

        Se `use_local_target` for True, converte `dest_lat,dest_lon` em deslocamentos `dx,dy` (metros)
        relativos ao ponto inicial (`start_lat`,`start_lon`) quando disponível.
        Por padrão é False para manter targets em lat/lon (graus), compatível com `ModelTrainer`.
        """
        from sklearn.preprocessing import StandardScaler
        from sklearn.feature_selection import SelectKBest, f_regression
        from sklearn.decomposition import PCA

        # Separar features e target (remover trajectory_id se presente)
        feature_cols = [col for col in train_features.columns if col not in target_cols + ['trajectory_id']]

        # Trabalhar em cópias para evitar modificar os DataFrames originais
        df_train = train_features[feature_cols].copy()
        df_test = test_features[feature_cols].copy()

        # Detectar e remover features constantes (pouca variância) e quase-constantes
        const_cols = []
        low_var_cols = []
        n_samples = len(df_train)
        for col in df_train.columns:
            try:
                nunq = df_train[col].nunique(dropna=True)
                if nunq <= 1:
                    const_cols.append(col)
                    continue

                # proporção de valores únicos muito pequena -> quase-constante
                if nunq / max(1, n_samples) < 0.02:
                    low_var_cols.append(col)
                    continue

                # checar std para tipos numéricos
                if pd.api.types.is_numeric_dtype(df_train[col]):
                    if float(df_train[col].std()) <= 1e-8:
                        const_cols.append(col)
            except Exception:
                continue

        drop_cols = list(set(const_cols + low_var_cols))

        if drop_cols:
            try:
                from utils.logger import get_logger
                get_logger(__name__).info(f"Removendo {len(drop_cols)} features com baixa variancia: {drop_cols}")
            except Exception:
                pass

            df_train.drop(columns=drop_cols, inplace=True, errors='ignore')
            df_test.drop(columns=drop_cols, inplace=True, errors='ignore')

            # Atualizar lista de colunas
            feature_cols = [c for c in feature_cols if c not in drop_cols]

        X_train = df_train.values
        X_test = df_test.values

        # Preparar y_train
        if use_local_target and all(c in train_features.columns for c in ['start_lat', 'start_lon']):
            refs_lat = train_features['start_lat'].values
            refs_lon = train_features['start_lon'].values
            dest_lats = train_features[target_cols[0]].values
            dest_lons = train_features[target_cols[1]].values

            dx_list = []
            dy_list = []
            for lat0, lon0, dlat, dlon in zip(refs_lat, refs_lon, dest_lats, dest_lons):
                # lidar com NaNs
                if pd.isna(dlat) or pd.isna(dlon):
                    dx_list.append(0.0)
                    dy_list.append(0.0)
                    continue
                dx, dy = FeatureEngineer.latlon_to_local_xy(lat0, lon0, dlat, dlon)
                dx_list.append(float(dx))
                dy_list.append(float(dy))

            y_train = np.column_stack([np.array(dx_list), np.array(dy_list)])
        else:
            # fallback: usar lat/lon diretamente (graus) se não for possível aplicar conversão
            y_train = train_features[target_cols].values

        # Normalizar features (Robust por padrão)
        from sklearn.preprocessing import RobustScaler

        if scaler_type == 'robust':
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Feature Selection: Select top features based on correlation with target
        k_features = max(50, int(0.7 * X_train_scaled.shape[1]))  # Select at least 50, up to 70% of features
        selector = SelectKBest(score_func=f_regression, k=k_features)
        
        # Use the first target (latitude) for feature selection
        y_for_selection = y_train[:, 0] if len(y_train.shape) > 1 else y_train
        X_train_selected = selector.fit_transform(X_train_scaled, y_for_selection)
        X_test_selected = selector.transform(X_test_scaled)
        
        # Get selected feature names
        selected_mask = selector.get_support()
        selected_feature_names = [feature_cols[i] for i in range(len(feature_cols)) if selected_mask[i]]
        
        # Log feature selection
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.info(f"Feature selection: {X_train_scaled.shape[1]} -> {X_train_selected.shape[1]} features")
            logger.info(f"Top features: {selected_feature_names[:10]}...")  # Log first 10
        except:
            pass

        # Update X_train and X_test
        X_train_scaled = X_train_selected
        X_test_scaled = X_test_selected
        feature_cols = selected_feature_names

        # Log apenas se logger estiver disponível
        try:
            from utils.logger import get_logger
            logger = get_logger(__name__)
            logger.info(f"Features preparadas - Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")
        except:
            pass

        result = {
            'X_train': X_train_scaled,
            'y_train': y_train,
            'X_test': X_test_scaled,
            'feature_names': feature_cols,
            'scaler': scaler
        }

        if 'trajectory_id' in train_features.columns:
            result['groups'] = train_features['trajectory_id'].values

        return result