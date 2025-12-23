# features/outlier_detection.py
"""
Módulo para detecção de outliers em dados de trajetórias geográficas
"""
import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class OutlierDetector:
    """Classe para detecção de outliers em trajetórias geográficas"""
    
    # Limites geográficos válidos (coordenadas válidas globalmente)
    VALID_LAT_RANGE = (-90, 90)
    VALID_LON_RANGE = (-180, 180)
    
    def __init__(self, 
                 max_jump_distance_km: float = 500.0,
                 max_speed_kmh: float = 800.0,
                 contamination: float = 0.05,
                 use_isolation_forest: bool = True,
                 use_geographic_bounds: bool = False,
                 max_outlier_percentage: float = 0.20):
        """
        Inicializa o detector de outliers
        
        Args:
            max_jump_distance_km: Distância máxima permitida entre pontos consecutivos (km)
                Padrão: 500.0 km (permite gaps GPS maiores, viagens de avião, etc.)
            max_speed_kmh: Velocidade máxima permitida (km/h) - assumindo 1 segundo entre pontos
                Padrão: 800.0 km/h (permite trens rápidos, aviões, etc.)
            contamination: Proporção esperada de outliers (para Isolation Forest)
                Padrão: 0.05 (5% - mais conservador)
            use_isolation_forest: Se True, usa Isolation Forest para detecção adicional
            use_geographic_bounds: Se True, usa limites geográficos baseados na distribuição
            max_outlier_percentage: Máximo de dados que podem ser marcados como outliers
                Padrão: 0.20 (20% - proteção contra remoção excessiva)
        """
        self.max_jump_distance_km = max_jump_distance_km
        self.max_speed_kmh = max_speed_kmh
        self.contamination = contamination
        self.use_isolation_forest = use_isolation_forest
        self.use_geographic_bounds = use_geographic_bounds
        self.max_outlier_percentage = max_outlier_percentage  # Proteção: não remover mais que X% dos dados
        self.logger = self._get_logger()
        self.isolation_forest = None
        self.scaler = None
        
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
    def haversine_distance_km(lat1, lon1, lat2, lon2):
        """Calcula distância Haversine entre dois pontos em km"""
        R = 6371  # raio da Terra em km
        
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        return R * c
    
    def detect_geographic_outliers(self, df: pd.DataFrame) -> pd.Series:
        """
        Detecta outliers geográficos (coordenadas fora de faixas válidas)
        
        Args:
            df: DataFrame com colunas path_lat_parsed e path_lon_parsed
            
        Returns:
            Series booleana indicando outliers (True = outlier)
        """
        outliers = pd.Series(False, index=df.index)
        
        if 'path_lat_parsed' not in df.columns or 'path_lon_parsed' not in df.columns:
            self.logger.warning("Colunas path_lat_parsed ou path_lon_parsed não encontradas")
            return outliers
        
        for idx, row in df.iterrows():
            lat_list = row.get('path_lat_parsed', [])
            lon_list = row.get('path_lon_parsed', [])
            
            # Marcar como outlier apenas se a trajetória estiver completamente vazia
            if not lat_list or not lon_list:
                outliers[idx] = True
                continue
            
            # Verificar limites geográficos válidos (coordenadas válidas globalmente)
            lat_array = np.array(lat_list)
            lon_array = np.array(lon_list)
            
            invalid_lat = (lat_array < self.VALID_LAT_RANGE[0]) | (lat_array > self.VALID_LAT_RANGE[1])
            invalid_lon = (lon_array < self.VALID_LON_RANGE[0]) | (lon_array > self.VALID_LON_RANGE[1])
            
            # Apenas marcar como outlier se houver coordenadas inválidas
            if invalid_lat.any() or invalid_lon.any():
                outliers[idx] = True
                continue
            
            # Se use_geographic_bounds estiver ativado, detectar outliers baseado na distribuição dos dados
            if self.use_geographic_bounds:
                # Calcular limites baseados nos dados (percentis 1 e 99)
                all_lats = []
                all_lons = []
                for _, r in df.iterrows():
                    lats = r.get('path_lat_parsed', [])
                    lons = r.get('path_lon_parsed', [])
                    if lats and lons:
                        all_lats.extend(lats)
                        all_lons.extend(lons)
                
                if all_lats and all_lons:
                    lat_p1, lat_p99 = np.percentile(all_lats, [1, 99])
                    lon_p1, lon_p99 = np.percentile(all_lons, [1, 99])
                    
                    # Expandir um pouco os limites (10% de margem)
                    lat_range = lat_p99 - lat_p1
                    lon_range = lon_p99 - lon_p1
                    lat_min = lat_p1 - 0.1 * lat_range
                    lat_max = lat_p99 + 0.1 * lat_range
                    lon_min = lon_p1 - 0.1 * lon_range
                    lon_max = lon_p99 + 0.1 * lon_range
                    
                    # Verificar se a trajetória está muito fora da distribuição
                    if (lat_array.min() < lat_min or lat_array.max() > lat_max or
                        lon_array.min() < lon_min or lon_array.max() > lon_max):
                        outliers[idx] = True
        
        return outliers
    
    def detect_trajectory_outliers(self, df: pd.DataFrame) -> pd.Series:
        """
        Detecta outliers em trajetórias (saltos grandes, velocidades impossíveis)
        
        Args:
            df: DataFrame com colunas path_lat_parsed e path_lon_parsed
            
        Returns:
            Series booleana indicando outliers (True = outlier)
        """
        outliers = pd.Series(False, index=df.index)
        
        if 'path_lat_parsed' not in df.columns or 'path_lon_parsed' not in df.columns:
            return outliers
        
        for idx, row in df.iterrows():
            lat_list = row.get('path_lat_parsed', [])
            lon_list = row.get('path_lon_parsed', [])
            
            if len(lat_list) < 2:
                continue
            
            # Calcular distâncias entre pontos consecutivos
            # Marcar como outlier apenas se houver MÚLTIPLOS saltos grandes
            # (um único salto pode ser um gap GPS válido)
            large_jumps = 0
            extreme_speeds = 0
            total_segments = len(lat_list) - 1
            
            for i in range(1, len(lat_list)):
                distance_km = self.haversine_distance_km(
                    lat_list[i-1], lon_list[i-1],
                    lat_list[i], lon_list[i]
                )
                
                # Verificar se o salto é muito grande
                if distance_km > self.max_jump_distance_km:
                    large_jumps += 1
                
                # Verificar velocidade (assumindo 1 segundo entre pontos)
                # Velocidade em km/h = distância_km / tempo_horas
                # Se assumirmos 1 segundo entre pontos: velocidade = distance_km * 3600
                speed_kmh = distance_km * 3600  # km/h
                
                if speed_kmh > self.max_speed_kmh:
                    extreme_speeds += 1
            
            # Marcar como outlier se mais de 10% dos segmentos forem problemáticos
            # (reduzido de 30% para ser mais agressivo na detecção)
            problematic_ratio = (large_jumps + extreme_speeds) / total_segments if total_segments > 0 else 0
            
            if problematic_ratio > 0.1:
                    outliers[idx] = True
        
        return outliers
    
    def detect_target_outliers(self, df: pd.DataFrame) -> pd.Series:
        """
        Detecta outliers no target (dest_lat, dest_lon)
        
        Args:
            df: DataFrame com colunas dest_lat e dest_lon
            
        Returns:
            Series booleana indicando outliers (True = outlier)
        """
        outliers = pd.Series(False, index=df.index)
        
        if 'dest_lat' not in df.columns or 'dest_lon' not in df.columns:
            return outliers
        
        # Verificar apenas limites geográficos válidos (coordenadas inválidas)
        # NÃO remover destinos que estão longe da trajetória, pois isso é válido
        # (o objetivo é prever destinos que podem estar longe do início)
        invalid_lat = (df['dest_lat'] < self.VALID_LAT_RANGE[0]) | (df['dest_lat'] > self.VALID_LAT_RANGE[1])
        invalid_lon = (df['dest_lon'] < self.VALID_LON_RANGE[0]) | (df['dest_lon'] > self.VALID_LON_RANGE[1])
        
        outliers = invalid_lat | invalid_lon
        
        # NÃO verificar distância do destino à trajetória, pois:
        # 1. O objetivo é prever destinos que podem estar longe
        # 2. Trajetórias podem ter destinos válidos muito distantes
        # 3. Isso estava removendo dados válidos importantes para o modelo
        
        return outliers
    
    def detect_feature_outliers_iqr(self, features_df: pd.DataFrame, 
                                     factor: float = 3.0) -> pd.Series:
        """
        Detecta outliers nas features usando método IQR (Interquartile Range)
        
        Args:
            features_df: DataFrame com features extraídas
            factor: Fator multiplicador para IQR (padrão 3.0 - mais conservador que 1.5)
            
        Returns:
            Series booleana indicando outliers (True = outlier)
        """
        outliers = pd.Series(False, index=features_df.index)
        
        # Contar quantas features marcam cada amostra como outlier
        outlier_counts = pd.Series(0, index=features_df.index)
        
        for col in features_df.columns:
            if features_df[col].dtype not in [np.float64, np.float32, np.int64, np.int32]:
                continue
            
            Q1 = features_df[col].quantile(0.25)
            Q3 = features_df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR == 0:
                continue
            
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            
            col_outliers = (features_df[col] < lower_bound) | (features_df[col] > upper_bound)
            outlier_counts += col_outliers.astype(int)
        
        # Marcar como outlier apenas se for outlier em múltiplas features
        # (evita marcar como outlier por uma única feature extrema)
        total_features = len([c for c in features_df.columns 
                             if features_df[c].dtype in [np.float64, np.float32, np.int64, np.int32]])
        
        if total_features > 0:
            # Marcar como outlier se for outlier em mais de 20% das features
            threshold = max(1, int(total_features * 0.2))
            outliers = outlier_counts >= threshold
        
        return outliers
    
    def detect_feature_outliers_isolation_forest(self, features_df: pd.DataFrame) -> pd.Series:
        """
        Detecta outliers nas features usando Isolation Forest
        
        Args:
            features_df: DataFrame com features extraídas
            
        Returns:
            Series booleana indicando outliers (True = outlier)
        """
        # Selecionar apenas colunas numéricas
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return pd.Series(False, index=features_df.index)
        
        # Verificar se há dados suficientes
        if len(features_df) == 0:
            self.logger.warning("DataFrame vazio, não é possível detectar outliers")
            return pd.Series(False, index=features_df.index)
        
        X = features_df[numeric_cols].fillna(0).values
        
        # Verificar se há pelo menos 1 amostra
        if X.shape[0] < 1:
            self.logger.warning("Menos de 1 amostra, não é possível detectar outliers")
            return pd.Series(False, index=features_df.index)
        
        # Normalizar
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Treinar Isolation Forest
        self.isolation_forest = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=100
        )
        
        outlier_labels = self.isolation_forest.fit_predict(X_scaled)
        
        # -1 indica outlier, 1 indica normal
        outliers = pd.Series(outlier_labels == -1, index=features_df.index)
        
        return outliers
    
    def detect_all_outliers(self, 
                           df: pd.DataFrame,
                           features_df: Optional[pd.DataFrame] = None,
                           use_geographic: bool = True,
                           use_trajectory: bool = True,
                           use_target: bool = True,
                           use_features: bool = True) -> Dict[str, pd.Series]:
        """
        Detecta todos os tipos de outliers
        
        Args:
            df: DataFrame original com trajetórias
            features_df: DataFrame com features extraídas (opcional)
            use_geographic: Se True, detecta outliers geográficos
            use_trajectory: Se True, detecta outliers em trajetórias
            use_target: Se True, detecta outliers no target
            use_features: Se True, detecta outliers nas features
            
        Returns:
            Dicionário com diferentes tipos de outliers detectados
        """
        results = {}
        
        if use_geographic:
            self.logger.info("Detectando outliers geográficos...")
            results['geographic'] = self.detect_geographic_outliers(df)
            self.logger.info(f"   Encontrados: {results['geographic'].sum()} outliers geográficos")
        
        if use_trajectory:
            self.logger.info("Detectando outliers em trajetórias...")
            results['trajectory'] = self.detect_trajectory_outliers(df)
            self.logger.info(f"   Encontrados: {results['trajectory'].sum()} outliers em trajetórias")
        
        if use_target and 'dest_lat' in df.columns:
            self.logger.info("Detectando outliers no target...")
            results['target'] = self.detect_target_outliers(df)
            self.logger.info(f"   Encontrados: {results['target'].sum()} outliers no target")
        
        if use_features and features_df is not None:
            self.logger.info("Detectando outliers nas features (IQR)...")
            results['features_iqr'] = self.detect_feature_outliers_iqr(features_df)
            self.logger.info(f"   Encontrados: {results['features_iqr'].sum()} outliers nas features (IQR)")
            
            if self.use_isolation_forest:
                self.logger.info("Detectando outliers nas features (Isolation Forest)...")
                results['features_isolation'] = self.detect_feature_outliers_isolation_forest(features_df)
                self.logger.info(f"   Encontrados: {results['features_isolation'].sum()} outliers nas features (Isolation Forest)")
        
        return results
    
    def get_combined_outliers(self, outlier_dict: Dict[str, pd.Series], 
                              method: str = 'any') -> pd.Series:
        """
        Combina diferentes tipos de outliers
        
        Args:
            outlier_dict: Dicionário com diferentes tipos de outliers
            method: 'any' (qualquer outlier) ou 'all' (todos os tipos)
            
        Returns:
            Series booleana com outliers combinados
        """
        if not outlier_dict:
            return pd.Series(False, index=[])
        
        # Obter índice de referência
        first_series = next(iter(outlier_dict.values()))
        combined = pd.Series(False, index=first_series.index)
        
        if method == 'any':
            # Qualquer outlier em qualquer categoria
            for outlier_series in outlier_dict.values():
                combined = combined | outlier_series
        elif method == 'all':
            # Outlier em todas as categorias
            for outlier_series in outlier_dict.values():
                combined = combined & outlier_series
        else:
            raise ValueError(f"Método desconhecido: {method}. Use 'any' ou 'all'")
        
        # Proteção: não marcar mais que max_outlier_percentage como outliers
        n_outliers = combined.sum()
        total = len(combined)
        if total > 0 and n_outliers / total > self.max_outlier_percentage:
            self.logger.warning(
                f"Proteção ativada: {n_outliers}/{total} ({n_outliers/total*100:.1f}%) marcados como outliers, "
                f"limitando a {self.max_outlier_percentage*100:.1f}%"
            )
            # Manter apenas os outliers mais extremos (baseado na quantidade de tipos de outliers)
            outlier_scores = pd.Series(0, index=combined.index)
            for outlier_series in outlier_dict.values():
                outlier_scores += outlier_series.astype(int)
            
            # Priorizar outliers geográficos e de trajetória (mais confiáveis)
            # Dar peso maior para esses tipos
            if 'geographic' in outlier_dict:
                outlier_scores += outlier_dict['geographic'].astype(int) * 3
            if 'trajectory' in outlier_dict:
                outlier_scores += outlier_dict['trajectory'].astype(int) * 2
            
            # Ordenar por score e manter apenas os top outliers
            max_outliers = int(total * self.max_outlier_percentage)
            top_outliers = outlier_scores.nlargest(max_outliers).index
            combined = pd.Series(False, index=combined.index)
            combined.loc[top_outliers] = True
        
        return combined
    
    def remove_outliers(self, 
                       df: pd.DataFrame,
                       outlier_mask: pd.Series,
                       inplace: bool = False) -> pd.DataFrame:
        """
        Remove outliers do DataFrame
        
        Args:
            df: DataFrame original
            outlier_mask: Series booleana indicando outliers (True = remover)
            inplace: Se True, modifica o DataFrame original
            
        Returns:
            DataFrame sem outliers
        """
        # Garantir que os índices estão alinhados
        common_indices = df.index.intersection(outlier_mask.index)
        if len(common_indices) != len(df.index):
            self.logger.warning(
                f"Índices não totalmente alinhados: df tem {len(df.index)} índices, "
                f"mask tem {len(outlier_mask.index)} índices, "
                f"comum: {len(common_indices)}"
            )
            # Filtrar apenas índices comuns
            df_filtered = df.loc[common_indices]
            outlier_mask_filtered = outlier_mask.loc[common_indices]
        else:
            df_filtered = df
            outlier_mask_filtered = outlier_mask
        
        if inplace:
            df_filtered.drop(df_filtered.index[outlier_mask_filtered], inplace=True)
            return df_filtered
        else:
            return df_filtered[~outlier_mask_filtered].copy()
    
    def get_outlier_report(self, 
                          df: pd.DataFrame,
                          outlier_dict: Dict[str, pd.Series],
                          combined_outliers: pd.Series) -> Dict:
        """
        Gera relatório de outliers
        
        Args:
            df: DataFrame original
            outlier_dict: Dicionário com diferentes tipos de outliers
            combined_outliers: Series com outliers combinados
            
        Returns:
            Dicionário com estatísticas de outliers
        """
        total_samples = len(df)
        n_outliers = combined_outliers.sum()
        pct_outliers = (n_outliers / total_samples * 100) if total_samples > 0 else 0
        
        report = {
            'total_samples': total_samples,
            'total_outliers': n_outliers,
            'percentage_outliers': pct_outliers,
            'clean_samples': total_samples - n_outliers,
            'by_type': {}
        }
        
        for outlier_type, outlier_series in outlier_dict.items():
            n_type = outlier_series.sum()
            pct_type = (n_type / total_samples * 100) if total_samples > 0 else 0
            report['by_type'][outlier_type] = {
                'count': n_type,
                'percentage': pct_type
            }
        
        return report

