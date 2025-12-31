# features/clustering.py
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class DataClusterer:
    """Classe para clusterização de dados de trajetória"""

    def __init__(self, n_clusters=None, method='kmeans', random_state=42):
        """
        Inicializa o clusterer

        Args:
            n_clusters: Número de clusters (para K-means)
            method: Método de clusterização ('kmeans' ou 'dbscan')
            random_state: Seed para reprodutibilidade
        """
        self.n_clusters = n_clusters
        self.method = method
        self.random_state = random_state
        self.clusterer = None
        self.scaler = StandardScaler()
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

    def extract_clustering_features(self, df):
        """
        Extrai features relevantes para clusterização

        Args:
            df: DataFrame com dados de trajetória

        Returns:
            DataFrame com features para clusterização
        """
        features = []

        for idx, row in df.iterrows():
            feat_dict = {}

            # Features básicas de posição
            lat_list = row.get('path_lat_parsed', [])
            lon_list = row.get('path_lon_parsed', [])

            if len(lat_list) >= 2:
                feat_dict['start_lat'] = lat_list[0]
                feat_dict['start_lon'] = lon_list[0]
                feat_dict['end_lat'] = lat_list[-1]
                feat_dict['end_lon'] = lon_list[-1]

                # Centroide
                feat_dict['centroid_lat'] = np.mean(lat_list)
                feat_dict['centroid_lon'] = np.mean(lon_list)

                # Estatísticas básicas
                feat_dict['lat_range'] = np.max(lat_list) - np.min(lat_list)
                feat_dict['lon_range'] = np.max(lon_list) - np.min(lon_list)

                # Distância total (simples aproximação)
                total_dist = 0
                for i in range(1, len(lat_list)):
                    dist = self._haversine_distance(
                        lat_list[i-1], lon_list[i-1],
                        lat_list[i], lon_list[i]
                    )
                    total_dist += dist
                feat_dict['total_distance'] = total_dist

                # Distância em linha reta
                straight_dist = self._haversine_distance(
                    lat_list[0], lon_list[0],
                    lat_list[-1], lon_list[-1]
                )
                feat_dict['straight_distance'] = straight_dist

                # Straightness
                feat_dict['straightness'] = straight_dist / total_dist if total_dist > 0 else 1.0

                # Número de pontos
                feat_dict['num_points'] = len(lat_list)

                # Bearing inicial
                if len(lat_list) >= 2:
                    bearing = self._calculate_bearing(
                        lat_list[0], lon_list[0],
                        lat_list[-1], lon_list[-1]
                    )
                    feat_dict['bearing'] = bearing

            features.append(feat_dict)

        return pd.DataFrame(features)

    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calcula distância Haversine em metros"""
        from math import radians, sin, cos, sqrt, atan2
        R = 6371000  # Raio da Terra em metros

        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))

        return R * c

    def _calculate_bearing(self, lat1, lon1, lat2, lon2):
        """Calcula bearing entre dois pontos"""
        from math import radians, atan2, sin, cos
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

        dlon = lon2 - lon1
        y = sin(dlon) * cos(lat2)
        x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
        bearing = atan2(y, x)

        # Converter para graus e normalizar para 0-360
        return (bearing * 180 / 3.14159) % 360

    def fit_predict(self, df, eps=0.5, min_samples=5):
        """
        Aplica clusterização aos dados

        Args:
            df: DataFrame com dados de trajetória
            eps: Parâmetro eps para DBSCAN
            min_samples: Parâmetro min_samples para DBSCAN

        Returns:
            Array com labels dos clusters
        """
        # Extrair features para clusterização
        clustering_features = self.extract_clustering_features(df)

        # Remover linhas com NaN
        clustering_features = clustering_features.dropna()

        if len(clustering_features) == 0:
            self.logger.warning("Nenhuma feature válida para clusterização")
            return np.zeros(len(df))

        # Padronizar features
        X_scaled = self.scaler.fit_transform(clustering_features.values)

        if self.method == 'kmeans':
            if self.n_clusters is None:
                # Tentar diferentes números de clusters e escolher o melhor
                self.n_clusters = self._find_optimal_clusters(X_scaled, max_clusters=10)

            self.clusterer = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                n_init=10
            )

        elif self.method == 'dbscan':
            self.clusterer = DBSCAN(
                eps=eps,
                min_samples=min_samples
            )

        else:
            raise ValueError(f"Método de clusterização '{self.method}' não suportado")

        # Aplicar clusterização
        labels = self.clusterer.fit_predict(X_scaled)

        self.logger.info(f"Clusterização aplicada: {len(np.unique(labels))} clusters encontrados")
        self.logger.info(f"Distribuição por cluster: {pd.Series(labels).value_counts().to_dict()}")

        return labels

    def _find_optimal_clusters(self, X, max_clusters=10):
        """
        Encontra o número ótimo de clusters usando silhouette score
        """
        best_score = -1
        best_n = 2

        for n_clusters in range(2, min(max_clusters + 1, len(X))):
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(X)

            try:
                score = silhouette_score(X, labels)
                if score > best_score:
                    best_score = score
                    best_n = n_clusters
            except:
                continue

        self.logger.info(f"Número ótimo de clusters: {best_n} (silhouette: {best_score:.3f})")
        return best_n

    def get_largest_cluster_data(self, df, labels):
        """
        Retorna os dados do maior cluster

        Args:
            df: DataFrame original
            labels: Labels dos clusters

        Returns:
            DataFrame filtrado com apenas o maior cluster
        """
        # Encontrar o maior cluster (excluindo outliers se DBSCAN)
        unique_labels, counts = np.unique(labels, return_counts=True)

        # Para DBSCAN, ignorar cluster -1 (outliers)
        if self.method == 'dbscan':
            valid_labels = unique_labels[unique_labels != -1]
            valid_counts = counts[unique_labels != -1]
        else:
            valid_labels = unique_labels
            valid_counts = counts

        if len(valid_labels) == 0:
            self.logger.warning("Nenhum cluster válido encontrado")
            return df

        largest_cluster = valid_labels[np.argmax(valid_counts)]
        cluster_size = np.max(valid_counts)

        self.logger.info(f"Maior cluster: {largest_cluster} com {cluster_size} amostras")

        # Filtrar dados do maior cluster
        mask = labels == largest_cluster
        filtered_df = df[mask].copy()

        self.logger.info(f"Dados filtrados: {len(filtered_df)}/{len(df)} amostras mantidas")

        return filtered_df

    def plot_clusters(self, df, labels, save_path=None):
        """
        Plota os clusters (apenas para visualização 2D)
        """
        try:
            clustering_features = self.extract_clustering_features(df)

            if 'start_lat' in clustering_features.columns and 'start_lon' in clustering_features.columns:
                plt.figure(figsize=(12, 8))

                # Plotar pontos de início coloridos por cluster
                scatter = plt.scatter(
                    clustering_features['start_lon'],
                    clustering_features['start_lat'],
                    c=labels,
                    cmap='tab10',
                    alpha=0.6,
                    s=50
                )

                plt.xlabel('Longitude')
                plt.ylabel('Latitude')
                plt.title(f'Clusters de Trajetórias ({self.method.upper()})')
                plt.colorbar(scatter, label='Cluster')
                plt.grid(True, alpha=0.3)

                if save_path:
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    self.logger.info(f"Gráfico salvo em: {save_path}")

                plt.close()

        except Exception as e:
            self.logger.warning(f"Erro ao plotar clusters: {e}")