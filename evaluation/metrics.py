import numpy as np
from math import radians, sin, cos, sqrt, atan2

def haversine_distance_km(y_true, y_pred):
    """Calcula distância Haversine em km entre predições e valores reais"""
    
    R = 6371  # raio da Terra em km
    
    # Separar coordenadas
    lat_true = y_true[:, 0]
    lon_true = y_true[:, 1]
    lat_pred = y_pred[:, 0]
    lon_pred = y_pred[:, 1]
    
    # Converter para radianos
    lat_true, lon_true, lat_pred, lon_pred = map(
        np.radians, [lat_true, lon_true, lat_pred, lon_pred]
    )
    
    # Calcular diferenças
    dlat = lat_pred - lat_true
    dlon = lon_pred - lon_true
    
    # Fórmula Haversine
    a = np.sin(dlat/2)**2 + np.cos(lat_true) * np.cos(lat_pred) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), sqrt(1-a))
    
    return R * c

def calculate_metrics(y_true, y_pred):
    """Calcula várias métricas de avaliação"""
    
    # Distância Haversine
    distances = haversine_distance_km(y_true, y_pred)
    
    metrics = {
        'mean_distance_km': np.mean(distances),
        'median_distance_km': np.median(distances),
        'std_distance_km': np.std(distances),
        'min_distance_km': np.min(distances),
        'max_distance_km': np.max(distances),
        'p95_distance_km': np.percentile(distances, 95),
        'p99_distance_km': np.percentile(distances, 99)
    }
    
    return metrics