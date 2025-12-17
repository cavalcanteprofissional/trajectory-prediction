# training/metrics.py
"""
Módulo de métricas para avaliação de modelos de predição de trajetórias.
"""
import numpy as np
import math

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calcula a distância Haversine entre dois pontos na Terra.
    
    Args:
        lat1, lon1: Latitude e longitude do ponto 1 (graus decimais)
        lat2, lon2: Latitude e longitude do ponto 2 (graus decimais)
    
    Returns:
        Distância em quilômetros
    """
    # Raio da Terra em quilômetros
    R = 6371.0
    
    # Converter graus decimais para radianos
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)
    
    # Diferenças
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    # Fórmula Haversine
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    return R * c

def haversine_distance_vectorized(coords_true, coords_pred):
    """
    Versão vetorizada da distância Haversine.
    
    Args:
        coords_true: Array numpy shape (n_samples, 2) com [lat, lon] reais
        coords_pred: Array numpy shape (n_samples, 2) com [lat, lon] previstos
    
    Returns:
        Array com distâncias para cada par de coordenadas
    """
    # Raio da Terra em quilômetros
    R = 6371.0
    
    # Converter para radianos
    lat1 = np.radians(coords_true[:, 0])
    lon1 = np.radians(coords_true[:, 1])
    lat2 = np.radians(coords_pred[:, 0])
    lon2 = np.radians(coords_pred[:, 1])
    
    # Diferenças
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    # Fórmula Haversine vetorizada
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    return R * c

def mean_haversine_distance(coords_true, coords_pred):
    """
    Calcula a distância Haversine média.
    
    Args:
        coords_true: Array numpy shape (n_samples, 2)
        coords_pred: Array numpy shape (n_samples, 2)
    
    Returns:
        Distância média em km
    """
    distances = haversine_distance_vectorized(coords_true, coords_pred)
    return np.mean(distances)

def median_haversine_distance(coords_true, coords_pred):
    """
    Calcula a distância Haversine mediana.
    
    Args:
        coords_true: Array numpy shape (n_samples, 2)
        coords_pred: Array numpy shape (n_samples, 2)
    
    Returns:
        Distância mediana em km
    """
    distances = haversine_distance_vectorized(coords_true, coords_pred)
    return np.median(distances)

def haversine_distance_quantiles(coords_true, coords_pred, quantiles=[0.25, 0.5, 0.75, 0.9, 0.95]):
    """
    Calcula quantis da distribuição de distâncias Haversine.
    
    Args:
        coords_true: Array numpy shape (n_samples, 2)
        coords_pred: Array numpy shape (n_samples, 2)
        quantiles: Lista de quantis a calcular
    
    Returns:
        Dicionário com quantis
    """
    distances = haversine_distance_vectorized(coords_true, coords_pred)
    quantile_values = np.quantile(distances, quantiles)
    
    return dict(zip(quantiles, quantile_values))

def calculate_all_metrics(coords_true, coords_pred):
    """
    Calcula todas as métricas de erro.
    
    Args:
        coords_true: Array numpy shape (n_samples, 2)
        coords_pred: Array numpy shape (n_samples, 2)
    
    Returns:
        Dicionário com todas as métricas
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    # Erros de coordenada
    mse_lat = mean_squared_error(coords_true[:, 0], coords_pred[:, 0])
    mse_lon = mean_squared_error(coords_true[:, 1], coords_pred[:, 1])
    
    mae_lat = mean_absolute_error(coords_true[:, 0], coords_pred[:, 0])
    mae_lon = mean_absolute_error(coords_true[:, 1], coords_pred[:, 1])
    
    # R² score
    r2_lat = r2_score(coords_true[:, 0], coords_pred[:, 0])
    r2_lon = r2_score(coords_true[:, 1], coords_pred[:, 1])
    
    # Distância Haversine
    distances = haversine_distance_vectorized(coords_true, coords_pred)
    mean_hdist = np.mean(distances)
    median_hdist = np.median(distances)
    std_hdist = np.std(distances)
    
    # Quantis
    quantiles = haversine_distance_quantiles(coords_true, coords_pred)
    
    return {
        'mean_haversine_km': mean_hdist,
        'median_haversine_km': median_hdist,
        'std_haversine_km': std_hdist,
        'mse_latitude': mse_lat,
        'mse_longitude': mse_lon,
        'mae_latitude': mae_lat,
        'mae_longitude': mae_lon,
        'r2_latitude': r2_lat,
        'r2_longitude': r2_lon,
        'quantiles_km': quantiles,
        'n_samples': len(coords_true)
    }

# Teste do módulo
if __name__ == "__main__":
    print("🧪 Testando módulo de métricas...")
    
    # Teste com coordenadas conhecidas
    # São Paulo para Rio de Janeiro (~358 km)
    sp = np.array([[-23.550520, -46.633308]])
    rj = np.array([[-22.906847, -43.172897]])
    
    dist = mean_haversine_distance(sp, rj)
    print(f"São Paulo → Rio de Janeiro: {dist:.2f} km (esperado: ~358 km)")
    
    # Teste vetorizado
    coords_true = np.array([
        [-23.550520, -46.633308],  # São Paulo
        [-22.906847, -43.172897],  # Rio de Janeiro
        [-15.793889, -47.882778]   # Brasília
    ])
    
    coords_pred = np.array([
        [-23.550520, -46.633308],  # Mesmo lugar (distância 0)
        [-23.550520, -46.633308],  # São Paulo (distância ~358)
        [-23.550520, -46.633308]   # São Paulo (distância ~872)
    ])
    
    metrics = calculate_all_metrics(coords_true, coords_pred)
    
    print("\n📊 Métricas calculadas:")
    for key, value in metrics.items():
        if key != 'quantiles_km':
            print(f"  {key}: {value}")
    
    print("\n📈 Quantis de distância:")
    for q, v in metrics['quantiles_km'].items():
        print(f"  {q*100:.0f}%: {v:.2f} km")
    
    print("\n✅ Teste concluído!")