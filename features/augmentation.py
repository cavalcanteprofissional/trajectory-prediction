import numpy as np
import pandas as pd
from .engineering import FeatureEngineer


def jitter_points(lat_list, lon_list, sigma_m=5.0):
    """Adiciona jitter gaussiano em metros convertendo para lat/lon aproximado."""
    if len(lat_list) == 0:
        return lat_list, lon_list

    lat0, lon0 = lat_list[0], lon_list[0]
    dx = np.random.normal(0, sigma_m, size=len(lat_list))
    dy = np.random.normal(0, sigma_m, size=len(lat_list))

    new_lats, new_lons = FeatureEngineer.local_xy_to_latlon(lat0, lon0, dx, dy)
    return list(new_lats), list(new_lons)


def drop_random_points(lat_list, lon_list, drop_prob=0.1):
    """Remove aleatoriamente pontos com probabilidade `drop_prob` (mantém sempre primeiro e último)."""
    if len(lat_list) <= 2:
        return lat_list, lon_list

    keep = [True]  # sempre manter primeiro
    for _ in range(1, len(lat_list)-1):
        keep.append(np.random.rand() > drop_prob)
    keep.append(True)  # sempre manter último

    new_lats = [lat for k, lat in zip(keep, lat_list) if k]
    new_lons = [lon for k, lon in zip(keep, lon_list) if k]
    return new_lats, new_lons


def rotate_trajectory(lat_list, lon_list, angle_degrees=5.0):
    """Rotaciona a trajetória em torno do seu centróide por `angle_degrees` (aproximação local)."""
    if len(lat_list) == 0:
        return lat_list, lon_list

    lat_c = np.mean(lat_list)
    lon_c = np.mean(lon_list)

    # converter para dx/dy em metros
    dx, dy = FeatureEngineer.latlon_to_local_xy(lat_c, lon_c, lat_list, lon_list)

    theta = np.radians(angle_degrees)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    dx_r = dx * cos_t - dy * sin_t
    dy_r = dx * sin_t + dy * cos_t

    new_lats, new_lons = FeatureEngineer.local_xy_to_latlon(lat_c, lon_c, dx_r, dy_r)
    return list(new_lats), list(new_lons)


def augment_dataframe(df, methods=None, p=0.3, seed=None):
    """Aplica augmentations ao DataFrame `df` com colunas `path_lat_parsed` e `path_lon_parsed`.

    methods: lista de métodos possíveis: 'jitter','drop','rotate'
    p: probabilidade de aplicar augmentação por linha
    """
    if seed is not None:
        np.random.seed(seed)

    if methods is None:
        methods = ['jitter', 'drop', 'rotate']

    rows = []
    for idx, row in df.iterrows():
        lat_list = row.get('path_lat_parsed', [])
        lon_list = row.get('path_lon_parsed', [])
        if len(lat_list) == 0:
            rows.append(row)
            continue

        if np.random.rand() < p:
            method = np.random.choice(methods)
            if method == 'jitter':
                nlats, nlons = jitter_points(lat_list, lon_list)
            elif method == 'drop':
                nlats, nlons = drop_random_points(lat_list, lon_list)
            elif method == 'rotate':
                nlats, nlons = rotate_trajectory(lat_list, lon_list, angle_degrees=np.random.uniform(-10,10))
            else:
                nlats, nlons = lat_list, lon_list

            new_row = row.copy()
            new_row['path_lat_parsed'] = nlats
            new_row['path_lon_parsed'] = nlons
            rows.append(new_row)
        else:
            rows.append(row)

    return pd.DataFrame(rows)
