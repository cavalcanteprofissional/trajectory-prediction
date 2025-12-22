import pandas as pd
import numpy as np

def _ensure_list_like(x):
    if isinstance(x, (list, tuple)):
        return list(x)
    # If it's a string representation like '[1,2,3]' avoid eval for safety
    # fallback: return empty list
    return []


def clean_dataframe(df: pd.DataFrame, *, is_train: bool = True, min_points: int = 2, drop_missing_target: bool = True) -> pd.DataFrame:
    """Limpeza conservadora do DataFrame de trajetórias.

    Regras aplicadas:
    - Remove duplicatas por `trajectory_id` mantendo a primeira ocorrência.
    - Converte colunas numéricas básicas para tipos numéricos (coerce errors).
    - Remove coordenadas impossíveis (lat fora de [-90,90], lon fora de [-180,180]).
    - Garante que `path_lat_parsed` e `path_lon_parsed` sejam listas; drop se pontos < min_points para treino.
    - Para treino, opcionalmente remove linhas com target ausente (`dest_lat`/`dest_lon`) se `drop_missing_target`.

    Esta função é intencionalmente conservadora: evita transformações complexas que possam mascarar problemas.
    """
    if df is None:
        return df

    df = df.copy()

    # Drop duplicates by trajectory_id if present
    if 'trajectory_id' in df.columns:
        df = df.drop_duplicates(subset=['trajectory_id'])

    # Convert numeric columns if exist
    for col in ['start_lat', 'start_lon', 'end_lat', 'end_lon', 'dest_lat', 'dest_lon']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Remove impossible coordinates
    if 'start_lat' in df.columns:
        df = df[df['start_lat'].between(-90, 90) | df['start_lat'].isna()]
    if 'end_lat' in df.columns:
        df = df[df['end_lat'].between(-90, 90) | df['end_lat'].isna()]
    if 'start_lon' in df.columns:
        df = df[df['start_lon'].between(-180, 180) | df['start_lon'].isna()]
    if 'end_lon' in df.columns:
        df = df[df['end_lon'].between(-180, 180) | df['end_lon'].isna()]

    # Handle parsed path columns
    if 'path_lat_parsed' in df.columns and 'path_lon_parsed' in df.columns:
        df['path_lat_parsed'] = df['path_lat_parsed'].apply(_ensure_list_like)
        df['path_lon_parsed'] = df['path_lon_parsed'].apply(_ensure_list_like)
    else:
        # ensure columns exist to avoid KeyErrors downstream
        if 'path_lat_parsed' not in df.columns:
            df['path_lat_parsed'] = [[] for _ in range(len(df))]
        if 'path_lon_parsed' not in df.columns:
            df['path_lon_parsed'] = [[] for _ in range(len(df))]

    # Drop rows with too few points in train set (insufficient trajectory information)
    if is_train:
        mask_enough_points = df['path_lat_parsed'].apply(lambda x: len(x) >= min_points)
        if mask_enough_points.sum() < len(df):
            df = df[mask_enough_points]

    # Drop rows with missing targets for training
    if is_train and drop_missing_target:
        if 'dest_lat' in df.columns and 'dest_lon' in df.columns:
            df = df[df['dest_lat'].notna() & df['dest_lon'].notna()]

    # Reset index for safety
    df = df.reset_index(drop=True)

    return df


def clean_train_test(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
    """Aplica limpeza tanto no treino quanto no teste usando regras conservadoras."""
    train_clean = clean_dataframe(train_df, is_train=True)
    test_clean = clean_dataframe(test_df, is_train=False)
    return train_clean, test_clean
