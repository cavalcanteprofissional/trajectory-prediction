# evaluation/visualization.py
"""
Módulo para visualização de trajetórias usando Folium

Este módulo cria mapas interativos com Folium para visualizar:
- Trajetórias de treino (azul)
- Trajetórias de teste (vermelho)
- Predições da última submissão (verde)

Uso:
    # Mapa básico com 50 trajetórias
    python evaluation/visualization.py

    # Mapa completo com todas as trajetórias
    python evaluation/visualization.py --full

    # Abrir mapa no navegador automaticamente
    python evaluation/visualization.py --open

    # Análise detalhada de uma trajetória específica
    python evaluation/visualization.py --trajectory 000_20081028003826

Exemplos de uso programático:
    from evaluation.visualization import TrajectoryVisualizer

    visualizer = TrajectoryVisualizer()

    # Criar mapa geral
    visualizer.create_trajectory_map(max_trajectories=100)

    # Abrir no navegador
    visualizer.open_map_in_browser()

    # Análise detalhada
    visualizer.plot_trajectory_comparison('trajectory_id')
"""
import os
import pandas as pd
import folium
from folium import plugins
import numpy as np
from pathlib import Path
import sys

# Adicionar diretório raiz ao path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from data.loader import DataLoader
from config.settings import config


class TrajectoryVisualizer:
    """Classe para visualização de trajetórias com Folium"""

    def __init__(self):
        self.loader = DataLoader()
        self.data_dir = config.DATA_DIR
        self.submissions_dir = ROOT_DIR / 'submissions'

    def load_latest_submission(self):
        """Carrega a última submissão gerada"""
        try:
            # Listar arquivos de submissão
            submission_files = list(self.submissions_dir.glob('submission_*.csv'))

            if not submission_files:
                print("⚠️  Nenhuma submissão encontrada")
                return None

            # Pegar a mais recente (ordenada por data no nome do arquivo)
            latest_submission = max(submission_files, key=lambda x: x.stat().st_mtime)
            print(f"📄 Carregando última submissão: {latest_submission.name}")

            submission_df = pd.read_csv(latest_submission)
            return submission_df

        except Exception as e:
            print(f"❌ Erro ao carregar submissão: {e}")
            return None

    def create_trajectory_map(self, max_trajectories=100, save_path=None):
        """
        Cria um mapa interativo com Folium mostrando trajetórias de treino, teste e predições

        Args:
            max_trajectories: Número máximo de trajetórias para plotar (para performance)
            save_path: Caminho para salvar o mapa HTML (opcional)
        """
        print("🗺️  Criando mapa de trajetórias...")

        # Carregar dados
        train_df, test_df = self.loader.load_data()
        submission_df = self.load_latest_submission()

        if train_df is None or test_df is None:
            print("❌ Erro ao carregar dados")
            return None

        # Calcular centro do mapa baseado nos dados de treino
        # Usar os pontos das trajetórias para calcular o centro
        all_lats = []
        all_lons = []
        for idx, row in train_df.head(100).iterrows():  # Amostra para performance
            try:
                lats = eval(row['path_lat'])
                lons = eval(row['path_lon'])
                all_lats.extend(lats)
                all_lons.extend(lons)
            except:
                continue
        
        if all_lats and all_lons:
            center_lat = np.mean(all_lats)
            center_lon = np.mean(all_lons)
        else:
            center_lat = 39.9  # Beijing area
            center_lon = 116.4

        # Criar mapa base
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=10,
            tiles='OpenStreetMap'
        )

        # Adicionar controle de camadas
        feature_group_train = folium.FeatureGroup(name='Trajetórias Treino', show=True)
        feature_group_test = folium.FeatureGroup(name='Trajetórias Teste', show=True)
        feature_group_predictions = folium.FeatureGroup(name='Predições', show=True)

        # Plotar trajetórias de TREINO (AZUL)
        print(f"📍 Plotando {min(max_trajectories if max_trajectories != float('inf') else len(train_df), len(train_df))} trajetórias de treino...")

        if max_trajectories == float('inf'):
            train_sample = train_df
        else:
            train_sample = train_df.head(max_trajectories)
        for idx, row in train_sample.iterrows():
            trajectory_id = row['trajectory_id']

            # Parse da trajetória
            try:
                lat_points = eval(row['path_lat'])
                lon_points = eval(row['path_lon'])

                # Criar linha da trajetória
                points = list(zip(lat_points, lon_points))

                if len(points) > 1:
                    folium.PolyLine(
                        points,
                        color='blue',
                        weight=2,
                        opacity=0.7,
                        popup=f'Treino: {trajectory_id}'
                    ).add_to(feature_group_train)

                    # Adicionar marcador no início (verde escuro)
                    folium.CircleMarker(
                        location=points[0],
                        radius=4,
                        color='darkblue',
                        fill=True,
                        fill_color='darkblue',
                        popup=f'Início Treino: {trajectory_id}'
                    ).add_to(feature_group_train)

                    # Adicionar marcador no fim (vermelho escuro)
                    folium.CircleMarker(
                        location=points[-1],
                        radius=4,
                        color='darkred',
                        fill=True,
                        fill_color='darkred',
                        popup=f'Fim Treino: {trajectory_id}'
                    ).add_to(feature_group_train)

            except Exception as e:
                print(f"⚠️  Erro ao processar trajetória treino {trajectory_id}: {e}")
                continue

        # Plotar trajetórias de TESTE (VERMELHO)
        print(f"📍 Plotando {min(max_trajectories if max_trajectories != float('inf') else len(test_df), len(test_df))} trajetórias de teste...")

        if max_trajectories == float('inf'):
            test_sample = test_df
        else:
            test_sample = test_df.head(max_trajectories)
        for idx, row in test_sample.iterrows():
            trajectory_id = row['trajectory_id']

            # Parse da trajetória
            try:
                lat_points = eval(row['path_lat'])
                lon_points = eval(row['path_lon'])

                # Criar linha da trajetória
                points = list(zip(lat_points, lon_points))

                if len(points) > 1:
                    folium.PolyLine(
                        points,
                        color='red',
                        weight=2,
                        opacity=0.7,
                        popup=f'Teste: {trajectory_id}'
                    ).add_to(feature_group_test)

                    # Adicionar marcador no início (rosa)
                    folium.CircleMarker(
                        location=points[0],
                        radius=4,
                        color='purple',
                        fill=True,
                        fill_color='purple',
                        popup=f'Início Teste: {trajectory_id}'
                    ).add_to(feature_group_test)

                    # Adicionar marcador no fim (laranja)
                    folium.CircleMarker(
                        location=points[-1],
                        radius=4,
                        color='orange',
                        fill=True,
                        fill_color='orange',
                        popup=f'Fim Teste: {trajectory_id}'
                    ).add_to(feature_group_test)

            except Exception as e:
                print(f"⚠️  Erro ao processar trajetória teste {trajectory_id}: {e}")
                continue

        # Plotar PREDIÇÕES da última submissão (VERDE)
        if submission_df is not None:
            print(f"📍 Plotando {len(submission_df)} predições...")

            for idx, row in submission_df.iterrows():
                trajectory_id = row['trajectory_id']
                pred_lat = row['latitude_pred']
                pred_lon = row['longitude_pred']

                # Adicionar marcador da predição
                folium.CircleMarker(
                    location=[pred_lat, pred_lon],
                    radius=6,
                    color='green',
                    fill=True,
                    fill_color='green',
                    popup=f'Predição: {trajectory_id}<br>Lat: {pred_lat:.6f}<br>Lon: {pred_lon:.6f}'
                ).add_to(feature_group_predictions)

        # Adicionar grupos de features ao mapa
        feature_group_train.add_to(m)
        feature_group_test.add_to(m)
        if submission_df is not None:
            feature_group_predictions.add_to(m)

        # Adicionar controle de camadas
        folium.LayerControl().add_to(m)

        # Adicionar minimap
        plugins.MiniMap().add_to(m)

        # Salvar mapa
        if save_path is None:
            save_path = ROOT_DIR / 'reports' / 'trajectory_map.html'

        save_path.parent.mkdir(exist_ok=True)
        m.save(str(save_path))

        print(f"✅ Mapa salvo em: {save_path}")
        print(f"📊 Estatísticas:")
        print(f"   • Trajetórias treino plotadas: {min(max_trajectories, len(train_df))}")
        print(f"   • Trajetórias teste plotadas: {min(max_trajectories, len(test_df))}")
        if submission_df is not None:
            print(f"   • Predições plotadas: {len(submission_df)}")

        return m

    def plot_trajectory_comparison(self, trajectory_id, save_path=None):
        """
        Plota uma comparação detalhada de uma trajetória específica

        Args:
            trajectory_id: ID da trajetória para comparar
            save_path: Caminho para salvar o mapa
        """
        print(f"🔍 Analisando trajetória: {trajectory_id}")

        # Carregar dados
        train_df, test_df = self.loader.load_data()
        submission_df = self.load_latest_submission()

        # Procurar trajetória no treino ou teste
        trajectory_data = None
        is_train = True

        # Verificar se está no treino
        train_row = train_df[train_df['trajectory_id'] == trajectory_id]
        if not train_row.empty:
            trajectory_data = train_row.iloc[0]
        else:
            # Verificar se está no teste
            test_row = test_df[test_df['trajectory_id'] == trajectory_id]
            if not test_row.empty:
                trajectory_data = test_row.iloc[0]
                is_train = False

        if trajectory_data is None:
            print(f"❌ Trajetória {trajectory_id} não encontrada")
            return None

        # Pegar predição se existir
        prediction = None
        if submission_df is not None:
            pred_row = submission_df[submission_df['trajectory_id'] == trajectory_id]
            if not pred_row.empty:
                prediction = pred_row.iloc[0]

        # Criar mapa focado na trajetória
        try:
            lat_points = eval(trajectory_data['path_lat'])
            lon_points = eval(trajectory_data['path_lon'])
            center_lat = np.mean(lat_points)
            center_lon = np.mean(lon_points)

            m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

            # Plotar trajetória real
            points = list(zip(lat_points, lon_points))
            color = 'blue' if is_train else 'red'
            label = 'Treino' if is_train else 'Teste'

            folium.PolyLine(
                points,
                color=color,
                weight=4,
                opacity=0.8,
                popup=f'Trajetória Real - {label}: {trajectory_id}'
            ).add_to(m)

            # Marcadores de início e fim
            folium.Marker(
                points[0],
                popup=f'Início: {trajectory_id}',
                icon=folium.Icon(color='green', icon='play')
            ).add_to(m)

            folium.Marker(
                points[-1],
                popup=f'Fim Real: {trajectory_id}',
                icon=folium.Icon(color='red', icon='stop')
            ).add_to(m)

            # Plotar predição se existir
            if prediction is not None:
                folium.Marker(
                    [prediction['latitude_pred'], prediction['longitude_pred']],
                    popup=f'Predição: {trajectory_id}<br>Erro estimado',
                    icon=folium.Icon(color='orange', icon='flag')
                ).add_to(m)

                # Calcular distância aproximada entre fim real e predição
                from geopy.distance import geodesic
                real_end = (points[-1][0], points[-1][1])
                pred_point = (prediction['latitude_pred'], prediction['longitude_pred'])
                distance = geodesic(real_end, pred_point).km

                folium.Popup(f'Distância: {distance:.2f} km').add_to(
                    folium.Marker(
                        [(points[-1][0] + prediction['latitude_pred'])/2,
                         (points[-1][1] + prediction['longitude_pred'])/2],
                        popup=f'Distância predição: {distance:.2f} km'
                    ).add_to(m)
                )

            # Salvar mapa
            if save_path is None:
                save_path = ROOT_DIR / 'reports' / f'trajectory_{trajectory_id}.html'

            save_path.parent.mkdir(exist_ok=True)
            m.save(str(save_path))

            print(f"✅ Mapa detalhado salvo em: {save_path}")
            return m

        except Exception as e:
            print(f"❌ Erro ao criar mapa detalhado: {e}")
            return None

    def open_map_in_browser(self, map_path=None):
        """
        Abre o mapa no navegador padrão
        
        Args:
            map_path: Caminho do arquivo HTML do mapa (opcional, usa o geral por padrão)
        """
        import webbrowser
        
        if map_path is None:
            map_path = ROOT_DIR / 'reports' / 'trajectory_map.html'
        
        if map_path.exists():
            webbrowser.open(str(map_path))
            print(f"🌐 Mapa aberto no navegador: {map_path}")
        else:
            print(f"❌ Arquivo não encontrado: {map_path}")

    def create_full_trajectory_map(self, save_path=None):
        """
        Cria um mapa com TODAS as trajetórias (mais lento, mas completo)
        
        Args:
            save_path: Caminho para salvar o mapa HTML
        """
        print("🗺️  Criando mapa completo de TODAS as trajetórias...")
        print("⚠️  Isso pode levar alguns minutos...")
        
        return self.create_trajectory_map(max_trajectories=float('inf'), save_path=save_path)


def main():
    """Função principal para demonstração"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualização de trajetórias com Folium')
    parser.add_argument('--full', action='store_true', help='Criar mapa com todas as trajetórias')
    parser.add_argument('--open', action='store_true', help='Abrir mapa no navegador')
    parser.add_argument('--trajectory', type=str, help='ID da trajetória para análise detalhada')
    
    args = parser.parse_args()
    
    visualizer = TrajectoryVisualizer()
    
    # Criar mapa geral
    if args.full:
        print("Criando mapa completo de todas as trajetórias...")
        visualizer.create_full_trajectory_map()
    else:
        print("Criando mapa de amostra (50 trajetórias)...")
        visualizer.create_trajectory_map(max_trajectories=50)
    
    # Abrir no navegador se solicitado
    if args.open:
        visualizer.open_map_in_browser()
    
    # Análise detalhada se ID fornecido
    if args.trajectory:
        print(f"Analisando trajetória específica: {args.trajectory}")
        visualizer.plot_trajectory_comparison(args.trajectory)


if __name__ == '__main__':
    main()