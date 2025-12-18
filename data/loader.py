# data/loader.py
import pandas as pd
import numpy as np
import ast
from pathlib import Path
import sys

# Adicionar diretório raiz ao path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config.settings import config

class DataLoader:
    """Classe para carregamento e pré-processamento de dados"""
    
    def __init__(self):
        self.train_data = None
        self.test_data = None
        self.sample_submission = None
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
    
    def ensure_data_exists(self, download_if_missing=True):
        """
        Garante que os dados existem, baixando se necessário
        
        Args:
            download_if_missing: Se True, baixa dados se não existirem
        
        Returns:
            bool: True se os dados existem/foram baixados
        """
        required_files = [
            config.TRAIN_DATA_PATH,
            config.TEST_DATA_PATH
        ]
        
        all_exist = all(f.exists() for f in required_files)
        
        if all_exist:
            self.logger.info("✅ Todos os arquivos de dados existem")
            return True
        
        if download_if_missing:
            self.logger.info("📥 Alguns arquivos estão faltando, baixando...")
            
            try:
                from .downloader import download_data
                success = download_data(force=False)
                
                if success:
                    # Verificar novamente após download
                    all_exist = all(f.exists() for f in required_files)
                    if all_exist:
                        self.logger.info("✅ Dados baixados com sucesso")
                        return True
                    else:
                        self.logger.error("❌ Download concluído mas arquivos ainda faltando")
                        return False
                else:
                    self.logger.error("❌ Falha no download dos dados")
                    return False
                    
            except Exception as e:
                self.logger.error(f"❌ Erro ao baixar dados: {e}")
                return False
        else:
            self.logger.warning("⚠ Arquivos de dados não encontrados")
            return False
    
    @staticmethod
    def parse_path_string(path_str):
        """Converte string de lista para lista de floats"""
        try:
            if isinstance(path_str, str):
                # Remover possíveis espaços extras
                path_str = path_str.strip()
                return ast.literal_eval(path_str)
            elif isinstance(path_str, list):
                return path_str
            elif pd.isna(path_str):
                return []
            else:
                return []
        except (ValueError, SyntaxError) as e:
            print(f"⚠ Erro ao parsear string: {path_str[:50]}... - Erro: {e}")
            return []
    
    def load_data(self, use_sample_if_missing=True):
        """
        Carrega os dados de treino e teste
        
        Args:
            use_sample_if_missing: Se True, cria dados de exemplo se reais não existirem
        
        Returns:
            tuple: (train_data, test_data)
        """
        # Garantir que os dados existem
        if not self.ensure_data_exists(download_if_missing=True):
            if use_sample_if_missing:
                self.logger.warning("⚠ Usando dados de exemplo como fallback")
                return self.create_sample_data()
            else:
                raise FileNotFoundError(
                    f"Arquivos de dados não encontrados em {config.DATA_DIR}. "
                    f"Execute primeiro: kaggle competitions download -c {config.KAGGLE_COMPETITION}"
                )
        
        # Carregar dados reais
        try:
            self.logger.info(f"📖 Lendo dados de {config.DATA_DIR}")
            
            # Carregar train.csv
            self.train_data = pd.read_csv(config.TRAIN_DATA_PATH)
            self.logger.info(f"✅ Train carregado: {len(self.train_data)} linhas, {len(self.train_data.columns)} colunas")
            
            # Carregar test.csv
            self.test_data = pd.read_csv(config.TEST_DATA_PATH)
            self.logger.info(f"✅ Test carregado: {len(self.test_data)} linhas, {len(self.test_data.columns)} colunas")
            
            # Carregar sample_submission se existir
            if config.SAMPLE_SUBMISSION_PATH.exists():
                self.sample_submission = pd.read_csv(config.SAMPLE_SUBMISSION_PATH)
                self.logger.info(f"✅ Sample submission carregado: {len(self.sample_submission)} linhas")
            
            # Parse das trajetórias
            self._parse_trajectories()
            
            # Validar dados
            self._validate_data()
            
            return self.train_data, self.test_data
            
        except Exception as e:
            self.logger.error(f"❌ Erro ao carregar dados: {e}")
            
            if use_sample_if_missing:
                self.logger.info("📝 Criando dados de exemplo...")
                return self.create_sample_data()
            else:
                raise
    
    def _parse_trajectories(self):
        """Parse das colunas de trajetória"""
        # Verificar colunas existentes
        if 'path_lat' in self.train_data.columns:
            self.train_data['path_lat_parsed'] = self.train_data['path_lat'].apply(self.parse_path_string)
            self.train_data['path_lon_parsed'] = self.train_data['path_lon'].apply(self.parse_path_string)
        
        if 'path_lat' in self.test_data.columns:
            self.test_data['path_lat_parsed'] = self.test_data['path_lat'].apply(self.parse_path_string)
            self.test_data['path_lon_parsed'] = self.test_data['path_lon'].apply(self.parse_path_string)
    
    def _validate_data(self):
        """Valida os dados carregados"""
        self.logger.info("🔍 Validando dados...")
        
        if self.train_data is not None:
            train_nulls = self.train_data.isnull().sum().sum()
            self.logger.info(f"   Train - Valores nulos: {train_nulls}")
            
            if 'path_lat_parsed' in self.train_data.columns:
                train_empty = (self.train_data['path_lat_parsed'].str.len() == 0).sum()
                self.logger.info(f"   Train - Trajetórias vazias: {train_empty}")
        
        if self.test_data is not None:
            test_nulls = self.test_data.isnull().sum().sum()
            self.logger.info(f"   Test - Valores nulos: {test_nulls}")
            
            if 'path_lat_parsed' in self.test_data.columns:
                test_empty = (self.test_data['path_lat_parsed'].str.len() == 0).sum()
                self.logger.info(f"   Test - Trajetórias vazias: {test_empty}")
    
    def get_data_summary(self):
        """Retorna resumo dos dados"""
        if self.train_data is None or self.test_data is None:
            raise ValueError("Dados não carregados. Execute load_data() primeiro.")
        
        summary = {
            'train_samples': len(self.train_data),
            'test_samples': len(self.test_data),
            'train_columns': list(self.train_data.columns),
            'test_columns': list(self.test_data.columns),
            'train_memory_mb': self.train_data.memory_usage(deep=True).sum() / 1024**2,
            'test_memory_mb': self.test_data.memory_usage(deep=True).sum() / 1024**2,
            'has_target': 'dest_lat' in self.train_data.columns and 'dest_lon' in self.train_data.columns
        }
        
        return summary
    
    def create_sample_data(self, n_train=1000, n_test=200):
        """Cria dados de exemplo (fallback)"""
        self.logger.info(f"📝 Criando dados de exemplo: {n_train} train, {n_test} test")
        
        np.random.seed(config.SEED)
        
        # Função para criar trajetória realista
        def create_trajectory(n_points=None):
            if n_points is None:
                n_points = np.random.randint(5, 20)
            
            start_lat = 40.0 + np.random.uniform(-2, 2)
            start_lon = -73.0 + np.random.uniform(-2, 2)
            
            # Criar pontos com direção
            lat_points = [start_lat]
            lon_points = [start_lon]
            
            for _ in range(1, n_points):
                lat_points.append(lat_points[-1] + np.random.uniform(-0.01, 0.02))
                lon_points.append(lon_points[-1] + np.random.uniform(-0.01, 0.02))
            
            return lat_points, lon_points
        
        # Dados de treino
        train_records = []
        for i in range(n_train):
            n_points = np.random.randint(5, 20)
            lat_points, lon_points = create_trajectory(n_points)
            
            # Destino baseado na direção
            if len(lat_points) > 1:
                lat_trend = lat_points[-1] - lat_points[0]
                lon_trend = lon_points[-1] - lon_points[0]
                dest_lat = lat_points[-1] + lat_trend * np.random.uniform(1, 3)
                dest_lon = lon_points[-1] + lon_trend * np.random.uniform(1, 3)
            else:
                dest_lat = 40.0
                dest_lon = -73.0
            
            train_records.append({
                'trajectory_id': f'train_{i:04d}',
                'path_lat': str(lat_points),
                'path_lon': str(lon_points),
                'dest_lat': dest_lat,
                'dest_lon': dest_lon
            })
        
        self.train_data = pd.DataFrame(train_records)
        
        # Dados de teste
        test_records = []
        for i in range(n_test):
            n_points = np.random.randint(5, 20)
            lat_points, lon_points = create_trajectory(n_points)
            
            test_records.append({
                'trajectory_id': f'test_{i:04d}',
                'path_lat': str(lat_points),
                'path_lon': str(lon_points)
            })
        
        self.test_data = pd.DataFrame(test_records)
        
        # Parse das trajetórias
        self._parse_trajectories()
        
        self.logger.info(f"✅ Dados de exemplo criados")
        
        return self.train_data, self.test_data
    
    def save_sample_data(self):
        """Salva os dados de exemplo para referência"""
        if self.train_data is not None:
            sample_train_path = config.DATA_DIR / 'train_sample.csv'
            self.train_data.to_csv(sample_train_path, index=False)
            self.logger.info(f"💾 Train sample salvo: {sample_train_path}")
        
        if self.test_data is not None:
            sample_test_path = config.DATA_DIR / 'test_sample.csv'
            self.test_data.to_csv(sample_test_path, index=False)
            self.logger.info(f"💾 Test sample salvo: {sample_test_path}")

# Função de conveniência
def load_trajectory_data():
    """Carrega dados de forma simples"""
    loader = DataLoader()
    return loader.load_data()