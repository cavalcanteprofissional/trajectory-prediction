import os
from pathlib import Path
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

# Configurações do projeto
class Config:
    # Seeds para reprodutibilidade
    SEED = int(os.getenv('SEED', 42))
    
    # Credenciais Kaggle
    KAGGLE_USERNAME = os.getenv('KAGGLE_USERNAME')
    KAGGLE_KEY = os.getenv('KAGGLE_KEY')
    KAGGLE_COMPETITION = os.getenv('KAGGLE_COMPETITION', 'te-aprendizado-de-maquina')
    
    # Diretórios (usando caminhos absolutos)
    @property
    def ROOT_DIR(self):
        return Path(__file__).parent.parent
    
    @property
    def DATA_DIR(self):
        return self.ROOT_DIR / 'data'
    
    @property
    def MODELS_DIR(self):
        return self.ROOT_DIR / 'models'
    
    @property
    def SUBMISSIONS_DIR(self):
        return self.ROOT_DIR / 'submissions'
    
    @property
    def LOGS_DIR(self):
        return self.ROOT_DIR / 'logs'
    
    @property
    def TRAIN_DATA_PATH(self):
        return self.DATA_DIR / 'train.csv'
    
    @property
    def TEST_DATA_PATH(self):
        return self.DATA_DIR / 'test.csv'
    
    # Configurações de treinamento
    KFOLD_SPLITS = 5
    TEST_SIZE = 0.2
    
    # Configurações do modelo
    DEFAULT_MODELS = [
        'RandomForest',
        'XGBoost',
        'LightGBM',
        'GradientBoosting',
        'CatBoost',
        'LinearRegression'
    ]
    
    def __init__(self):
        """Inicializar criando diretórios"""
        self._create_directories()
    
    def _create_directories(self):
        """Criar diretórios necessários"""
        for dir_path in [
            self.DATA_DIR,
            self.MODELS_DIR,
            self.SUBMISSIONS_DIR,
            self.LOGS_DIR
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

# Instância de configuração global
config = Config()