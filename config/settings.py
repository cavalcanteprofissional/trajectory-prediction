# config/settings.py
import os
from pathlib import Path
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

class Config:
    """Configurações do projeto"""
    
    # Seeds para reprodutibilidade
    SEED = int(os.getenv('SEED', 42))
    
    # Credenciais Kaggle
    KAGGLE_USERNAME = os.getenv('KAGGLE_USERNAME')
    KAGGLE_KEY = os.getenv('KAGGLE_KEY')
    KAGGLE_COMPETITION = os.getenv('KAGGLE_COMPETITION', 'te-aprendizado-de-maquina')
    KAGGLE_DOWNLOAD_COMMAND = os.getenv('KAGGLE_DOWNLOAD_COMMAND', 
                                       f'kaggle competitions download -c {KAGGLE_COMPETITION}')
    
    # Configurações do modelo
    DEFAULT_MODELS = [
        'RandomForest',
        'XGBoost',
        'LightGBM',
        'GradientBoosting',
        'CatBoost',
        'LinearRegression'
    ]
    
    # Configurações de treinamento
    KFOLD_SPLITS = 5
    TEST_SIZE = 0.2
    
    # Diretórios do projeto
    @property
    def ROOT_DIR(self):
        return Path(__file__).parent.parent
    
    @property
    def DATA_DIR(self):
        # Corrigido: usar caminho absoluto relativo ao ROOT_DIR
        data_dir = Path(os.getenv('DATA_DIR', 'data'))
        if not data_dir.is_absolute():
            data_dir = self.ROOT_DIR / data_dir
        return data_dir.resolve()
    
    @property
    def MODELS_DIR(self):
        return self.ROOT_DIR / 'models'
    
    @property
    def SUBMISSIONS_DIR(self):
        return self.ROOT_DIR / 'submissions'
    
    @property
    def LOGS_DIR(self):
        return self.ROOT_DIR / 'logs'
    
    # Caminhos dos arquivos de dados
    @property
    def TRAIN_DATA_PATH(self):
        return self.DATA_DIR / 'train.csv'
    
    @property
    def TEST_DATA_PATH(self):
        return self.DATA_DIR / 'test.csv'
    
    @property
    def SAMPLE_SUBMISSION_PATH(self):
        return self.DATA_DIR / 'sample_submission.csv'
    
    def __init__(self):
        """Inicializa criando diretórios"""
        self._create_directories()
        self._setup_kaggle()
    
    def _create_directories(self):
        """Cria diretórios necessários"""
        directories = [
            self.DATA_DIR,
            self.MODELS_DIR,
            self.SUBMISSIONS_DIR,
            self.LOGS_DIR
        ]
        
        for dir_path in directories:
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"Diretorio criado/verificado: {dir_path}")
    
    def _setup_kaggle(self):
        """Configura credenciais do Kaggle"""
        if not self.KAGGLE_USERNAME or not self.KAGGLE_KEY:
            print("Credenciais do Kaggle nao configuradas no .env")
            return
        
        kaggle_dir = Path.home() / '.kaggle'
        kaggle_dir.mkdir(exist_ok=True)
        
        kaggle_json = {
            "username": self.KAGGLE_USERNAME,
            "key": self.KAGGLE_KEY
        }
        
        import json
        kaggle_json_path = kaggle_dir / 'kaggle.json'
        
        try:
            with open(kaggle_json_path, 'w') as f:
                json.dump(kaggle_json, f)
            
            # Configurar permissões
            os.chmod(kaggle_json_path, 0o600)
            
            print(f"Credenciais Kaggle salvas em: {kaggle_json_path}")
            
        except Exception as e:
            print(f"Erro ao salvar credenciais Kaggle: {e}")

# Instância global
config = Config()