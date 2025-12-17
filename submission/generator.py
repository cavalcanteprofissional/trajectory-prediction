# submission/generator.py
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

class SubmissionGenerator:
    """Classe para geração de arquivos de submissão"""
    
    def __init__(self):
        self.logger = self._get_logger()
        self._load_config()
    
    def _get_logger(self):
        """Obtém logger de forma segura"""
        try:
            from utils.logger import get_logger
            return get_logger(__name__)
        except ImportError:
            import logging
            logging.basicConfig(level=logging.INFO)
            return logging.getLogger(__name__)
    
    def _load_config(self):
        """Carrega configurações"""
        try:
            from config.settings import config
            self.config = config
            self.submissions_dir = config.SUBMISSIONS_DIR
        except ImportError:
            # Fallback
            self.submissions_dir = Path(__file__).parent.parent / 'submissions'
            self.submissions_dir.mkdir(exist_ok=True)
    
    def generate_submission(self, test_ids, predictions, model_name, description=""):
        """Gera arquivo de submissão no formato correto"""
        
        # Criar DataFrame de submissão
        submission_df = pd.DataFrame({
            'trajectory_id': test_ids,
            'latitude_pred': predictions[:, 0],
            'longitude_pred': predictions[:, 1]
        })
        
        # Validar dados
        self._validate_submission(submission_df)
        
        # Gerar nome do arquivo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"submission_{model_name}_{timestamp}.csv"
        filepath = self.submissions_dir / filename
        
        # Salvar arquivo
        submission_df.to_csv(filepath, index=False)
        
        logger.info(f"Arquivo de submissão salvo: {filepath}")
        logger.info(f"Descrição: {description}")
        logger.info(f"Número de previsões: {len(submission_df)}")
        
        return filepath
    
    def _validate_submission(self, submission_df):
        """Valida o arquivo de submissão"""
        
        # Verificar valores nulos
        null_count = submission_df.isnull().sum().sum()
        if null_count > 0:
            logger.warning(f"Submissão contém {null_count} valores nulos")
        
        # Verificar ranges
        lat_min = submission_df['latitude_pred'].min()
        lat_max = submission_df['latitude_pred'].max()
        lon_min = submission_df['longitude_pred'].min()
        lon_max = submission_df['longitude_pred'].max()
        
        logger.info(f"Latitude range: [{lat_min:.6f}, {lat_max:.6f}]")
        logger.info(f"Longitude range: [{lon_min:.6f}, {lon_max:.6f}]")
        
        # Verificar IDs únicos
        unique_ids = submission_df['trajectory_id'].nunique()
        total_ids = len(submission_df)
        
        if unique_ids != total_ids:
            logger.warning(f"IDs não únicos: {unique_ids} únicos de {total_ids} totais")
    
    def submit_to_kaggle(self, submission_file, description=""):
        """Envia submissão para o Kaggle via API"""
        
        try:
            import kaggle
            from kaggle.api.kaggle_api_extended import KaggleApi
            
            api = KaggleApi()
            api.authenticate()
            
            logger.info(f"Enviando submissão para Kaggle: {submission_file}")
            
            # Enviar submissão
            api.competition_submit(
                file_name=str(submission_file),
                message=description,
                competition=config.KAGGLE_COMPETITION
            )
            
            logger.info("Submissão enviada com sucesso!")
            
            # Verificar status das submissões
            submissions = api.competition_submissions(config.KAGGLE_COMPETITION)
            latest_submission = submissions[0] if submissions else None
            
            if latest_submission:
                logger.info(f"Última submissão: {latest_submission}")
            
        except Exception as e:
            logger.error(f"Erro ao enviar para Kaggle: {e}")
            raise
    
    def generate_baseline_submission(self, test_ids, train_targets):
        """Gera uma submissão baseline (média dos destinos de treino)"""
        
        # Calcular média dos destinos de treino
        mean_lat = np.mean(train_targets[:, 0])
        mean_lon = np.mean(train_targets[:, 1])
        
        predictions = np.column_stack([
            np.full(len(test_ids), mean_lat),
            np.full(len(test_ids), mean_lon)
        ])
        
        filepath = self.generate_submission(
            test_ids, predictions, 
            model_name="baseline",
            description="Baseline - média dos destinos de treino"
        )
        
        logger.info(f"Baseline - Lat: {mean_lat:.6f}, Lon: {mean_lon:.6f}")
        
        return filepath