# submission/generator.py
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import subprocess
import os
import logging

class SubmissionGenerator:
    """Classe para geração de arquivos de submissão"""
    
    def __init__(self):
        self.logger = self._get_logger()
        self._load_config()
    
    def _get_logger(self):
        """Obtém logger"""
        import logging
        return logging.getLogger(__name__)
    
    def _load_config(self):
        """Carrega configurações"""
        try:
            from config import config
            self.config = config
            self.submissions_dir = Path(config.SUBMISSIONS_DIR)
        except ImportError:
            # Fallback
            import os
            from dotenv import load_dotenv
            load_dotenv()
            
            self.submissions_dir = Path(__file__).parent.parent / 'submissions'
            self.config = type('Config', (), {
                'KAGGLE_COMPETITION': os.getenv('KAGGLE_COMPETITION', 'topicos-especiais-em-aprendizado-de-maquina-v2')
            })()
        
        # Garantir que o diretório existe
        self.submissions_dir.mkdir(exist_ok=True)
    
    def generate_submission(self, test_ids, predictions, model_name, description=""):
        """Gera arquivo de submissão no formato correto para o Kaggle"""
        
        self.logger.info(f"Gerando submissão para {len(test_ids)} trajetórias...")
        
        # Criar DataFrame de submissão com os nomes de colunas CORRETOS
        submission_df = pd.DataFrame({
            'trajectory_id': test_ids,
            'latitude_pred': predictions[:, 0],  # CORRETO: latitude_pred
            'longitude_pred': predictions[:, 1]   # CORRETO: longitude_pred
        })
        
        # Validar dados
        self._validate_submission(submission_df)
        
        # Gerar nome do arquivo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"submission_{model_name}_{timestamp}.csv"
        filepath = self.submissions_dir / filename
        
        # Salvar arquivo
        submission_df.to_csv(filepath, index=False)
        
        self.logger.info(f"✅ Arquivo de submissão salvo: {filepath}")
        self.logger.info(f"📝 Descrição: {description}")
        self.logger.info(f"📊 Número de previsões: {len(submission_df)}")
        
        # Mostrar preview
        self.logger.info(f"\n📋 Prévia da submissão:")
        self.logger.info(f"   Colunas: {list(submission_df.columns)}")
        self.logger.info(f"   Primeira linha:")
        self.logger.info(f"     ID: {submission_df.iloc[0]['trajectory_id']}")
        self.logger.info(f"     Latitude: {submission_df.iloc[0]['latitude_pred']:.6f}")
        self.logger.info(f"     Longitude: {submission_df.iloc[0]['longitude_pred']:.6f}")
        
        return str(filepath)
    
    def _validate_submission(self, submission_df):
        """Valida o arquivo de submissão"""
        
        # Verificar colunas obrigatórias
        required_columns = ['trajectory_id', 'latitude_pred', 'longitude_pred']
        missing_columns = [col for col in required_columns if col not in submission_df.columns]
        
        if missing_columns:
            raise ValueError(f"Colunas obrigatórias faltando: {missing_columns}")
        
        self.logger.info(f"✅ Colunas validadas: {list(submission_df.columns)}")
        
        # Verificar valores nulos
        null_count = submission_df.isnull().sum().sum()
        if null_count > 0:
            self.logger.warning(f"⚠️  Submissão contém {null_count} valores nulos")
            # Mostrar onde estão os nulos
            for col in submission_df.columns:
                col_nulls = submission_df[col].isnull().sum()
                if col_nulls > 0:
                    self.logger.warning(f"   - {col}: {col_nulls} valores nulos")
        else:
            self.logger.info("✅ Submissão validada: Sem valores nulos")
        
        # Verificar ranges das coordenadas
        lat_min = submission_df['latitude_pred'].min()
        lat_max = submission_df['latitude_pred'].max()
        lon_min = submission_df['longitude_pred'].min()
        lon_max = submission_df['longitude_pred'].max()
        
        self.logger.info(f"📍 Range Latitude: [{lat_min:.6f}, {lat_max:.6f}]")
        self.logger.info(f"📍 Range Longitude: [{lon_min:.6f}, {lon_max:.6f}]")
        
        # Verificar se as coordenadas estão em ranges razoáveis
        if lat_min < -90 or lat_max > 90:
            self.logger.warning(f"⚠️  Latitude fora do range normal [-90, 90]")
        
        if lon_min < -180 or lon_max > 180:
            self.logger.warning(f"⚠️  Longitude fora do range normal [-180, 180]")
        
        # Verificar IDs únicos
        unique_ids = submission_df['trajectory_id'].nunique()
        total_ids = len(submission_df)
        
        if unique_ids != total_ids:
            self.logger.warning(f"⚠️  IDs não únicos: {unique_ids} únicos de {total_ids} totais")
            # Encontrar duplicatas
            duplicates = submission_df['trajectory_id'].duplicated().sum()
            self.logger.warning(f"   - {duplicates} IDs duplicados encontrados")
        else:
            self.logger.info(f"✅ IDs únicos: {unique_ids} trajetórias")
    
    def submit_with_cli(self, submission_file, message=""):
        """Envia submissão para Kaggle usando CLI"""
        try:
            # Verificar se o arquivo existe
            if not os.path.exists(submission_file):
                self.logger.error(f"❌ Arquivo não encontrado: {submission_file}")
                return False
            
            # Verificar conteúdo do arquivo
            try:
                df_check = pd.read_csv(submission_file)
                required_cols = ['trajectory_id', 'latitude_pred', 'longitude_pred']
                missing_cols = [col for col in required_cols if col not in df_check.columns]
                if missing_cols:
                    self.logger.error(f"❌ Colunas faltando no arquivo: {missing_cols}")
                    return False
            except Exception as e:
                self.logger.error(f"❌ Erro ao ler arquivo {submission_file}: {e}")
                return False
            
            # Se não houver mensagem, criar uma padrão
            if not message:
                # Extrair nome do modelo do nome do arquivo
                filename = os.path.basename(submission_file)
                if '_' in filename:
                    parts = filename.split('_')
                    model_name = parts[1] if len(parts) > 1 else "Modelo"
                else:
                    model_name = "Modelo"
                
                message = f"Submissão automática - {model_name} - {self.config.KAGGLE_COMPETITION}"
            
            # Construir comando
            kaggle_cmd = [
                "kaggle", "competitions", "submit",
                "-c", self.config.KAGGLE_COMPETITION,
                "-f", str(submission_file),
                "-m", message
            ]
            
            self.logger.info(f"📤 Enviando submissão via CLI: {' '.join(kaggle_cmd)}")
            
            # Executar comando
            result = subprocess.run(kaggle_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info("✅ Submissão enviada com sucesso via CLI!")
                self.logger.info(f"📄 Arquivo: {submission_file}")
                self.logger.info(f"💬 Mensagem: {message}")
                
                # Mostrar saída
                if result.stdout:
                    self.logger.info(f"📋 Saída: {result.stdout.strip()}")
                
                return True
            else:
                self.logger.error(f"❌ Erro ao enviar via CLI:")
                if result.stderr:
                    self.logger.error(f"   {result.stderr.strip()}")
                return False
                
        except FileNotFoundError:
            self.logger.error("❌ Kaggle CLI não encontrado. Instale com: pip install kaggle")
            return False
        except Exception as e:
            self.logger.error(f"❌ Erro inesperado: {e}")
            return False
    
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
        
        self.logger.info(f"📊 Baseline - Lat: {mean_lat:.6f}, Lon: {mean_lon:.6f}")
        
        return filepath
    
    def get_latest_submission(self):
        """Obtém o arquivo de submissão mais recente"""
        if not self.submissions_dir.exists():
            return None
        
        csv_files = list(self.submissions_dir.glob("submission_*.csv"))
        if not csv_files:
            return None
        
        # Ordenar por data de modificação (mais recente primeiro)
        csv_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return str(csv_files[0])
    
    def validate_existing_submission(self, filepath):
        """Valida um arquivo de submissão existente"""
        try:
            df = pd.read_csv(filepath)
            self.logger.info(f"🔍 Validando arquivo: {filepath}")
            self.logger.info(f"   Linhas: {len(df)}")
            self.logger.info(f"   Colunas: {list(df.columns)}")
            
            # Verificar colunas obrigatórias
            required_columns = ['trajectory_id', 'latitude_pred', 'longitude_pred']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                self.logger.error(f"❌ Colunas faltando: {missing_columns}")
                return False
            
            self.logger.info("✅ Estrutura do arquivo válida")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Erro ao validar arquivo: {e}")
            return False

# Para uso direto do arquivo
if __name__ == "__main__":
    # Testar a classe
    logging.basicConfig(level=logging.INFO)
    
    gen = SubmissionGenerator()
    print(f"✅ Submission Generator inicializado")
    print(f"📁 Diretório: {gen.submissions_dir}")
    
    # Testar validação de arquivo existente
    latest = gen.get_latest_submission()
    if latest:
        print(f"\n🔍 Último arquivo: {latest}")
        gen.validate_existing_submission(latest)