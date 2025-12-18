# data/downloader.py
import os
import subprocess
import zipfile
import shutil
from pathlib import Path
import sys

# Adicionar diretório raiz ao path para imports
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config.settings import config

class KaggleDownloader:
    """Classe para download de dados do Kaggle usando CLI"""
    
    def __init__(self):
        self.logger = self._get_logger()
        self._verify_kaggle_installation()
    
    def _get_logger(self):
        """Obtém logger de forma segura"""
        try:
            from utils.logger import get_logger
            return get_logger(__name__)
        except ImportError:
            import logging
            logging.basicConfig(level=logging.INFO)
            return logging.getLogger(__name__)
    
    def _verify_kaggle_installation(self):
        """Verifica se o Kaggle CLI está instalado"""
        try:
            result = subprocess.run(
                ['kaggle', '--version'],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                self.logger.info(f"Kaggle CLI instalado: {result.stdout.strip()}")
                return True
            else:
                self.logger.warning("Kaggle CLI não encontrado ou com erro")
                return False
        except FileNotFoundError:
            self.logger.error("Kaggle CLI não está instalado!")
            self.logger.info("Para instalar: pip install kaggle")
            return False
    
    def download_competition_data(self, force_download=False):
        """
        Baixa dados da competição usando comando Kaggle CLI
        
        Args:
            force_download: Se True, baixa mesmo se já existir
        
        Returns:
            bool: True se o download foi bem sucedido
        """
        # Verificar se os arquivos CSV já existem
        required_files = ['train.csv', 'test.csv']
        csv_files_exist = all((config.DATA_DIR / f).exists() for f in required_files)
        
        if csv_files_exist and not force_download:
            self.logger.info(f"Arquivos CSV já existem em {config.DATA_DIR}")
            return True
        
        # Criar diretório de dados se não existir
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        # Mudar para o diretório de dados
        original_cwd = os.getcwd()
        os.chdir(config.DATA_DIR)
        
        try:
            self.logger.info(f"Baixando dados da competição: {config.KAGGLE_COMPETITION}")
            self.logger.info(f"Destino: {config.DATA_DIR}")
            
            # Executar comando Kaggle
            command = config.KAGGLE_DOWNLOAD_COMMAND
            self.logger.info(f"Executando: {command}")
            
            result = subprocess.run(
                command.split(),
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self.logger.info("Download concluído com sucesso!")
                
                # Extrair arquivos ZIP
                self._extract_zip_files()
                
                # Verificar arquivos baixados
                success = self._verify_downloaded_files()
                
                # Se ainda não tem os arquivos CSV, tentar extrair do ZIP específico
                if not success:
                    self.logger.info("Tentando extrair do arquivo ZIP específico...")
                    zip_file = config.DATA_DIR / f"{config.KAGGLE_COMPETITION}.zip"
                    if zip_file.exists():
                        self._extract_specific_zip(zip_file)
                        success = self._verify_downloaded_files()
                
                return success
            else:
                self.logger.error(f"Erro no download: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Erro durante download: {e}")
            return False
            
        finally:
            # Voltar ao diretório original
            os.chdir(original_cwd)
    
    def _extract_zip_files(self):
        """Extrai todos os arquivos ZIP no diretório de dados"""
        zip_files = list(config.DATA_DIR.glob('*.zip'))
        
        if not zip_files:
            self.logger.info("Nenhum arquivo ZIP encontrado para extrair")
            return
        
        for zip_file in zip_files:
            self._extract_specific_zip(zip_file)
    
    def _extract_specific_zip(self, zip_file):
        """Extrai um arquivo ZIP específico"""
        try:
            self.logger.info(f"Extraindo: {zip_file.name}")
            
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                # Listar conteúdo
                file_list = zip_ref.namelist()
                self.logger.info(f"Conteúdo do ZIP: {file_list}")
                
                # Extrair todos os arquivos
                zip_ref.extractall(config.DATA_DIR)
                
                # Verificar se extraímos para uma subpasta
                for file_name in file_list:
                    file_path = config.DATA_DIR / file_name
                    if file_path.is_dir():
                        # Se extraímos para uma pasta, mover arquivos para o diretório principal
                        self._move_files_from_subfolder(file_path)
            
            self.logger.info(f"Extraído: {zip_file.name}")
            
            # Opcional: remover arquivo ZIP após extrair
            # zip_file.unlink()
            # self.logger.info(f"Removido: {zip_file.name}")
            
        except zipfile.BadZipFile:
            self.logger.error(f"Arquivo ZIP corrompido: {zip_file.name}")
        except Exception as e:
            self.logger.error(f"Erro ao extrair {zip_file.name}: {e}")
    
    def _move_files_from_subfolder(self, subfolder):
        """Move arquivos de uma subpasta para o diretório principal"""
        try:
            if subfolder.exists() and subfolder.is_dir():
                self.logger.info(f"Movendo arquivos de {subfolder} para {config.DATA_DIR}")
                
                for item in subfolder.iterdir():
                    if item.is_file():
                        target_path = config.DATA_DIR / item.name
                        # Se já existe, adicionar sufixo
                        if target_path.exists():
                            stem = item.stem
                            suffix = item.suffix
                            counter = 1
                            while target_path.exists():
                                target_path = config.DATA_DIR / f"{stem}_{counter}{suffix}"
                                counter += 1
                        
                        shutil.move(str(item), str(target_path))
                        self.logger.info(f"  Movido: {item.name} -> {target_path.name}")
                
                # Tentar remover a pasta vazia
                try:
                    subfolder.rmdir()
                    self.logger.info(f"Pasta removida: {subfolder}")
                except:
                    pass
        except Exception as e:
            self.logger.error(f"Erro ao mover arquivos: {e}")
    
    def _verify_downloaded_files(self):
        """Verifica se os arquivos necessários foram baixados"""
        expected_files = [
            ('train.csv', 'Arquivo de treino'),
            ('test.csv', 'Arquivo de teste'),
        ]
        
        self.logger.info("Verificando arquivos baixados:")
        
        all_found = True
        for filename, description in expected_files:
            file_path = config.DATA_DIR / filename
            
            if file_path.exists():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                self.logger.info(f"  OK {description}: {filename} ({size_mb:.2f} MB)")
            else:
                self.logger.warning(f"  FALTANDO {description}: {filename}")
                all_found = False
        
        # Verificar também por arquivos ZIP
        zip_files = list(config.DATA_DIR.glob('*.zip'))
        if zip_files:
            self.logger.info(f"Arquivos ZIP encontrados: {[f.name for f in zip_files]}")
        
        return all_found
    
    def list_available_files(self):
        """Lista arquivos disponíveis na competição"""
        try:
            command = f"kaggle competitions files -c {config.KAGGLE_COMPETITION}"
            
            result = subprocess.run(
                command.split(),
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self.logger.info(f"Arquivos disponíveis na competição {config.KAGGLE_COMPETITION}:")
                print(result.stdout)
                return result.stdout
            else:
                self.logger.error(f"Erro ao listar arquivos: {result.stderr}")
                return None
                
        except Exception as e:
            self.logger.error(f"Erro: {e}")
            return None
    
    def clean_data_directory(self, keep_zips=False):
        """Limpa o diretório de dados"""
        confirmation = input("Tem certeza que deseja limpar o diretório de dados? (s/n): ")
        
        if confirmation.lower() == 's':
            try:
                # Remover arquivos CSV
                for csv_file in config.DATA_DIR.glob('*.csv'):
                    csv_file.unlink()
                    self.logger.info(f"Removido: {csv_file.name}")
                
                # Remover arquivos ZIP se não for para manter
                if not keep_zips:
                    for zip_file in config.DATA_DIR.glob('*.zip'):
                        zip_file.unlink()
                        self.logger.info(f"Removido: {zip_file.name}")
                
                self.logger.info("Diretório de dados limpo")
                return True
            except Exception as e:
                self.logger.error(f"Erro ao limpar diretório: {e}")
                return False
        else:
            self.logger.info("Operação cancelada")
            return False

# Função de conveniência
def download_data(force=False):
    """Função simples para download de dados"""
    downloader = KaggleDownloader()
    return downloader.download_competition_data(force_download=force)

if __name__ == "__main__":
    # Teste direto do downloader
    downloader = KaggleDownloader()
    
    print("=" * 60)
    print("TESTE DO DOWNLOADER KAGGLE")
    print("=" * 60)
    
    success = downloader.download_competition_data(force_download=False)
    
    if success:
        print("\nTeste concluído com sucesso!")
    else:
        print("\nTeste falhou")