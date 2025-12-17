# utils/logger.py
import logging
import sys
from pathlib import Path

def setup_logger(name, log_file=None, level=logging.INFO):
    """Configura um logger sem emojis para compatibilidade Windows"""
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Formatação sem cores ou emojis
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Handler para console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Handler para arquivo
    if log_file:
        # Certificar que o diretório existe
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_logger(name):
    """Retorna um logger configurado"""
    # Usar caminho relativo
    project_root = Path(__file__).parent.parent
    
    # Criar diretório de logs se não existir
    logs_dir = project_root / "logs"
    
    # Nome do arquivo de log
    log_filename = f"{name.replace('.', '_')}.log"
    log_file = logs_dir / log_filename
    
    return setup_logger(name, log_file)