# init_project.py
import sys
from pathlib import Path

def init_project():
    """Inicializa o projeto adicionando o diretório raiz ao path"""
    # Adicionar diretório raiz ao Python path
    ROOT_DIR = Path(__file__).parent
    sys.path.insert(0, str(ROOT_DIR))
    
    # Criar diretórios necessários
    directories = ['data', 'logs', 'models', 'submissions']
    for dir_name in directories:
        dir_path = ROOT_DIR / dir_name
        dir_path.mkdir(exist_ok=True)
    
    print(f"Projeto inicializado. ROOT_DIR: {ROOT_DIR}")
    return ROOT_DIR

if __name__ == "__main__":
    init_project()