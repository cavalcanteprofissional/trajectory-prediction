#!/usr/bin/env python3
"""
Scanner para detectar dependências e atualizar requirements.txt dinamicamente.
Ajustado para a nova estrutura de projeto de predição de trajetórias.
"""

import os
import re
import ast
import sys
import json
from pathlib import Path
from typing import Set, List, Dict, Optional
from dataclasses import dataclass, field
import subprocess

@dataclass
class ImportInfo:
    """Informações sobre um import detectado."""
    name: str
    line_number: int = 0
    file_path: Path = None
    is_stdlib: bool = False
    package_name: str = None

class RequirementsScanner:
    """Scanner para detectar dependências e gerar requirements.txt."""
    
    # Mapeamento de imports para nomes de pacotes PyPI
    IMPORT_TO_PACKAGE = {
        # Machine Learning & Data Science
        'sklearn': 'scikit-learn',
        'scikit_learn': 'scikit-learn',
        'catboost': 'catboost',
        'lightgbm': 'lightgbm',
        'xgboost': 'xgboost',
        'tensorflow': 'tensorflow',
        'tf': 'tensorflow',
        'keras': 'tensorflow',
        
        # Modelos do nosso projeto
        'RandomForestRegressor': 'scikit-learn',
        'GradientBoostingRegressor': 'scikit-learn',
        'LinearRegression': 'scikit-learn',
        'Ridge': 'scikit-learn',
        'Lasso': 'scikit-learn',
        'SVR': 'scikit-learn',
        'KNeighborsRegressor': 'scikit-learn',
        'XGBRegressor': 'xgboost',
        'LGBMRegressor': 'lightgbm',
        'CatBoostRegressor': 'catboost',
        
        # Otimização
        'BayesSearchCV': 'scikit-optimize',
        'skopt': 'scikit-optimize',
        
        # Data Science Core
        'numpy': 'numpy',
        'np': 'numpy',
        'pandas': 'pandas',
        'pd': 'pandas',
        'matplotlib': 'matplotlib',
        'plt': 'matplotlib',
        'seaborn': 'seaborn',
        'sns': 'seaborn',
        
        # Engenharia de Features
        'StandardScaler': 'scikit-learn',
        'MinMaxScaler': 'scikit-learn',
        'train_test_split': 'scikit-learn',
        'KFold': 'scikit-learn',
        'cross_val_score': 'scikit-learn',
        'MultiOutputRegressor': 'scikit-learn',
        
        # Métricas
        'mean_squared_error': 'scikit-learn',
        'mean_absolute_error': 'scikit-learn',
        
        # Kaggle
        'kaggle': 'kaggle',
        
        # Utilitários
        'tqdm': 'tqdm',
        'dotenv': 'python-dotenv',
        'warnings': 'python',  # stdlib
        'ast': 'python',  # stdlib
        'math': 'python',  # stdlib
        
        # Processamento de Trajetórias
        'pymove': 'pymove',
        
        # Deep Learning
        'tensorflow.keras': 'tensorflow',
        'keras': 'tensorflow',
        'layers': 'tensorflow',
        'Sequential': 'tensorflow',
        'LSTM': 'tensorflow',
        'Dense': 'tensorflow',
        'Dropout': 'tensorflow',
        'Conv1D': 'tensorflow',
        'MaxPooling1D': 'tensorflow',
        'Flatten': 'tensorflow',
        'Bidirectional': 'tensorflow',
    }
    
    # Módulos da biblioteca padrão do Python (não precisam ser instalados)
    STDLIB_MODULES = {
        'os', 'sys', 're', 'json', 'math', 'datetime', 'time', 'pathlib',
        'collections', 'itertools', 'functools', 'typing', 'subprocess',
        'random', 'statistics', 'csv', 'hashlib', 'base64', 'html', 'xml',
        'email', 'urllib', 'http', 'socket', 'ssl', 'threading', 'multiprocessing',
        'asyncio', 'copy', 'pprint', 'textwrap', 'string', 'decimal', 'fractions',
        'array', 'bisect', 'heapq', 'weakref', 'types', 'enum', 'numbers',
        'inspect', 'ast', 'symtable', 'tokenize', 'keyword', 'token', 'codecs',
        'unicodedata', 'stringprep', 'locale', 'gettext', 'getpass',
        'platform', 'errno', 'ctypes', 'select', 'mmap', 'readline', 'rlcompleter',
        'sysconfig', 'site', 'code', 'codeop', 'zipfile', 'tarfile', 'shutil',
        'tempfile', 'filecmp', 'stat', 'fnmatch', 'linecache', 'shlex', 'shelve',
        'marshal', 'dbm', 'sqlite3', 'pickle', 'socketserver', 'http.server',
        'wsgiref', 'cgi', 'cgitb', 'webbrowser', 'uuid', 'secrets', 'builtins',
        '__future__', 'abc', 'contextlib', 'dataclasses', 'io', 'logging',
        'operator', 'signal', 'traceback', 'warnings', 'argparse', 'configparser',
        'getopt', 'readline', 'rlcompleter', 'statistics', 'decimal', 'fractions',
        'itertools', 'collections', 'heapq', 'bisect', 'array', 'weakref',
        'types', 'copy', 'pprint', 'reprlib', 'enum', 'graphlib', 'dataclasses'
    }
    
    def __init__(self, project_root: str = None):
        """Inicializa o scanner com a estrutura do projeto."""
        self.tools_dir = Path(__file__).parent
        self.project_root = self.tools_dir.parent
        
        if project_root:
            self.project_root = Path(project_root).resolve()
        
        # Diretórios específicos da nossa estrutura
        self.target_dirs = [
            "config",
            "data",
            "features",
            "models",
            "training",
            "evaluation",
            "submission",
            "utils",
            "experiments",
            "notebooks"
        ]
        
        # Arquivos de saída
        self.requirements_file = self.project_root / "requirements.txt"
        self.dev_requirements_file = self.project_root / "requirements-dev.txt"
        self.setup_file = self.project_root / "setup.py"
        self.pyproject_file = self.project_root / "pyproject.toml"
        
        # Cache
        self.imports_found: List[ImportInfo] = []
        self.packages_found: Set[str] = set()
        
        # Configurações do projeto
        self.project_config = {
            'name': 'trajectory-prediction',
            'version': '1.0.0',
            'description': 'Trajectory destination prediction using machine learning',
            'python_requires': '>=3.8'
        }
    
    def find_files(self) -> List[Path]:
        """Encontra todos os arquivos Python/Notebook para análise."""
        files = []
        extensions = ['.py', '.ipynb', '.pyx']
        
        # Primeiro verificar arquivos na raiz
        for ext in extensions:
            for file in self.project_root.glob(f'*{ext}'):
                if file.name not in ['requirements_scanner.py', 'setup.py']:
                    files.append(file)
        
        # Verificar diretórios da estrutura
        for dir_name in self.target_dirs:
            dir_path = self.project_root / dir_name
            if dir_path.exists() and dir_path.is_dir():
                for ext in extensions:
                    for file in dir_path.rglob(f'*{ext}'):
                        files.append(file)
        
        # Verificar se main.py existe
        main_file = self.project_root / "main.py"
        if main_file.exists():
            files.append(main_file)
        
        return files
    
    def extract_imports_from_py(self, file_path: Path) -> List[ImportInfo]:
        """Extrai imports de arquivos .py usando AST para análise precisa."""
        imports = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            try:
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            # Extrair nome base do módulo (primeira parte)
                            base_name = alias.name.split('.')[0]
                            import_info = ImportInfo(
                                name=base_name,
                                line_number=getattr(node, 'lineno', 0),
                                file_path=file_path,
                                is_stdlib=base_name in self.STDLIB_MODULES
                            )
                            imports.append(import_info)
                    
                    elif isinstance(node, ast.ImportFrom):
                        if node.module and node.module != "__future__":
                            base_name = node.module.split('.')[0]
                            import_info = ImportInfo(
                                name=base_name,
                                line_number=getattr(node, 'lineno', 0),
                                file_path=file_path,
                                is_stdlib=base_name in self.STDLIB_MODULES
                            )
                            imports.append(import_info)
                            
                            # Também adicionar imports específicos do módulo
                            for alias in node.names:
                                full_import = f"{base_name}.{alias.name}"
                                # Verificar se é uma classe/objeto específico que mapeamos
                                for key in self.IMPORT_TO_PACKAGE:
                                    if alias.name == key or full_import == key:
                                        imports.append(ImportInfo(
                                            name=key,
                                            line_number=getattr(node, 'lineno', 0),
                                            file_path=file_path,
                                            is_stdlib=False
                                        ))
                            
            except SyntaxError as e:
                print(f"  ⚠ Erro de sintaxe em {file_path.name}: {e}")
                # Fallback para análise simples
                lines = content.split('\n')
                for i, line in enumerate(lines, 1):
                    imports.extend(self._extract_from_line(line, i, file_path))
                    
        except UnicodeDecodeError:
            print(f"  ⚠ Problema de encoding em {file_path.name}")
        except Exception as e:
            print(f"  ⚠ Erro em {file_path.name}: {e}")
        
        return imports
    
    def _extract_from_line(self, line: str, line_num: int, file_path: Path) -> List[ImportInfo]:
        """Extrai imports de uma linha de código."""
        imports = []
        line = line.strip()
        
        # Remover comentários
        if '#' in line:
            line = line.split('#')[0].strip()
        
        if line.startswith('import '):
            parts = [p.strip() for p in line[7:].split(',')]
            for part in parts:
                if part:
                    module = part.split()[0].split('.')[0]
                    if module:
                        imports.append(ImportInfo(
                            name=module,
                            line_number=line_num,
                            file_path=file_path,
                            is_stdlib=module in self.STDLIB_MODULES
                        ))
        
        elif line.startswith('from '):
            if ' import ' in line:
                module_part = line[5:].split(' import ')[0].strip()
                if module_part and module_part != '__future__':
                    module = module_part.split('.')[0]
                    if module:
                        imports.append(ImportInfo(
                            name=module,
                            line_number=line_num,
                            file_path=file_path,
                            is_stdlib=module in self.STDLIB_MODULES
                        ))
        
        return imports
    
    def extract_imports_from_notebook(self, file_path: Path) -> List[ImportInfo]:
        """Extrai imports de notebooks Jupyter."""
        imports = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                notebook = json.load(f)
            
            cell_num = 0
            for cell in notebook.get('cells', []):
                cell_num += 1
                if cell.get('cell_type') == 'code':
                    source = cell.get('source', [])
                    if isinstance(source, list):
                        source = ''.join(source)
                    
                    # Usar AST para extração precisa
                    try:
                        tree = ast.parse(source)
                        for node in ast.walk(tree):
                            if isinstance(node, ast.Import):
                                for alias in node.names:
                                    base_name = alias.name.split('.')[0]
                                    imports.append(ImportInfo(
                                        name=base_name,
                                        line_number=cell_num * 1000,
                                        file_path=file_path,
                                        is_stdlib=base_name in self.STDLIB_MODULES
                                    ))
                            elif isinstance(node, ast.ImportFrom):
                                if node.module and node.module != "__future__":
                                    base_name = node.module.split('.')[0]
                                    imports.append(ImportInfo(
                                        name=base_name,
                                        line_number=cell_num * 1000,
                                        file_path=file_path,
                                        is_stdlib=base_name in self.STDLIB_MODULES
                                    ))
                    except SyntaxError:
                        # Fallback para análise linha por linha
                        lines = str(source).split('\n')
                        for i, line in enumerate(lines, 1):
                            imports.extend(self._extract_from_line(
                                line, cell_num * 1000 + i, file_path
                            ))
                    
        except Exception as e:
            print(f"  ⚠ Notebook {file_path.name}: {e}")
        
        return imports
    
    def scan_imports(self) -> None:
        """Escaneia todos os arquivos e coleta imports."""
        print("🔍 Escaneando imports...")
        
        files = self.find_files()
        print(f"📁 Encontrados {len(files)} arquivos para análise")
        
        total_imports = 0
        for file_path in files:
            relative_path = file_path.relative_to(self.project_root)
            
            if file_path.suffix == '.ipynb':
                imports = self.extract_imports_from_notebook(file_path)
            else:
                imports = self.extract_imports_from_py(file_path)
            
            if imports:
                self.imports_found.extend(imports)
                total_imports += len(imports)
                print(f"  {relative_path}: {len(imports)} imports")
        
        print(f"📦 Total de imports detectados: {total_imports}")
    
    def process_imports(self) -> Set[str]:
        """Processa imports e converte para nomes de pacotes."""
        # Filtrar módulos da stdlib e duplicados
        seen_imports = set()
        unique_imports = []
        
        for imp in self.imports_found:
            if imp.name and imp.name not in seen_imports:
                seen_imports.add(imp.name)
                unique_imports.append(imp)
        
        # Separar stdlib de externos
        external_imports = [
            imp for imp in unique_imports 
            if not imp.is_stdlib and not self._is_local_module(imp.name)
        ]
        
        print(f"\n📊 Imports únicos: {len(unique_imports)}")
        print(f"📦 Imports externos: {len(external_imports)}")
        
        # Converter para pacotes PyPI
        packages = set()
        for imp in external_imports:
            package_name = self.IMPORT_TO_PACKAGE.get(imp.name, imp.name)
            
            # Verificar se é um pacote válido
            if self._is_valid_package_name(package_name):
                packages.add(package_name)
            else:
                # Tentar inferir do nome
                inferred_package = self._infer_package_name(imp.name)
                if inferred_package:
                    packages.add(inferred_package)
        
        # Adicionar dependências base do projeto
        base_dependencies = {
            'scikit-learn',  # Para modelos ML
            'pandas',        # Para manipulação de dados
            'numpy',         # Para computação numérica
            'matplotlib',    # Para visualização
            'seaborn',       # Para visualização estatística
        }
        
        packages.update(base_dependencies)
        
        return packages
    
    def _is_local_module(self, module_name: str) -> bool:
        """Verifica se é um módulo local do nosso projeto."""
        # Módulos da nossa estrutura
        local_modules = {
            'config', 'data', 'features', 'models', 'training',
            'evaluation', 'submission', 'utils', 'main',
            'DataDownloader', 'DataLoader', 'FeatureEngineer',
            'ModelFactory', 'ModelTrainer', 'SubmissionGenerator',
            'TrajectoryPredictor', 'setup_logger'
        }
        
        # Verificar se está em minúsculas e parece ser um nome de módulo local
        if module_name in local_modules:
            return True
        
        # Verificar se está nos nossos diretórios
        if module_name.lower() in [d.lower() for d in self.target_dirs]:
            return True
        
        # Verificar padrão de módulos locais (geralmente em snake_case)
        if '_' in module_name and module_name.islower():
            # Verificar se não é um pacote conhecido
            known_packages = {k.lower() for k in self.IMPORT_TO_PACKAGE.keys()}
            if module_name.lower() not in known_packages:
                return True
        
        return False
    
    def _infer_package_name(self, import_name: str) -> Optional[str]:
        """Tenta inferir o nome do pacote PyPI a partir do import."""
        # Regras de inferência comuns
        inference_rules = {
            # sklearn -> scikit-learn
            r'^sklearn$': 'scikit-learn',
            r'^scikit_learn$': 'scikit-learn',
            
            # tf/tensorflow -> tensorflow
            r'^tf$': 'tensorflow',
            
            # PIL -> pillow
            r'^PIL$': 'pillow',
            r'^Image$': 'pillow',
            
            # cv2 -> opencv-python
            r'^cv2$': 'opencv-python',
            
            # bs4 -> beautifulsoup4
            r'^bs4$': 'beautifulsoup4',
            
            # yaml -> pyyaml
            r'^yaml$': 'pyyaml',
        }
        
        for pattern, package in inference_rules.items():
            if re.match(pattern, import_name, re.IGNORECASE):
                return package
        
        return None
    
    def _is_valid_package_name(self, package_name: str) -> bool:
        """Verifica se é um nome de pacote PyPI válido."""
        if not package_name or package_name.startswith('_'):
            return False
        
        # Deve conter apenas caracteres válidos
        if not re.match(r'^[a-zA-Z0-9._-]+$', package_name):
            return False
        
        # Não deve ser muito curto ou muito longo
        if len(package_name) < 2 or len(package_name) > 100:
            return False
        
        # Verificar se parece ser um pacote real (não um módulo local)
        local_indicators = {'test', 'main', 'utils', 'config', 'data', 'models'}
        if package_name.lower() in local_indicators:
            return False
        
        return True
    
    def read_current_requirements(self) -> Dict[str, str]:
        """Lê as dependências atuais do requirements.txt."""
        if not self.requirements_file.exists():
            return {}
        
        packages = {}
        try:
            with open(self.requirements_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Extrair nome do pacote e versão
                        parts = line.split('==')
                        if len(parts) == 2:
                            packages[parts[0]] = parts[1]
                        else:
                            parts = line.split('>=')
                            if len(parts) == 2:
                                packages[parts[0]] = f'>={parts[1]}'
                            else:
                                packages[line] = ''
        except Exception as e:
            print(f"⚠ Erro ao ler requirements.txt: {e}")
        
        return packages
    
    def get_suggested_versions(self) -> Dict[str, str]:
        """Retorna versões sugeridas para pacotes comuns."""
        return {
            'numpy': '>=1.24.0',
            'pandas': '>=2.0.0',
            'scikit-learn': '>=1.3.0',
            'matplotlib': '>=3.7.0',
            'seaborn': '>=0.12.0',
            'xgboost': '>=1.7.0',
            'lightgbm': '>=3.3.0',
            'catboost': '>=1.0.0',
            'tensorflow': '>=2.12.0',
            'scikit-optimize': '>=0.9.0',
            'python-dotenv': '>=1.0.0',
            'kaggle': '>=1.5.0',
            'tqdm': '>=4.65.0',
            'pymove': '>=2.0.0',
        }
    
    def generate_requirements_content(self, packages: Set[str]) -> str:
        """Gera conteúdo do requirements.txt com versões apropriadas."""
        suggested_versions = self.get_suggested_versions()
        
        # Ordenar pacotes alfabeticamente
        sorted_packages = sorted(packages, key=lambda x: x.lower())
        
        # Gerar cabeçalho
        lines = [
            "# Requirements gerado automaticamente pelo scanner",
            "# Para instalar: pip install -r requirements.txt",
            "",
            "# Dependências principais",
            ""
        ]
        
        # Adicionar cada pacote com versão sugerida
        for package in sorted_packages:
            version = suggested_versions.get(package, "")
            lines.append(f"{package}{version}")
        
        # Adicionar seção de informações
        lines.extend([
            "",
            "# Informações do projeto",
            f"# Projeto: {self.project_config['name']}",
            f"# Versão: {self.project_config['version']}",
            f"# Python requerido: {self.project_config['python_requires']}",
            "",
            "# Para desenvolvimento, instale também:",
            "# pip install -r requirements-dev.txt",
        ])
        
        return '\n'.join(lines)
    
    def update_requirements_file(self, packages: Set[str], dry_run: bool = False) -> bool:
        """Atualiza o arquivo requirements.txt."""
        current_packages = self.read_current_requirements()
        current_package_names = set(current_packages.keys())
        
        # Encontrar diferenças
        new_packages = packages - current_package_names
        removed_packages = current_package_names - packages
        
        if not new_packages and not removed_packages:
            print("\n📭 Nenhuma alteração necessária no requirements.txt")
            return True
        
        print(f"\n📋 Pacotes atuais: {len(current_package_names)}")
        print(f"➕ Novos pacotes: {len(new_packages)}")
        print(f"➖ Pacotes removidos: {len(removed_packages)}")
        
        if new_packages:
            print("\n🎁 Novos pacotes para adicionar:")
            for pkg in sorted(new_packages):
                print(f"  • {pkg}")
        
        if removed_packages:
            print("\n🗑️  Pacotes para remover:")
            for pkg in sorted(removed_packages):
                print(f"  • {pkg}")
        
        if dry_run:
            print("\n🔍 MODO DRY RUN - Nenhuma alteração foi feita")
            return True
        
        # Criar backup se o arquivo existir
        if self.requirements_file.exists():
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = self.requirements_file.with_name(f"requirements_backup_{timestamp}.txt")
            
            try:
                import shutil
                shutil.copy2(self.requirements_file, backup_file)
                print(f"\n📋 Backup criado: {backup_file.relative_to(self.project_root)}")
            except Exception as e:
                print(f"⚠ Não foi possível criar backup: {e}")
        
        # Gerar novo conteúdo
        content = self.generate_requirements_content(packages)
        
        try:
            with open(self.requirements_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"\n✅ requirements.txt atualizado: {self.requirements_file.relative_to(self.project_root)}")
            print(f"📦 Total de pacotes: {len(packages)}")
            
            # Mostrar comando para instalar
            print("\n🎯 Para instalar as dependências:")
            print(f"   pip install -r {self.requirements_file.name}")
            
            # Verificar se há pacotes problemáticos
            problem_packages = self._check_problematic_packages(packages)
            if problem_packages:
                print("\n⚠ Pacotes que podem precisar de atenção:")
                for pkg, reason in problem_packages.items():
                    print(f"  • {pkg}: {reason}")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Erro ao atualizar requirements.txt: {e}")
            return False
    
    def _check_problematic_packages(self, packages: Set[str]) -> Dict[str, str]:
        """Verifica pacotes que podem ter problemas de instalação."""
        problematic = {}
        
        for package in packages:
            if package == 'tensorflow':
                # TensorFlow pode ter problemas de compatibilidade
                problematic[package] = "Verifique compatibilidade com CUDA se usar GPU"
            elif package == 'pymove':
                # pymove pode não estar disponível em todas as plataformas
                problematic[package] = "Pode precisar de dependências extras do sistema"
            elif package == 'opencv-python':
                problematic[package] = "Pode ser grande e ter dependências de sistema"
        
        return problematic
    
    def create_dev_requirements(self) -> None:
        """Cria um arquivo requirements-dev.txt com dependências de desenvolvimento."""
        dev_packages = {
            'black>=23.0.0',          # Formatação
            'flake8>=6.0.0',          # Linting
            'mypy>=1.0.0',            # Type checking
            'pytest>=7.0.0',          # Testing
            'pytest-cov>=4.0.0',      # Coverage
            'ipython>=8.0.0',         # REPL melhorado
            'jupyter>=1.0.0',         # Notebooks
            'jupyterlab>=4.0.0',      # JupyterLab
            'ipykernel>=6.0.0',       # Kernel para notebooks
            'pre-commit>=3.0.0',      # Hooks de pré-commit
            'pylint>=2.17.0',         # Análise estática
            'bandit>=1.7.0',          # Segurança
            'safety>=2.0.0',          # Verificação de vulnerabilidades
            'pip-audit>=2.0.0',       # Auditoria de dependências
            'mkdocs>=1.5.0',          # Documentação
            'mkdocs-material>=9.0.0', # Tema para documentação
            'jupyter-contrib-nbextensions>=0.7.0',  # Extensões do notebook
        }
        
        content = [
            "# Dependências de desenvolvimento",
            "# Para instalar: pip install -r requirements-dev.txt",
            "",
            "# Ferramentas de qualidade de código",
        ]
        
        content.extend(sorted(dev_packages))
        
        content.extend([
            "",
            "# Para configurar pre-commit hooks:",
            "# pre-commit install",
            "# pre-commit run --all-files",
            "",
            "# Para executar testes:",
            "# pytest tests/",
            "# pytest --cov=trajectory_prediction tests/",
        ])
        
        try:
            with open(self.dev_requirements_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(content))
            print(f"\n📝 requirements-dev.txt criado: {self.dev_requirements_file.relative_to(self.project_root)}")
        except Exception as e:
            print(f"⚠ Não foi possível criar requirements-dev.txt: {e}")
    
    def create_setup_py(self) -> None:
        """Cria um arquivo setup.py básico para o projeto."""
        setup_content = f'''"""
Setup para o projeto de predição de trajetórias.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="{self.project_config['name']}",
    version="{self.project_config['version']}",
    author="Seu Nome",
    author_email="seu.email@example.com",
    description="{self.project_config['description']}",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/seu-usuario/trajectory-prediction",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires="{self.project_config['python_requires']}",
    install_requires=requirements,
    extras_require={{
        "dev": [
            "black>=23.0.0",
            "flake8>=6.0.0",
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ]
    }},
    entry_points={{
        "console_scripts": [
            "trajectory-predict=main:main",
        ]
    }},
)
'''
        
        try:
            with open(self.setup_file, 'w', encoding='utf-8') as f:
                f.write(setup_content)
            print(f"📦 setup.py criado: {self.setup_file.relative_to(self.project_root)}")
        except Exception as e:
            print(f"⚠ Não foi possível criar setup.py: {e}")
    
    def show_detailed_report(self) -> None:
        """Mostra relatório detalhado dos imports encontrados."""
        if not self.imports_found:
            print("\n📭 Nenhum import encontrado")
            return
        
        # Agrupar por módulo
        import_counts = {}
        module_files = {}
        
        for imp in self.imports_found:
            import_counts[imp.name] = import_counts.get(imp.name, 0) + 1
            if imp.name not in module_files:
                module_files[imp.name] = set()
            if imp.file_path:
                module_files[imp.name].add(imp.file_path.relative_to(self.project_root))
        
        print("\n📊 RELATÓRIO DETALHADO DE IMPORTS:")
        print("=" * 80)
        
        # Separar stdlib de externos
        stdlib_imports = {k: v for k, v in import_counts.items() 
                         if k in self.STDLIB_MODULES or self._is_stdlib_module(k)}
        external_imports = {k: v for k, v in import_counts.items() 
                          if k not in stdlib_imports and not self._is_local_module(k)}
        local_imports = {k: v for k, v in import_counts.items() 
                        if self._is_local_module(k)}
        
        print(f"\n📚 Módulos da stdlib ({len(stdlib_imports)}):")
        for module, count in sorted(stdlib_imports.items(), key=lambda x: (-x[1], x[0])):
            files = list(module_files.get(module, []))[:3]
            files_str = ", ".join(str(f) for f in files[:2])
            if len(files) > 2:
                files_str += f", ... (+{len(files)-2})"
            print(f"  {module:20s} {count:3d}×  [{files_str}]")
        
        print(f"\n📦 Módulos externos ({len(external_imports)}):")
        for module, count in sorted(external_imports.items(), key=lambda x: (-x[1], x[0])):
            package = self.IMPORT_TO_PACKAGE.get(module, module)
            files = list(module_files.get(module, []))[:3]
            files_str = ", ".join(str(f) for f in files[:2])
            if len(files) > 2:
                files_str += f", ... (+{len(files)-2})"
            print(f"  {module:20s} → {package:20s} {count:3d}×  [{files_str}]")
        
        if local_imports:
            print(f"\n🏠 Módulos locais ({len(local_imports)}):")
            for module, count in sorted(local_imports.items(), key=lambda x: (-x[1], x[0])):
                files = list(module_files.get(module, []))[:3]
                files_str = ", ".join(str(f) for f in files[:2])
                if len(files) > 2:
                    files_str += f", ... (+{len(files)-2})"
                print(f"  {module:20s} {count:3d}×  [{files_str}]")
    
    def _is_stdlib_module(self, module_name: str) -> bool:
        """Verifica se um módulo é da biblioteca padrão."""
        if module_name in self.STDLIB_MODULES:
            return True
        
        # Verificar sub-módulos (como tensorflow.keras -> tensorflow)
        for stdlib_module in self.STDLIB_MODULES:
            if module_name.startswith(stdlib_module + '.'):
                return True
        
        return False
    
    def run(self, dry_run: bool = False, create_all: bool = False) -> None:
        """Executa o scanner completo."""
        print("=" * 80)
        print("📦 SCANNER DE DEPENDÊNCIAS - Projeto de Predição de Trajetórias")
        print("=" * 80)
        print(f"📂 Diretório raiz: {self.project_root}")
        print(f"📁 Estrutura do projeto: {', '.join(self.target_dirs)}")
        print("-" * 80)
        
        # Verificar estrutura do projeto
        print("\n🔍 Verificando estrutura do projeto:")
        missing_dirs = []
        for dir_name in self.target_dirs:
            dir_path = self.project_root / dir_name
            if dir_path.exists():
                print(f"  ✓ {dir_name}/")
            else:
                print(f"  ⚠ {dir_name}/ (não encontrado)")
                missing_dirs.append(dir_name)
        
        if missing_dirs:
            print(f"\n⚠ Diretórios faltando na estrutura: {', '.join(missing_dirs)}")
        
        # Escanear imports
        print("\n" + "=" * 80)
        self.scan_imports()
        
        # Processar imports para pacotes
        packages = self.process_imports()
        
        if not packages:
            print("\n📭 Nenhuma dependência externa encontrada.")
            return
        
        # Mostrar relatório detalhado
        self.show_detailed_report()
        
        # Atualizar requirements.txt
        print("\n" + "=" * 80)
        success = self.update_requirements_file(packages, dry_run)
        
        if success and create_all and not dry_run:
            print("\n" + "=" * 80)
            
            # Criar requirements-dev.txt
            self.create_dev_requirements()
            
            # Criar setup.py
            self.create_setup_py()
            
            # Sugerir próximos passos
            print("\n🎯 PRÓXIMOS PASSOS:")
            print("1. Instale as dependências:")
            print("   pip install -r requirements.txt")
            print("\n2. Para desenvolvimento, instale também:")
            print("   pip install -r requirements-dev.txt")
            print("\n3. Configure pre-commit hooks:")
            print("   pre-commit install")
            print("\n4. Teste a instalação:")
            print("   python -c \"import pandas; import sklearn; print('✓ Dependências OK')\"")
        
        print("=" * 80)


def main():
    """Função principal."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Scanner para gerar/atualizar requirements.txt para projeto de predição de trajetórias",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  %(prog)s                    # Executa scanner normalmente
  %(prog)s --dry-run          # Mostra o que seria feito sem alterar
  %(prog)s --create-all       # Cria requirements.txt, requirements-dev.txt e setup.py
  %(prog)s --root ../outro    # Escaneia outro diretório
        """
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Mostra o que seria feito sem modificar arquivos"
    )
    parser.add_argument(
        "--create-all",
        action="store_true",
        help="Cria requirements.txt, requirements-dev.txt e setup.py"
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Diretório raiz do projeto"
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Mostra apenas relatório detalhado sem modificar arquivos"
    )
    parser.add_argument(
        "--no-dev",
        action="store_true",
        help="Não cria requirements-dev.txt mesmo com --create-all"
    )
    
    args = parser.parse_args()
    
    try:
        scanner = RequirementsScanner(args.root)
        
        if args.report:
            scanner.scan_imports()
            scanner.show_detailed_report()
            packages = scanner.process_imports()
            print(f"\n🎯 Pacotes PyPI detectados: {len(packages)}")
            for pkg in sorted(packages):
                print(f"  • {pkg}")
        else:
            scanner.run(
                dry_run=args.dry_run,
                create_all=args.create_all and not args.no_dev
            )
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Interrompido pelo usuário")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()