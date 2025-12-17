#!/usr/bin/env python3
"""
Scanner dinâmico para detectar dependências Python e atualizar requirements.txt.
Versão genérica que funciona com qualquer estrutura de projeto.
Suporte para pip, Poetry e gerenciamento de ambientes virtuais.
"""

import os
import re
import ast
import sys
import json
import shutil
from pathlib import Path
from typing import Set, List, Dict, Optional, Tuple
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
    """Scanner genérico para detectar dependências e gerar requirements.txt."""
    
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
        'torch': 'torch',
        
        # Modelos comuns
        'RandomForestRegressor': 'scikit-learn',
        'RandomForestClassifier': 'scikit-learn',
        'GradientBoostingRegressor': 'scikit-learn',
        'GradientBoostingClassifier': 'scikit-learn',
        'LinearRegression': 'scikit-learn',
        'LogisticRegression': 'scikit-learn',
        'Ridge': 'scikit-learn',
        'Lasso': 'scikit-learn',
        'SVR': 'scikit-learn',
        'SVC': 'scikit-learn',
        'KNeighborsRegressor': 'scikit-learn',
        'KNeighborsClassifier': 'scikit-learn',
        'XGBRegressor': 'xgboost',
        'XGBClassifier': 'xgboost',
        'LGBMRegressor': 'lightgbm',
        'LGBMClassifier': 'lightgbm',
        'CatBoostRegressor': 'catboost',
        'CatBoostClassifier': 'catboost',
        
        # Otimização
        'BayesSearchCV': 'scikit-optimize',
        'skopt': 'scikit-optimize',
        'optuna': 'optuna',
        'hyperopt': 'hyperopt',
        
        # Data Science Core
        'numpy': 'numpy',
        'np': 'numpy',
        'pandas': 'pandas',
        'pd': 'pandas',
        'matplotlib': 'matplotlib',
        'plt': 'matplotlib',
        'seaborn': 'seaborn',
        'sns': 'seaborn',
        'scipy': 'scipy',
        'statsmodels': 'statsmodels',
        
        # Engenharia de Features
        'StandardScaler': 'scikit-learn',
        'MinMaxScaler': 'scikit-learn',
        'RobustScaler': 'scikit-learn',
        'OneHotEncoder': 'scikit-learn',
        'LabelEncoder': 'scikit-learn',
        'train_test_split': 'scikit-learn',
        'KFold': 'scikit-learn',
        'StratifiedKFold': 'scikit-learn',
        'cross_val_score': 'scikit-learn',
        'GridSearchCV': 'scikit-learn',
        'RandomizedSearchCV': 'scikit-learn',
        'MultiOutputRegressor': 'scikit-learn',
        
        # Métricas
        'mean_squared_error': 'scikit-learn',
        'mean_absolute_error': 'scikit-learn',
        'r2_score': 'scikit-learn',
        'accuracy_score': 'scikit-learn',
        'precision_score': 'scikit-learn',
        'recall_score': 'scikit-learn',
        'f1_score': 'scikit-learn',
        'confusion_matrix': 'scikit-learn',
        'classification_report': 'scikit-learn',
        
        # Bancos de dados e APIs
        'sqlite3': 'python',  # stdlib
        'psycopg2': 'psycopg2-binary',
        'pymysql': 'pymysql',
        'sqlalchemy': 'sqlalchemy',
        'flask': 'flask',
        'django': 'django',
        'fastapi': 'fastapi',
        'requests': 'requests',
        
        # Utilitários
        'tqdm': 'tqdm',
        'dotenv': 'python-dotenv',
        'warnings': 'python',  # stdlib
        'ast': 'python',  # stdlib
        'math': 'python',  # stdlib
        'yaml': 'pyyaml',
        'toml': 'toml',
        
        # Visualização
        'plotly': 'plotly',
        'bokeh': 'bokeh',
        'altair': 'altair',
        'graphviz': 'graphviz',
        
        # Processamento de texto
        'nltk': 'nltk',
        'spacy': 'spacy',
        'transformers': 'transformers',
        
        # Processamento de imagem
        'PIL': 'pillow',
        'Image': 'pillow',
        'cv2': 'opencv-python',
        
        # Web scraping
        'bs4': 'beautifulsoup4',
        'BeautifulSoup': 'beautifulsoup4',
        'selenium': 'selenium',
        'scrapy': 'scrapy',
        
        # Processamento paralelo
        'joblib': 'joblib',
        'dask': 'dask',
        'multiprocessing': 'python',  # stdlib
        'concurrent': 'python',  # stdlib
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
        'types', 'copy', 'pprint', 'reprlib', 'enum', 'graphlib', 'dataclasses',
        'typing_extensions', 'contextvars', 'importlib', 'pkgutil', 'zipimport'
    }
    
    # Diretórios e arquivos para ignorar
    DEFAULT_IGNORE_PATTERNS = [
        # Diretórios
        r'^\.venv$',
        r'^venv$',
        r'^\.env$',
        r'^env$',
        r'^\.git$',
        r'^__pycache__$',
        r'^\.pytest_cache$',
        r'^\.mypy_cache$',
        r'^node_modules$',
        r'^\.ipynb_checkpoints$',
        r'^build$',
        r'^dist$',
        r'^\.eggs$',
        r'^\.tox$',
        r'^\.coverage$',
        r'^\.hypothesis$',
        r'^\.vscode$',
        r'^\.idea$',
        r'^\.vs$',
        r'^\.history$',
        # Arquivos
        r'^\.gitignore$',
        r'^\.env\.*',
        r'^requirements.*\.txt$',
        r'^setup\.py$',
        r'^pyproject\.toml$',
        r'^poetry\.lock$',
        r'^Pipfile$',
        r'^Pipfile\.lock$',
        r'^\.pre-commit-config\.yaml$',
        r'^\.flake8$',
        r'^\.pylintrc$',
        r'^\.coveragerc$',
        r'^\.dockerignore$',
        r'^Dockerfile$',
        r'^docker-compose\.yml$',
        r'^\.editorconfig$',
    ]
    
    # Extensões de arquivo para analisar
    VALID_EXTENSIONS = {'.py', '.ipynb', '.pyx', '.pyi'}
    
    def __init__(self, project_root: str = None, ignore_patterns: List[str] = None):
        """Inicializa o scanner com configuração personalizável."""
        self.tools_dir = Path(__file__).parent
        self.project_root = self.tools_dir.parent
        
        if project_root:
            self.project_root = Path(project_root).resolve()
        
        # Padrões para ignorar
        self.ignore_patterns = ignore_patterns or self.DEFAULT_IGNORE_PATTERNS
        self.ignore_regexes = [re.compile(pattern) for pattern in self.ignore_patterns]
        
        # Arquivos de saída
        self.requirements_file = self.project_root / "requirements.txt"
        self.dev_requirements_file = self.project_root / "requirements-dev.txt"
        self.setup_file = self.project_root / "setup.py"
        self.pyproject_file = self.project_root / "pyproject.toml"
        self.poetry_lock_file = self.project_root / "poetry.lock"
        
        # Cache
        self.imports_found: List[ImportInfo] = []
        self.packages_found: Set[str] = set()
        
        # Detectar configurações do projeto
        self.project_config = self._detect_project_config()
        
        # Detectar gerenciador de pacotes
        self.package_manager = self._detect_package_manager()
    
    def _detect_package_manager(self) -> str:
        """Detecta qual gerenciador de pacotes está sendo usado."""
        # Verificar se Poetry está configurado
        if self.pyproject_file.exists():
            try:
                with open(self.pyproject_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if '[tool.poetry]' in content:
                        return 'poetry'
            except:
                pass
        
        # Verificar se Pipenv está configurado
        pipfile = self.project_root / "Pipfile"
        if pipfile.exists():
            return 'pipenv'
        
        # Verificar se requirements.txt existe (pip)
        if self.requirements_file.exists():
            return 'pip'
        
        # Default para pip
        return 'pip'
    
    def _detect_project_config(self) -> Dict:
        """Detecta configurações do projeto automaticamente."""
        config = {
            'name': self.project_root.name.lower().replace(' ', '-'),
            'version': '1.0.0',
            'description': f'Python project: {self.project_root.name}',
            'python_requires': '>=3.8',
            'authors': ['Your Name <you@example.com>'],
            'license': 'MIT'
        }
        
        # Tentar ler do pyproject.toml se existir
        if self.pyproject_file.exists():
            try:
                import tomli
                with open(self.pyproject_file, 'rb') as f:
                    pyproject_data = tomli.load(f)
                
                if 'project' in pyproject_data:
                    project_data = pyproject_data['project']
                    config['name'] = project_data.get('name', config['name'])
                    config['version'] = project_data.get('version', config['version'])
                    config['description'] = project_data.get('description', config['description'])
                    config['python_requires'] = project_data.get('requires-python', config['python_requires'])
                
                # Tentar ler do Poetry
                elif 'tool' in pyproject_data and 'poetry' in pyproject_data['tool']:
                    poetry_data = pyproject_data['tool']['poetry']
                    config['name'] = poetry_data.get('name', config['name'])
                    config['version'] = poetry_data.get('version', config['version'])
                    config['description'] = poetry_data.get('description', config['description'])
                    config['authors'] = poetry_data.get('authors', config['authors'])
                    config['license'] = poetry_data.get('license', config['license'])
            except ImportError:
                pass  # tomli não está disponível
            except Exception:
                pass  # Ignorar erros na leitura
        
        # Tentar ler do setup.py se existir
        elif self.setup_file.exists():
            try:
                with open(self.setup_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extrações simples por regex
                name_match = re.search(r"name=['\"]([^'\"]+)['\"]", content)
                if name_match:
                    config['name'] = name_match.group(1)
                
                version_match = re.search(r"version=['\"]([^'\"]+)['\"]", content)
                if version_match:
                    config['version'] = version_match.group(1)
                
                desc_match = re.search(r"description=['\"]([^'\"]+)['\"]", content)
                if desc_match:
                    config['description'] = desc_match.group(1)
                
                python_match = re.search(r"python_requires=['\"]([^'\"]+)['\"]", content)
                if python_match:
                    config['python_requires'] = python_match.group(1)
                
                author_match = re.search(r"author=['\"]([^'\"]+)['\"]", content)
                if author_match:
                    config['authors'] = [f"{author_match.group(1)}"]
            except Exception:
                pass  # Ignorar erros na leitura
        
        return config
    
    def should_ignore(self, path: Path) -> bool:
        """Verifica se um caminho deve ser ignorado."""
        # Verificar se é um diretório ou arquivo oculto (exceto .py, .ipynb, etc.)
        if path.name.startswith('.') and path.suffix not in self.VALID_EXTENSIONS:
            return True
        
        # Verificar padrões de regex
        for pattern in self.ignore_regexes:
            if pattern.search(str(path.name)):
                return True
        
        # Verificar se está em um diretório ignorado
        for parent in path.parents:
            parent_name = parent.name
            for pattern in self.ignore_regexes:
                if pattern.search(parent_name):
                    return True
        
        return False
    
    def find_files(self) -> List[Path]:
        """Encontra todos os arquivos Python/Notebook para análise, ignorando padrões."""
        files = []
        
        # Usar walk recursivo
        for root, dirs, filenames in os.walk(self.project_root, topdown=True):
            root_path = Path(root)
            
            # Filtrar diretórios para ignorar
            dirs[:] = [d for d in dirs if not self.should_ignore(root_path / d)]
            
            # Processar arquivos
            for filename in filenames:
                file_path = root_path / filename
                
                # Ignorar arquivos baseado nos padrões
                if self.should_ignore(file_path):
                    continue
                
                # Verificar extensão
                if file_path.suffix.lower() in self.VALID_EXTENSIONS:
                    # Ignorar o próprio scanner se estiver no projeto
                    if file_path.name == 'requirements_scanner.py' and 'tools' in str(file_path):
                        continue
                    files.append(file_path)
        
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
                            if alias.name:
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
                                # Verificar se é uma classe/objeto específico que mapeamos
                                for key in self.IMPORT_TO_PACKAGE:
                                    if alias.name == key:
                                        imports.append(ImportInfo(
                                            name=key,
                                            line_number=getattr(node, 'lineno', 0),
                                            file_path=file_path,
                                            is_stdlib=False
                                        ))
                            
            except SyntaxError as e:
                print(f"  ⚠ Erro de sintaxe em {file_path.relative_to(self.project_root)}: {e}")
                # Fallback para análise simples
                lines = content.split('\n')
                for i, line in enumerate(lines, 1):
                    imports.extend(self._extract_from_line(line, i, file_path))
                    
        except UnicodeDecodeError:
            print(f"  ⚠ Problema de encoding em {file_path.relative_to(self.project_root)}")
        except Exception as e:
            print(f"  ⚠ Erro em {file_path.relative_to(self.project_root)}: {e}")
        
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
                    if module and not module.startswith('_'):
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
                    if module and not module.startswith('_'):
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
                                    if alias.name:
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
                    
        except json.JSONDecodeError:
            print(f"  ⚠ Notebook inválido: {file_path.relative_to(self.project_root)}")
        except Exception as e:
            print(f"  ⚠ Notebook {file_path.relative_to(self.project_root)}: {e}")
        
        return imports
    
    def scan_imports(self) -> None:
        """Escaneia todos os arquivos e coleta imports."""
        print("🔍 Escaneando imports...")
        
        files = self.find_files()
        print(f"📁 Encontrados {len(files)} arquivos para análise")
        
        if not files:
            print("⚠ Nenhum arquivo Python/Jupyter encontrado para análise")
            return
        
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
                if len(imports) > 0:
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
        
        # Adicionar dependências base comuns
        if packages:  # Só adicionar se já houver pacotes
            base_dependencies = {
                'setuptools',  # Para instalação de pacotes
            }
            packages.update(base_dependencies)
        
        return packages
    
    def _is_local_module(self, module_name: str) -> bool:
        """Verifica se é um módulo local do projeto."""
        # Verificar se o módulo existe localmente
        possible_paths = [
            self.project_root / module_name,
            self.project_root / f"{module_name}.py",
            self.project_root / f"{module_name}/__init__.py",
        ]
        
        for path in possible_paths:
            if path.exists():
                return True
        
        # Verificar se é um arquivo Python no diretório raiz
        for py_file in self.project_root.glob("*.py"):
            if py_file.stem == module_name:
                return True
        
        # Verificar subdiretórios
        for subdir in self.project_root.iterdir():
            if subdir.is_dir():
                # Verificar se é um pacote Python
                if (subdir / "__init__.py").exists() and subdir.name == module_name:
                    return True
                
                # Verificar se é um módulo Python dentro do subdiretório
                for py_file in subdir.glob("*.py"):
                    if py_file.stem == module_name:
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
            r'^ImageDraw$': 'pillow',
            r'^ImageFilter$': 'pillow',
            
            # cv2 -> opencv-python
            r'^cv2$': 'opencv-python',
            
            # bs4 -> beautifulsoup4
            r'^bs4$': 'beautifulsoup4',
            r'^BeautifulSoup$': 'beautifulsoup4',
            
            # yaml -> pyyaml
            r'^yaml$': 'pyyaml',
            r'^ruamel\.yaml$': 'ruamel.yaml',
            
            # configparser -> parte do stdlib em Python 3
            r'^configparser$': 'python',
            
            # sklearn extras
            r'^sklearn\.': 'scikit-learn',
            
            # tensorflow extras
            r'^tensorflow\.': 'tensorflow',
            r'^tf\.': 'tensorflow',
            
            # matplotlib extras
            r'^mpl_toolkits$': 'matplotlib',
            
            # django extras
            r'^django\.': 'django',
            
            # flask extras
            r'^flask\.': 'flask',
            
            # fastapi extras
            r'^fastapi\.': 'fastapi',
        }
        
        for pattern, package in inference_rules.items():
            if re.match(pattern, import_name, re.IGNORECASE):
                return package
        
        # Inferência por padrões comuns
        if import_name.endswith('_lib'):
            return import_name[:-4]
        elif import_name.endswith('_utils'):
            return import_name[:-6]
        elif '_' in import_name and import_name.replace('_', '-') != import_name:
            return import_name.replace('_', '-')
        
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
        
        # Não deve ser apenas números
        if package_name.isdigit():
            return False
        
        # Verificar se parece ser um pacote real (não um módulo local)
        local_indicators = {
            'test', 'tests', 'main', 'utils', 'config', 'data', 'models',
            'src', 'lib', 'core', 'common', 'shared', 'helpers', 'scripts',
            'app', 'api', 'web', 'cli', 'backend', 'frontend', 'database'
        }
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
                        parts = re.split(r'(==|>=|<=|>|<|~=)', line, maxsplit=1)
                        if len(parts) >= 2:
                            packages[parts[0]] = ''.join(parts[1:])
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
            'torch': '>=2.0.0',
            'scikit-optimize': '>=0.9.0',
            'python-dotenv': '>=1.0.0',
            'kaggle': '>=1.5.0',
            'tqdm': '>=4.65.0',
            'requests': '>=2.31.0',
            'flask': '>=2.3.0',
            'fastapi': '>=0.100.0',
            'sqlalchemy': '>=2.0.0',
            'pytest': '>=7.4.0',
            'black': '>=23.0.0',
            'flake8': '>=6.0.0',
            'mypy': '>=1.5.0',
        }
    
    def generate_requirements_content(self, packages: Set[str]) -> str:
        """Gera conteúdo do requirements.txt com versões apropriadas."""
        suggested_versions = self.get_suggested_versions()
        
        # Ordenar pacotes alfabeticamente
        sorted_packages = sorted(packages, key=lambda x: x.lower())
        
        # Agrupar por categoria
        categories = {
            'Data Science & ML': {'numpy', 'pandas', 'scikit-learn', 'scipy', 
                                 'matplotlib', 'seaborn', 'statsmodels'},
            'Deep Learning': {'tensorflow', 'torch', 'keras'},
            'Boosting': {'xgboost', 'lightgbm', 'catboost'},
            'Utils & Tools': {'tqdm', 'python-dotenv', 'requests', 'joblib'},
            'Web & APIs': {'flask', 'fastapi', 'django'},
            'Database': {'sqlalchemy', 'psycopg2-binary', 'pymysql'},
            'Optimization': {'scikit-optimize', 'optuna', 'hyperopt'},
            'Visualization': {'plotly', 'bokeh', 'altair'},
            'NLP': {'nltk', 'spacy', 'transformers'},
        }
        
        # Gerar cabeçalho
        lines = [
            f"# Requirements gerado automaticamente por requirements_scanner.py",
            f"# Projeto: {self.project_config['name']} v{self.project_config['version']}",
            f"# Python requerido: {self.project_config['python_requires']}",
            "#",
            "# Para instalar: pip install -r requirements.txt",
            "# Para Poetry: poetry add $(cat requirements.txt | grep -v '^#')",
            "#",
        ]
        
        # Adicionar pacotes por categoria
        categorized_packages = set()
        for category, cat_packages in categories.items():
            cat_found = [pkg for pkg in sorted_packages if pkg in cat_packages]
            if cat_found:
                lines.append(f"# {category}")
                for pkg in sorted(cat_found):
                    version = suggested_versions.get(pkg, "")
                    lines.append(f"{pkg}{version}")
                    categorized_packages.add(pkg)
                lines.append("")
        
        # Adicionar pacotes não categorizados
        uncategorized = [pkg for pkg in sorted_packages if pkg not in categorized_packages]
        if uncategorized:
            lines.append("# Outras dependências")
            for pkg in uncategorized:
                version = suggested_versions.get(pkg, "")
                lines.append(f"{pkg}{version}")
            lines.append("")
        
        # Adicionar informações úteis
        lines.extend([
            "# COMANDOS ÚTEIS:",
            "#",
            "# Usando pip:",
            "#   pip install -r requirements.txt",
            "#   pip install --upgrade -r requirements.txt",
            "#",
            "# Usando Poetry:",
            "#   poetry add $(cat requirements.txt | grep -v '^#' | xargs)",
            "#   poetry update",
            "#   poetry install",
            "#   poetry lock",
            "#",
            "# Geral:",
            "#   python -m pip install --upgrade pip",
            "#   pip list --outdated",
            "#   pip freeze > requirements.lock",
            "",
            "# Notas:",
            "# Este arquivo foi gerado automaticamente. Para editar manualmente,",
            "# modifique o código-fonte e execute o scanner novamente.",
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
            self.requirements_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.requirements_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"\n✅ requirements.txt atualizado: {self.requirements_file.relative_to(self.project_root)}")
            print(f"📦 Total de pacotes: {len(packages)}")
            
            # Mostrar comandos de instalação baseados no gerenciador detectado
            self._show_installation_commands(packages)
            
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
    
    def _show_installation_commands(self, packages: Set[str]) -> None:
        """Mostra comandos de instalação baseados no gerenciador de pacotes."""
        print("\n🎯 COMANDOS DE INSTALAÇÃO:")
        print("-" * 40)
        
        # Comandos para pip
        print("\n📦 Usando pip:")
        print(f"   pip install -r {self.requirements_file.name}")
        print(f"   pip install --upgrade -r {self.requirements_file.name}")
        
        # Comandos para Poetry (se pyproject.toml existir ou for Poetry)
        if self.package_manager == 'poetry' or self.pyproject_file.exists():
            print("\n🎵 Usando Poetry:")
            print(f"   poetry add $(cat {self.requirements_file.name} | grep -v '^#' | xargs)")
            print("   poetry update")
            print("   poetry install")
            print("   poetry lock")
            print("\n   # Ou adicionar pacotes individualmente:")
            for pkg in sorted(packages)[:5]:  # Mostrar primeiros 5
                print(f"   poetry add {pkg}")
            if len(packages) > 5:
                print(f"   # ... e mais {len(packages) - 5} pacotes")
        
        # Comandos para Pipenv
        elif self.package_manager == 'pipenv':
            print("\n🐍 Usando Pipenv:")
            print(f"   pipenv install -r {self.requirements_file.name}")
            print("   pipenv update")
            print("   pipenv sync")
        
        # Comandos gerais úteis
        print("\n🛠️  Comandos úteis:")
        print("   python -m pip install --upgrade pip")
        print("   pip list --outdated")
        print("   pip freeze > requirements.lock")
        
        # Sugerir criação de ambiente virtual se não existir
        venv_paths = [
            self.project_root / '.venv',
            self.project_root / 'venv',
            self.project_root / 'env'
        ]
        
        if not any(p.exists() for p in venv_paths):
            print("\n💡 Ambiente virtual não encontrado. Sugestões:")
            print("   python -m venv .venv")
            if os.name == 'nt':  # Windows
                print("   .venv\\Scripts\\activate")
            else:  # Unix/Linux/Mac
                print("   source .venv/bin/activate")
    
    def _check_problematic_packages(self, packages: Set[str]) -> Dict[str, str]:
        """Verifica pacotes que podem ter problemas de instalação."""
        problematic = {}
        
        # Verificar pacotes conhecidos por terem problemas
        problematic_checks = {
            'tensorflow': "Verifique compatibilidade com CUDA se usar GPU",
            'torch': "Pode precisar de instalação específica para CUDA/CPU",
            'opencv-python': "Pode ser grande e ter dependências de sistema",
            'pyspark': "Requer Java JDK instalado",
            'psycopg2-binary': "Para desenvolvimento apenas, use psycopg2 em produção",
            'mysqlclient': "Pode requerer bibliotecas de sistema no Linux/macOS",
            'prophet': "Pode ter dependências complicadas no Windows",
            'lightgbm': "Pode precisar de compilador C++ no Windows",
            'graphviz': "Requer Graphviz instalado no sistema",
            'dlib': "Compilação complexa, considere dlib-bin",
        }
        
        for package in packages:
            if package in problematic_checks:
                problematic[package] = problematic_checks[package]
        
        return problematic
    
    def create_dev_requirements(self) -> None:
        """Cria um arquivo requirements-dev.txt com dependências de desenvolvimento."""
        dev_packages = {
            # Testing
            'pytest>=7.4.0',
            'pytest-cov>=4.1.0',
            'pytest-xdist>=3.3.0',
            'pytest-mock>=3.11.0',
            'coverage>=7.3.0',
            
            # Code quality
            'black>=23.0.0',
            'flake8>=6.0.0',
            'mypy>=1.5.0',
            'pylint>=2.17.0',
            'isort>=5.12.0',
            'autoflake>=2.2.0',
            'bandit>=1.7.0',
            
            # Security
            'safety>=2.0.0',
            'pip-audit>=2.5.0',
            
            # Documentation
            'sphinx>=7.2.0',
            'sphinx-rtd-theme>=1.3.0',
            'myst-parser>=2.0.0',
            'pydocstyle>=6.3.0',
            
            # Development tools
            'ipython>=8.15.0',
            'jupyter>=1.0.0',
            'jupyterlab>=4.0.0',
            'ipykernel>=6.25.0',
            'pre-commit>=3.4.0',
            'bump2version>=1.0.1',
            'twine>=4.0.0',
            'build>=0.10.0',
            
            # Notebooks
            'jupyter-contrib-nbextensions>=0.7.0',
            'jupyter-nbextensions-configurator>=0.6.0',
            
            # Type stubs
            'types-requests>=2.31.0',
            'types-setuptools>=68.1.0',
            'types-pyyaml>=6.0.12',
        }
        
        content = [
            "# Dependências de desenvolvimento",
            f"# Projeto: {self.project_config['name']}",
            "#",
            "# Para instalar: pip install -r requirements-dev.txt",
            "# Para Poetry: poetry add --group dev $(cat requirements-dev.txt | grep -v '^#')",
            "",
            "# Ferramentas de desenvolvimento",
        ]
        
        content.extend(sorted(dev_packages))
        
        content.extend([
            "",
            "# CONFIGURAÇÃO DO AMBIENTE DE DESENVOLVIMENTO:",
            "#",
            "# 1. Instale as dependências principais:",
            f"#    pip install -r {self.requirements_file.name}",
            "#",
            "# 2. Instale as dependências de desenvolvimento:",
            f"#    pip install -r {self.dev_requirements_file.name}",
            "#",
            "# 3. Configure pre-commit hooks:",
            "#    pre-commit install",
            "#    pre-commit run --all-files",
            "#",
            "# 4. Para executar testes:",
            "#    pytest",
            "#    pytest --cov=. --cov-report=html",
            "#",
            "# 5. Para verificar qualidade do código:",
            "#    black . --check",
            "#    flake8 .",
            "#    mypy .",
            "",
            "# Para contribuir:",
            "# 1. Faça um fork do repositório",
            "# 2. Crie uma branch para sua feature",
            "# 3. Instale as dependências de desenvolvimento",
            "# 4. Desenvolva e teste suas alterações",
            "# 5. Execute os checks de qualidade de código",
            "# 6. Envie um pull request",
        ])
        
        try:
            self.dev_requirements_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.dev_requirements_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(content))
            print(f"\n📝 requirements-dev.txt criado: {self.dev_requirements_file.relative_to(self.project_root)}")
        except Exception as e:
            print(f"⚠ Não foi possível criar requirements-dev.txt: {e}")
    
    def create_setup_py(self) -> None:
        """Cria um arquivo setup.py básico para o projeto."""
        setup_content = f'''"""
Setup para {self.project_config['name']}.
"""

from setuptools import setup, find_packages

# Ler README
readme_path = Path(__file__).parent / "README.md"
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as fh:
        long_description = fh.read()
else:
    long_description = "{self.project_config['description']}"

# Ler requirements
req_path = Path(__file__).parent / "requirements.txt"
if req_path.exists():
    with open(req_path, "r", encoding="utf-8") as fh:
        requirements = [
            line.strip() 
            for line in fh 
            if line.strip() and not line.startswith("#")
        ]
else:
    requirements = []

setup(
    name="{self.project_config['name']}",
    version="{self.project_config['version']}",
    author="Seu Nome",
    author_email="seu.email@example.com",
    description="{self.project_config['description']}",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/seu-usuario/{self.project_config['name']}",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires="{self.project_config['python_requires']}",
    install_requires=requirements,
    extras_require={{
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ]
    }},
    entry_points={{
        "console_scripts": [
            "{self.project_config['name']}=main:main",
        ]
    }} if Path("main.py").exists() else {{}},
)
'''
        
        try:
            with open(self.setup_file, 'w', encoding='utf-8') as f:
                f.write(setup_content)
            print(f"📦 setup.py criado: {self.setup_file.relative_to(self.project_root)}")
        except Exception as e:
            print(f"⚠ Não foi possível criar setup.py: {e}")
    
    def create_pyproject_toml(self) -> None:
        """Cria um arquivo pyproject.toml para Poetry."""
        if self.pyproject_file.exists():
            print(f"📄 pyproject.toml já existe: {self.pyproject_file.relative_to(self.project_root)}")
            return
        
        pyproject_content = f'''[tool.poetry]
name = "{self.project_config['name']}"
version = "{self.project_config['version']}"
description = "{self.project_config['description']}"
authors = {self.project_config['authors']}
license = "{self.project_config['license']}"
readme = "README.md"
packages = [{{include = "{self.project_config['name']}"}}]

[tool.poetry.dependencies]
python = "{self.project_config['python_requires']}"

# Dependências serão adicionadas via poetry add
# ou copiadas do requirements.txt

[tool.poetry.group.dev.dependencies]
pytest = "^7.4.0"
black = "^23.0.0"
flake8 = "^6.0.0"
mypy = "^1.5.0"
pytest-cov = "^4.1.0"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
'''
        
        try:
            with open(self.pyproject_file, 'w', encoding='utf-8') as f:
                f.write(pyproject_content)
            print(f"📄 pyproject.toml criado: {self.pyproject_file.relative_to(self.project_root)}")
            
            # Sugerir comandos do Poetry
            print("\n🎵 Comandos do Poetry para adicionar dependências:")
            if self.requirements_file.exists():
                print(f"   poetry add $(cat {self.requirements_file.name} | grep -v '^#' | xargs)")
            print("   poetry install")
            print("   poetry update")
            print("   poetry lock")
            
        except Exception as e:
            print(f"⚠ Não foi possível criar pyproject.toml: {e}")
    
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
            print(f"  {module:25s} {count:3d}×  [{files_str}]")
        
        print(f"\n📦 Módulos externos ({len(external_imports)}):")
        for module, count in sorted(external_imports.items(), key=lambda x: (-x[1], x[0])):
            package = self.IMPORT_TO_PACKAGE.get(module, module)
            files = list(module_files.get(module, []))[:3]
            files_str = ", ".join(str(f) for f in files[:2])
            if len(files) > 2:
                files_str += f", ... (+{len(files)-2})"
            print(f"  {module:25s} → {package:25s} {count:3d}×  [{files_str}]")
        
        if local_imports:
            print(f"\n🏠 Módulos locais ({len(local_imports)}):")
            for module, count in sorted(local_imports.items(), key=lambda x: (-x[1], x[0])):
                files = list(module_files.get(module, []))[:3]
                files_str = ", ".join(str(f) for f in files[:2])
                if len(files) > 2:
                    files_str += f", ... (+{len(files)-2})"
                print(f"  {module:25s} {count:3d}×  [{files_str}]")
    
    def _is_stdlib_module(self, module_name: str) -> bool:
        """Verifica se um módulo é da biblioteca padrão."""
        if module_name in self.STDLIB_MODULES:
            return True
        
        # Verificar sub-módulos (como tensorflow.keras -> tensorflow)
        for stdlib_module in self.STDLIB_MODULES:
            if module_name.startswith(stdlib_module + '.'):
                return True
        
        # Verificar se é um subpacote da stdlib
        if '.' in module_name:
            base_module = module_name.split('.')[0]
            if base_module in self.STDLIB_MODULES:
                return True
        
        return False
    
    def run(self, dry_run: bool = False, create_all: bool = False, use_poetry: bool = False) -> None:
        """Executa o scanner completo."""
        print("=" * 80)
        print("📦 SCANNER DE DEPENDÊNCIAS - Versão Genérica")
        print("=" * 80)
        print(f"📂 Diretório raiz: {self.project_root}")
        print(f"📁 Gerenciador detectado: {self.package_manager}")
        print(f"📁 Padrões ignorados: {len(self.ignore_patterns)}")
        print("-" * 80)
        
        # Verificar estrutura do projeto
        print("\n🔍 Verificando estrutura do projeto:")
        print(f"  Nome do projeto: {self.project_config['name']}")
        print(f"  Versão: {self.project_config['version']}")
        print(f"  Python requerido: {self.project_config['python_requires']}")
        
        # Escanear imports
        print("\n" + "=" * 80)
        self.scan_imports()
        
        if not self.imports_found:
            print("\n📭 Nenhum import encontrado. Verifique:")
            print("  • Se há arquivos Python/Jupyter no projeto")
            print("  • Se os padrões de ignore não estão excluindo muitos arquivos")
            print("  • Se o diretório raiz está correto")
            return
        
        # Processar imports para pacotes
        packages = self.process_imports()
        
        if not packages:
            print("\n📭 Nenhuma dependência externa encontrada.")
            print("Isso pode significar que:")
            print("  1. O projeto não usa dependências externas")
            print("  2. Todos os imports são da biblioteca padrão")
            print("  3. Os imports são todos de módulos locais")
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
            
            # Criar pyproject.toml se solicitado
            if use_poetry:
                self.create_pyproject_toml()
            
            # Sugerir próximos passos
            self._show_next_steps(use_poetry)
        
        print("=" * 80)
    
    def _show_next_steps(self, use_poetry: bool = False) -> None:
        """Mostra os próximos passos após a configuração."""
        print("\n🎯 PRÓXIMOS PASSOS:")
        
        if use_poetry or self.package_manager == 'poetry':
            print("\n1. Configurar ambiente com Poetry:")
            print("   poetry install")
            print("   poetry shell")
            print("\n2. Ou instalar dependências manualmente:")
            print(f"   poetry add $(cat {self.requirements_file.name} | grep -v '^#' | xargs)")
            print("\n3. Para desenvolvimento:")
            print(f"   poetry add --group dev $(cat {self.dev_requirements_file.name} | grep -v '^#' | xargs)")
        
        else:
            print("\n1. Crie e ative um ambiente virtual:")
            print("   python -m venv .venv")
            if os.name == 'nt':  # Windows
                print("   .venv\\Scripts\\activate")
            else:  # Unix/Linux/Mac
                print("   source .venv/bin/activate")
            
            print("\n2. Instale as dependências principais:")
            print(f"   pip install -r {self.requirements_file.name}")
            
            print("\n3. Para desenvolvimento, instale também:")
            print(f"   pip install -r {self.dev_requirements_file.name}")
        
        print("\n4. Configure pre-commit hooks:")
        print("   pre-commit install")
        print("   pre-commit run --all-files")
        
        print("\n5. Teste a instalação:")
        print("   python -c \"import sys; print(f'Python {sys.version}')\"")
        print("   python -c \"try:\n    import pandas\n    import sklearn\n    print('✓ Dependências OK')\nexcept ImportError as e:\n    print(f'✗ Erro: {e}')\"")
        
        print("\n6. Verifique se há pacotes desatualizados:")
        print("   pip list --outdated")


def main():
    """Função principal."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Scanner genérico para gerar/atualizar requirements.txt",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  %(prog)s                    # Executa scanner normalmente
  %(prog)s --dry-run          # Mostra o que seria feito sem alterar
  %(prog)s --create-all       # Cria requirements.txt, requirements-dev.txt e setup.py
  %(prog)s --poetry           # Configura para usar Poetry
  %(prog)s --root ../meu-projeto  # Escaneia outro diretório
  %(prog)s --ignore ".test$" --ignore "tmp_"  # Adiciona padrões para ignorar
  %(prog)s --report           # Apenas mostra relatório sem modificar arquivos

Padrões ignorados por padrão:
  • Diretórios: .venv, venv, .env, __pycache__, .git, node_modules, etc.
  • Arquivos: .gitignore, requirements*.txt, setup.py, pyproject.toml, etc.
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
        "--poetry",
        action="store_true",
        help="Configura projeto para usar Poetry (cria pyproject.toml)"
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Diretório raiz do projeto (padrão: diretório atual)"
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
    parser.add_argument(
        "--ignore",
        action="append",
        default=[],
        help="Padrões adicionais para ignorar (pode ser usado múltiplas vezes)"
    )
    
    args = parser.parse_args()
    
    try:
        # Construir lista de padrões para ignorar
        ignore_patterns = RequirementsScanner.DEFAULT_IGNORE_PATTERNS.copy()
        ignore_patterns.extend(args.ignore)
        
        scanner = RequirementsScanner(
            project_root=args.root,
            ignore_patterns=ignore_patterns
        )
        
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
                create_all=args.create_all and not args.no_dev,
                use_poetry=args.poetry
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