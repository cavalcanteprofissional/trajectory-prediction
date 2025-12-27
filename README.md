# Projeto de Predição de Trajetórias

Projeto desenvolvido para a competição **Tópicos Especiais em Aprendizado de Máquina** do Kaggle, focado na predição de coordenadas de destino (latitude e longitude) com base em dados de trajetórias GPS.

## 📋 Índice

- [Sobre o Projeto](#sobre-o-projeto)
- [Características Principais](#características-principais)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Requisitos](#requisitos)
- [Instalação](#instalação)
- [Configuração](#configuração)
- [Uso](#uso)
- [Pipeline](#pipeline)
- [Modelos](#modelos)
- [Features](#features)
- [Validação e Métricas](#validação-e-métricas)
- [Estrutura de Diretórios](#estrutura-de-diretórios)
- [Troubleshooting](#troubleshooting)
- [Autor](#autor)

## 🎯 Sobre o Projeto

Este projeto implementa um pipeline completo de Machine Learning para predição de trajetórias GPS, utilizando múltiplos algoritmos de aprendizado supervisionado para prever coordenadas geográficas finais (destino) com base em dados históricos de trajetórias.

### Objetivo

Prever as coordenadas de destino (`dest_lat`, `dest_lon`) de trajetórias com base em:
- Dados de caminho percorrido (`path_lat`, `path_lon`) - apenas o prefixo inicial da trajetória
- Features extraídas da trajetória (espaciais, temporais e geométricas)
- Múltiplos modelos de regressão com validação cruzada robusta

### Métrica de Avaliação

O projeto utiliza a **Distância Haversine** (em quilômetros) como métrica principal, calculando a distância geodésica entre as coordenadas preditas e reais na superfície da Terra.

## ✨ Características Principais

- ✅ **Pipeline Completo**: Do carregamento de dados à geração de submissão
- ✅ **Múltiplos Modelos**: Suporte a 16+ algoritmos de ML
- ✅ **Validação Cruzada Robusta**: 5-fold cross-validation com métrica Haversine
- ✅ **Detecção de Outliers**: Sistema inteligente de detecção e remoção de outliers
- ✅ **Engenharia de Features Avançada**: 30+ features extraídas das trajetórias
- ✅ **Ensemble de Modelos**: Suporte a Voting Regressor e Bagging
- ✅ **Separação de Dados**: Garantia de que train.csv e test.csv são usados corretamente
- ✅ **Submissão Automática**: Integração com Kaggle CLI
- ✅ **Logging Completo**: Sistema de logs detalhado
- ✅ **Otimização de Hiperparâmetros**: Suporte a Optuna para GradientBoosting

## 📁 Estrutura do Projeto

```
trajectory-prediction/
├── config/                  # Configurações do projeto
│   ├── __init__.py
│   └── settings.py          # Configurações e variáveis de ambiente
├── data/                    # Dados e processamento
│   ├── __init__.py
│   ├── loader.py            # Carregamento e validação de dados
│   ├── downloader.py        # Download de dados do Kaggle
│   ├── train.csv            # Dados de treino
│   └── test.csv             # Dados de teste
├── features/                # Engenharia de features
│   ├── __init__.py
│   ├── engineering.py       # Extração e criação de features
│   ├── outlier_detection.py # Detecção de outliers
│   ├── augmentation.py      # Aumento de dados
│   ├── cleaning.py          # Limpeza de dados
│   └── clustering.py        # Clustering de trajetórias
├── models/                  # Modelos de ML
│   ├── __init__.py
│   ├── base_model.py        # Classe base para modelos
│   ├── model_factory.py     # Fábrica de modelos
│   └── predictors.py        # Predições
├── training/                # Treinamento e validação
│   ├── __init__.py
│   ├── trainer.py           # Treinador de modelos
│   └── cross_validation.py  # Validação cruzada
├── evaluation/              # Avaliação
│   ├── __init__.py
│   ├── metrics.py           # Métricas de avaliação
│   └── visualization.py     # Visualizações
├── submission/              # Geração de submissões
│   ├── __init__.py
│   └── generator.py         # Gerador de arquivos de submissão
├── utils/                   # Utilitários
│   ├── __init__.py
│   └── logger.py            # Sistema de logging
├── tools/                   # Ferramentas auxiliares
├── scripts/                 # Scripts de otimização
├── logs/                    # Arquivos de log
├── submissions/             # Arquivos de submissão gerados
├── reports/                 # Relatórios do pipeline
├── main.py                  # Script principal
├── requirements.txt         # Dependências Python
├── pyproject.toml          # Configuração Poetry
└── README.md               # Este arquivo
```

## 🔧 Requisitos

- **Python**: >= 3.13 (recomendado)
- **Kaggle CLI**: Para download de dados e submissões (opcional)
- **Git**: Para controle de versão

### Dependências Principais

- `scikit-learn` >= 1.3.0
- `pandas` >= 2.0.0
- `numpy` >= 1.24.0
- `xgboost` >= 1.7.0
- `lightgbm` >= 3.3.0
- `catboost` >= 1.0.0
- `optuna` >= 4.6.0
- `folium` >= 0.20.0
- `geopy` >= 2.4.1

## 📦 Instalação

### 1. Clone o repositório

```bash
git clone <url-do-repositorio>
cd trajectory-prediction
```

### 2. Crie um ambiente virtual (recomendado)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 3. Instale as dependências

**Opção A: Usando pip**
```bash
pip install -r requirements.txt
```

**Opção B: Usando Poetry** (recomendado)
```bash
poetry install
poetry shell
```

### 4. Instale o Kaggle CLI (opcional)

```bash
pip install kaggle
kaggle configure
```

## ⚙️ Configuração

### Variáveis de Ambiente

O projeto usa um arquivo `.env` para armazenar configurações sensíveis. Crie um arquivo `.env` na raiz do projeto:

```env
# Seed para reprodutibilidade
SEED=42

# Credenciais Kaggle (obtenha em https://www.kaggle.com/account)
KAGGLE_USERNAME=seu_usuario_kaggle
KAGGLE_KEY=sua_chave_api_kaggle

# Nome da competição
KAGGLE_COMPETITION=topicos-especiais-em-aprendizado-de-maquina-v2

# Diretório de dados (opcional)
DATA_DIR=data
```

**Como obter credenciais do Kaggle:**
1. Acesse https://www.kaggle.com/account
2. Vá em "API" → "Create New API Token"
3. Use `username` e `key` do arquivo JSON baixado

## 🚀 Uso

### Execução Básica

Execute o pipeline completo:

```bash
python main.py
```

### Execução com Submissão Automática

Execute o pipeline e envie automaticamente para o Kaggle:

```bash
python main.py --submit
```

### Apenas Enviar Submissão Existente

Envia apenas o último arquivo de submissão gerado:

```bash
python main.py --submit-only -m "Minha mensagem personalizada"
```

### Usar Ensemble

Execute com ensemble de modelos:

```bash
python main.py --ensemble
```

### Opções Disponíveis

```bash
python main.py [OPÇÕES]

Opções:
  --submit           Executa pipeline completo e envia submissão para Kaggle
  --submit-only      Apenas envia o último arquivo de submissão
  -m, --message      Mensagem customizada para submissão Kaggle
  --ensemble         Usa ensemble avançado de modelos
  -h, --help         Mostra ajuda
```

## 🔄 Pipeline

O pipeline executa as seguintes etapas em ordem:

### 1. Carregamento de Dados
- Verifica se os dados existem localmente
- Faz download automático do Kaggle se necessário
- Carrega `train.csv` e `test.csv`
- Valida integridade dos dados
- Parse de listas de coordenadas

### 2. Detecção de Outliers
- **Outliers Geográficos**: Coordenadas inválidas
- **Outliers de Trajetória**: Saltos grandes e velocidades impossíveis
- **Outliers de Target**: Destinos com coordenadas inválidas
- **Proteções**: Limite máximo de remoção para evitar perda excessiva de dados

### 3. Engenharia de Features
- Extração de 30+ features das trajetórias
- Features básicas, de distância, geométricas e direcionais
- Normalização e tratamento de valores faltantes

### 4. Preparação dos Dados
- Separação de features e target
- Normalização com StandardScaler
- **IMPORTANTE**: `train.csv` usado para treino/validação, `test.csv` apenas para predições

### 5. Treinamento com Validação Cruzada
- **5-fold cross-validation** no conjunto de treino
- Métrica: Distância Haversine média (km)
- Testa múltiplos modelos em paralelo
- Seleciona o melhor modelo baseado na métrica

### 6. Treinamento do Modelo Final
- Treina o melhor modelo em todos os dados de treino
- Usa hiperparâmetros otimizados (Optuna para GradientBoosting)

### 7. Predição
- Gera predições para `test.csv`
- Valida formato e ranges das predições

### 8. Geração de Submissão
- Cria arquivo CSV no formato do Kaggle
- Salva em `submissions/` com timestamp

### 9. Submissão ao Kaggle (opcional)
- Envia automaticamente via Kaggle CLI
- Registra status da submissão

## 🤖 Modelos

O projeto suporta 16+ algoritmos de Machine Learning:

### Modelos Prioritários

- **RandomForest**: Ensemble de árvores de decisão
- **XGBoost**: Gradient boosting otimizado
- **LightGBM**: Gradient boosting rápido
- **GradientBoosting**: Boosting tradicional (com otimização Optuna)
- **HistGradientBoosting**: Versão otimizada do scikit-learn

### Outros Modelos Disponíveis

- CatBoost
- Extra Trees
- Ridge Regression
- Lasso Regression
- Elastic Net
- Bayesian Ridge
- K-Nearest Neighbors (KNN)
- Support Vector Regression (SVR)
- Multi-Layer Perceptron (MLP)
- AdaBoost
- Bagged Gradient Boosting

### Ensemble

- **Ensemble Avançado**: Combinação de GradientBoosting otimizado + RandomForest
- **BaggedGB**: Bagging com GradientBoosting base

## 📊 Features

O projeto extrai **30+ features** das trajetórias:

### Features Básicas
- `start_lat`, `start_lon`: Posição inicial
- `end_lat`, `end_lon`: Posição final do prefixo
- `mean_lat`, `mean_lon`: Médias de latitude e longitude
- `std_lat`, `std_lon`: Desvios padrão

### Features de Distância
- `total_distance`: Distância total percorrida (metros)
- `mean_distance`: Distância média entre pontos
- `straight_distance`: Distância em linha reta
- `straightness`: Razão entre distância reta e total

### Features Geométricas
- `lat_range`, `lon_range`: Amplitude das coordenadas
- `area_bbox`: Área do bounding box
- `aspect_ratio`: Razão aspecto
- `centroid_lat`, `centroid_lon`: Centroide da trajetória

### Features Direcionais
- `bearing`: Direção do início ao fim (graus)
- `bearing_sin`, `bearing_cos`: Versões trigonométricas
- `direction_variance`: Variabilidade de direção

## 📈 Validação e Métricas

### Validação Cruzada

- **Método**: K-Fold Cross-Validation
- **Folds**: 5
- **Métrica**: Distância Haversine média (km)
- **Dados**: Apenas `train.csv`

### Métrica Principal: Distância Haversine

Calcula a distância geodésica entre dois pontos na Terra usando a fórmula:

```
d = 2R · arcsin(√(sin²(Δφ/2) + cos(φ₁)cos(φ₂)sin²(Δλ/2)))
```

Onde R = 6371 km (raio médio da Terra).

### Separação de Dados

**CRÍTICO**: Garantia de separação correta:
- ✅ `train.csv`: Treino e validação cruzada
- ✅ `test.csv`: Apenas predições finais
- ❌ `test.csv` NUNCA usado em treino/validação

## 📂 Estrutura de Diretórios

- **`data/`**: Dados brutos (`train.csv`, `test.csv`)
- **`logs/`**: Arquivos de log (`pipeline.log`)
- **`submissions/`**: Arquivos de submissão gerados
- **`reports/`**: Relatórios (`pipeline_report.txt`, resultados Optuna)
- **`scripts/`**: Scripts de otimização (Optuna)
- **`models/`**: Implementações de modelos
- **`features/`**: Engenharia de features
- **`training/`**: Lógica de treinamento
- **`evaluation/`**: Métricas e visualizações

## 🐛 Troubleshooting

### Erro ao baixar dados do Kaggle
- Verifique credenciais no `.env`
- Execute `kaggle configure` manualmente

### Dependências não encontradas
```bash
pip install --upgrade -r requirements.txt
```

### Erro de memória
- Reduza número de modelos testados
- Processe dados em lotes menores

### Erro no Ensemble
- Verifique se modelos base suportam multi-output

## 👤 Autor

**Lucas Cavalcante dos Santos**
- Email: cavalcanteprofissional@outlook.com

## 📚 Referências

- [Kaggle Competitions](https://www.kaggle.com/competitions)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Haversine Formula](https://en.wikipedia.org/wiki/Haversine_formula)

---

**Desenvolvido para a competição Tópicos Especiais em Aprendizado de Máquina - Kaggle**  
[Universidade Federal do Ceará (UFC)](https://www.ufc.br/)  
[Departamento de Computação (DC)](https://dc.ufc.br/pt/)  
[Capacitação Técnica e Empreendedora em IA (CTE-IA)](https://www.cteia.dc.ufc.br/)  

*Última atualização: Dezembro 2025*