# Projeto de Predição de Trajetórias

Projeto desenvolvido para a competição **TE Aprendizado de Máquina** do Kaggle, focado na predição de coordenadas de destino (latitude e longitude) com base em dados de trajetórias GPS.

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
- [Melhorias Implementadas](#melhorias-implementadas)
- [Troubleshooting](#troubleshooting)
- [Contribuindo](#contribuindo)
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
- ✅ **Múltiplos Modelos**: Suporte a 15+ algoritmos de ML
- ✅ **Validação Cruzada Robusta**: 10-fold cross-validation com métrica Haversine
- ✅ **Detecção de Outliers**: Sistema inteligente de detecção e remoção de outliers
- ✅ **Engenharia de Features Avançada**: 30+ features extraídas das trajetórias
- ✅ **Hiperparâmetros Otimizados**: Configurações ajustadas para melhor performance
- ✅ **Ensemble de Modelos**: Suporte a Voting Regressor
- ✅ **Separação de Dados**: Garantia de que train.csv e test.csv são usados corretamente
- ✅ **Submissão Automática**: Integração com Kaggle CLI
- ✅ **Logging Completo**: Sistema de logs detalhado

## 📁 Estrutura do Projeto

```
trajectory-prediction/
├── config/                  # Configurações do projeto
│   ├── __init__.py
│   └── settings.py          # Configurações e variáveis de ambiente
├── data/                    # Dados e processamento
│   ├── __init__.py
│   ├── loader.py            # Carregamento e validação de dados
│   └── downloader.py        # Download de dados do Kaggle
├── features/                # Engenharia de features
│   ├── __init__.py
│   ├── engineering.py       # Extração e criação de features
│   └── outlier_detection.py # Detecção de outliers
├── models/                  # Modelos de ML
│   ├── __init__.py
│   ├── base_model.py        # Classe base para modelos
│   ├── model_factory.py     # Fábrica de modelos
│   └── predictors.py        # Predições
├── training/                # Treinamento e validação
│   ├── __init__.py
│   ├── trainer.py           # Treinador de modelos
│   ├── cross_validation.py  # Validação cruzada (10 folds)
│   └── metrics.py           # Métricas de avaliação
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
│   ├── __init__.py
│   └── scanner.py           # Scanner de dados
├── logs/                    # Arquivos de log
├── submissions/             # Arquivos de submissão gerados
├── reports/                 # Relatórios do pipeline
├── data/                    # Dados brutos
│   ├── train.csv            # Dados de treino
│   └── test.csv             # Dados de teste
├── main.py                  # Script principal
├── requirements.txt         # Dependências Python
├── pyproject.toml          # Configuração Poetry
└── README.md               # Este arquivo
```

## 🔧 Requisitos

- **Python**: >= 3.8 (recomendado 3.13+)
- **Kaggle CLI**: Para download de dados e submissões (opcional)
- **Git**: Para controle de versão

### Dependências Principais

- `scikit-learn` >= 1.0.0
- `pandas` >= 1.3.0
- `numpy` >= 1.21.0
- `xgboost` >= 1.5.0
- `lightgbm` >= 3.3.0
- `catboost` >= 1.0.0 (opcional)

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

Crie um arquivo `.env` na raiz do projeto com as seguintes variáveis:

```env
# Seed para reprodutibilidade
SEED=42

# Credenciais Kaggle (obtenha em https://www.kaggle.com/account)
KAGGLE_USERNAME=seu_usuario
KAGGLE_KEY=sua_chave_api

# Nome da competição
KAGGLE_COMPETITION=te-aprendizado-de-maquina

# Diretório de dados (opcional)
DATA_DIR=data
```

**Como obter credenciais do Kaggle:**
1. Acesse https://www.kaggle.com/account
2. Vá em "API" → "Create New API Token"
3. Baixe o arquivo `kaggle.json`
4. Use `username` e `key` do arquivo JSON

## 🚀 Uso

### Execução Básica

Execute o pipeline completo sem enviar submissão:

```bash
python main.py
```

ou com Poetry:

```bash
poetry run python main.py
```

### Execução com Submissão Automática

Execute o pipeline e envie automaticamente para o Kaggle:

```bash
python main.py --submit
```

### Apenas Enviar Submissão Existente

Envia apenas o último arquivo de submissão gerado (sem executar o pipeline):

```bash
python main.py --submit-only
```

### Com Mensagem Customizada

```bash
python main.py --submit-only -m "Minha mensagem personalizada"
```

### Opções Disponíveis

```bash
python main.py [OPÇÕES]

Opções:
  --submit           Executa pipeline completo e envia submissão para Kaggle
  --submit-only      Apenas envia o último arquivo de submissão
  -m, --message      Mensagem customizada para submissão Kaggle
  --model            Modelo específico para usar (opcional)
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
- **Outliers Geográficos**: Coordenadas inválidas (fora de [-90, 90] lat, [-180, 180] lon)
- **Outliers de Trajetória**: Saltos grandes (>500km) e velocidades impossíveis (>800 km/h)
- **Outliers de Target**: Destinos com coordenadas inválidas
- **Outliers de Features**: Detecção via IQR e Isolation Forest
- **Proteções**: Limite máximo de 20% de remoção para evitar perda excessiva de dados

### 3. Engenharia de Features
- Extração de 30+ features das trajetórias
- Features básicas, de distância, geométricas e direcionais
- Normalização e tratamento de valores faltantes

### 4. Preparação dos Dados
- Separação de features e target
- Normalização com StandardScaler
- **IMPORTANTE**: 
  - `train.csv` → usado para treino e validação cruzada
  - `test.csv` → usado APENAS para predições finais

### 5. Treinamento com Validação Cruzada
- **10-fold cross-validation** no conjunto de treino
- Métrica: Distância Haversine (km)
- Testa múltiplos modelos em paralelo
- Seleciona o melhor modelo baseado na métrica

### 6. Treinamento do Modelo Final
- Treina o melhor modelo em todos os dados de treino
- Usa hiperparâmetros otimizados

### 7. Predição
- Gera predições para `test.csv`
- Valida formato e ranges das predições

### 8. Geração de Submissão
- Cria arquivo CSV no formato do Kaggle
- Valida estrutura e valores
- Salva em `submissions/` com timestamp

### 9. Submissão ao Kaggle (opcional)
- Envia automaticamente via Kaggle CLI
- Registra status da submissão

## 🤖 Modelos

O projeto suporta múltiplos algoritmos de Machine Learning, todos configurados para suportar multi-output (latitude e longitude):

### Modelos Prioritários (usados por padrão)

- **Random Forest**: Ensemble de árvores de decisão
  - `n_estimators`: 200+
  - `max_depth`: 20
  - Otimizado para reduzir overfitting

- **XGBoost**: Gradient boosting otimizado
  - `learning_rate`: 0.05
  - `max_depth`: 7
  - Regularização aumentada

- **LightGBM**: Gradient boosting rápido
  - `learning_rate`: 0.05
  - `max_depth`: 8
  - `num_leaves`: 50

- **Gradient Boosting**: Boosting tradicional (scikit-learn)
  - `learning_rate`: 0.05
  - `max_depth`: 5

- **HistGradientBoosting**: Versão otimizada do scikit-learn
  - Eficiente com grandes datasets

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

### Ensemble

- **EnsembleVoting**: Voting Regressor com MultiOutputRegressor
  - Combina os 3 melhores modelos
  - Suporta multi-output corretamente

## 📊 Features

O projeto extrai **30+ features** das trajetórias:

### Features Básicas
- `start_lat`, `start_lon`: Posição inicial
- `end_lat`, `end_lon`: Posição final do prefixo
- `mean_lat`, `mean_lon`: Médias de latitude e longitude
- `std_lat`, `std_lon`: Desvios padrão
- `median_lat`, `median_lon`: Medianas

### Features de Distância
- `total_distance`: Distância total percorrida (metros)
- `mean_distance`: Distância média entre pontos consecutivos
- `std_distance`: Desvio padrão das distâncias
- `straight_distance`: Distância em linha reta do início ao fim
- `straightness`: Razão entre distância reta e total (0-1)

### Features Geométricas
- `lat_range`: Amplitude de latitude
- `lon_range`: Amplitude de longitude
- `area_bbox`: Área do bounding box
- `aspect_ratio`: Razão aspecto (lat_range / lon_range)
- `centroid_lat`, `centroid_lon`: Centroide da trajetória

### Features Direcionais (NOVAS)
- `bearing`: Direção do início ao fim (graus)
- `bearing_sin`, `bearing_cos`: Versões trigonométricas do bearing
- `direction_variance`: Variabilidade de direção (mede curvas)
- `avg_direction_change`: Mudança média de direção

### Features de Velocidade (NOVAS)
- `avg_speed_ms`: Velocidade média (m/s)
- `avg_speed_kmh`: Velocidade média (km/h)

### Features Estatísticas
- `num_points`: Número de pontos na trajetória
- `density`: Densidade de pontos por distância
- `remaining_distance`: Distância do último ponto ao destino (quando disponível)

## 📈 Validação e Métricas

### Validação Cruzada

- **Método**: K-Fold Cross-Validation
- **Folds**: 10 (aumentado de 5 para maior robustez)
- **Métrica**: Distância Haversine média (km)
- **Dados**: Apenas `train.csv` (nunca `test.csv`)

### Métrica Principal: Distância Haversine

A distância Haversine calcula a distância geodésica entre dois pontos na superfície da Terra:

```
R = 6371 km (raio médio da Terra)
a = sin²(Δφ/2) + cos(φ1) · cos(φ2) · sin²(Δλ/2)
c = 2 · atan2(√a, √(1-a))
d = R · c
```

Onde:
- φ = latitude em radianos
- λ = longitude em radianos
- d = distância em quilômetros

### Separação de Dados

**IMPORTANTE**: O projeto garante a separação correta dos dados:

- ✅ **train.csv**: Usado para treino e validação cruzada
- ✅ **test.csv**: Usado APENAS para predições finais
- ❌ **test.csv NUNCA** é usado em treino ou validação

## 📂 Estrutura de Diretórios

- **`data/`**: Dados brutos e processados
  - `train.csv`: Dados de treino (com destino conhecido)
  - `test.csv`: Dados de teste (sem destino)
  - `sample_submission.csv`: Formato de submissão

- **`logs/`**: Arquivos de log do pipeline
  - `pipeline.log`: Log principal
  - Logs específicos por módulo

- **`submissions/`**: Arquivos de submissão gerados
  - Formato: `submission_<MODELO>_<TIMESTAMP>.csv`
  - Validados antes de salvar

- **`reports/`**: Relatórios gerados pelo pipeline
  - `pipeline_report.txt`: Relatório de execução
  - Estatísticas de outliers removidos
  - Performance dos modelos

## 🚀 Melhorias Implementadas

### Versão Atual (Última Atualização)

1. **Detecção de Outliers Melhorada**
   - Parâmetros mais conservadores (max_jump: 500km, max_speed: 800km/h)
   - Limite de 20% de remoção (antes 50%)
   - Detecção por múltiplos critérios com proteções

2. **Validação Cruzada Robusta**
   - Aumentado de 5 para 10 folds
   - Módulo dedicado (`cross_validation.py`)
   - Métrica Haversine vetorizada para performance

3. **Correção de Modelos Multi-Output**
   - LightGBM, GradientBoosting e HistGradientBoosting agora usam MultiOutputRegressor
   - EnsembleVoting corrigido para suportar multi-output

4. **Novas Features**
   - Features direcionais (bearing, direction_variance)
   - Features de velocidade (avg_speed_ms, avg_speed_kmh)
   - Features de mudança de direção

5. **Hiperparâmetros Otimizados**
   - Learning rates reduzidos (0.05 vs 0.1)
   - Regularização aumentada
   - Mais estimadores (200+)
   - Parâmetros ajustados para reduzir overfitting

6. **Garantia de Separação de Dados**
   - Logs explícitos sobre uso de train.csv vs test.csv
   - Validação cruzada usa apenas train.csv

## 🐛 Troubleshooting

### Erro ao baixar dados do Kaggle
- Verifique se as credenciais estão configuradas no `.env`
- Execute `kaggle configure` manualmente
- Verifique se você aceitou os termos da competição no Kaggle

### Erro de memória
- Reduza o número de modelos testados
- Use `priority_only=True` no código
- Processe os dados em lotes menores

### Dependências não encontradas
```bash
pip install --upgrade -r requirements.txt
```

### Erro "y should be a 1d array"
- Verifique se os modelos estão usando MultiOutputRegressor quando necessário
- Isso foi corrigido na versão atual

### Erro no EnsembleVoting
- O EnsembleVoting agora usa MultiOutputRegressor corretamente
- Se persistir, verifique se os modelos base suportam multi-output

## 🔍 Logs

Os logs são salvos em `logs/` e incluem:
- Informações sobre carregamento de dados
- Detecção e remoção de outliers
- Progresso do treinamento (fold por fold)
- Métricas de validação cruzada
- Erros e avisos
- Estatísticas finais

## 📝 Exemplo de Saída

```
============================================================
TRAJECTORY PREDICTION - te-aprendizado-de-maquina
============================================================

1. CARREGANDO DADOS
   • Train: 12050 amostras
   • Test: 3013 amostras

2. DETECCAO DE OUTLIERS
   • Total: 83 (0.69%)
   • Amostras limpas: 11967

3. ENGENHARIA DE FEATURES
   Features extraidas: 30 features

4. PREPARANDO DADOS
   ⚠️  VALIDAÇÃO CRUZADA: Usando apenas dados de TREINO (train.csv)
   • X_train: (10281, 30)
   • X_test: (3013, 30)

5. TREINANDO MODELO
   📊 RANKING DE MODELOS:
     1. RandomForest: 413.81 ± 28.63 km
     2. GradientBoosting: 431.41 ± 28.98 km
     ...

6. FAZENDO PREDICOES
   ⚠️  Usando test.csv APENAS para predições finais

7. SALVANDO SUBMISSAO
   ✅ Arquivo salvo: submissions/submission_RandomForest_20251218_215101.csv
```

## 🤝 Contribuindo

1. Faça um fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/MinhaFeature`)
3. Commit suas mudanças (`git commit -m 'Adiciona MinhaFeature'`)
4. Push para a branch (`git push origin feature/MinhaFeature`)
5. Abra um Pull Request

## 📝 Licença

Este projeto foi desenvolvido para fins educacionais e de competição.

## 👤 Autor

**Lucas Cavalcante dos Santos**
- Email: cavalcanteprofissional@outlook.com

## 📚 Referências

- [Kaggle Competitions](https://www.kaggle.com/competitions)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [Haversine Formula](https://en.wikipedia.org/wiki/Haversine_formula)

## 🎯 Próximos Passos

Melhorias futuras planejadas:
- [ ] Feature engineering com coordenadas relativas
- [ ] Neural Networks (MLP) com mais camadas
- [ ] Stacking com meta-learner
- [ ] GridSearch para hiperparâmetros
- [ ] Validação temporal (se houver ordem temporal)
- [ ] Pós-processamento de predições

---

**Desenvolvido para a competição Tópicos Especiais em Aprendizado de Máquina - Kaggle**
[Universidade Federal do Ceará (UFC)](https://www.ufc.br/)
[Departamento de Computação (DC)](https://dc.ufc.br/pt/)
[Capacitação Técnica e Empreendedora em IA (CTE-IA)](https://www.cteia.dc.ufc.br/)

*Última atualização: Dezembro 2025*
