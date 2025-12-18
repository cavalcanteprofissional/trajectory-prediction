# Projeto de Predição de Trajetórias

Projeto desenvolvido para a competição **TE Aprendizado de Máquina** do Kaggle, focado na predição de coordenadas de destino (latitude e longitude) com base em dados de trajetórias.

## 📋 Índice

- [Sobre o Projeto](#sobre-o-projeto)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Requisitos](#requisitos)
- [Instalação](#instalação)
- [Configuração](#configuração)
- [Uso](#uso)
- [Pipeline](#pipeline)
- [Modelos](#modelos)
- [Features](#features)
- [Estrutura de Diretórios](#estrutura-de-diretórios)
- [Contribuindo](#contribuindo)
- [Autor](#autor)

## 🎯 Sobre o Projeto

Este projeto implementa um pipeline completo de Machine Learning para predição de trajetórias, utilizando múltiplos algoritmos de aprendizado supervisionado para prever coordenadas geográficas finais (destino) com base em dados históricos de trajetórias.

### Objetivo

Prever as coordenadas de destino (`dest_lat`, `dest_lon`) de trajetórias com base em:
- Dados de caminho percorrido (`path_lat`, `path_lon`)
- Features extraídas da trajetória
- Múltiplos modelos de regressão

## 📁 Estrutura do Projeto

```
trajectory_prediction_project/
├── config/              # Configurações do projeto
│   └── settings.py      # Configurações e variáveis de ambiente
├── data/                # Dados e processamento
│   ├── loader.py        # Carregamento de dados
│   ├── downloader.py    # Download de dados do Kaggle
│   └── processed/       # Dados processados
├── features/            # Engenharia de features
│   └── engineering.py   # Extração e criação de features
├── models/              # Modelos de ML
│   ├── model_factory.py # Fábrica de modelos
│   ├── base_model.py    # Classe base para modelos
│   └── predictors.py    # Predições
├── training/            # Treinamento
│   ├── trainer.py       # Treinador de modelos
│   ├── cross_validation.py  # Validação cruzada
│   └── metrics.py       # Métricas de avaliação
├── evaluation/          # Avaliação
│   ├── metrics.py       # Métricas de avaliação
│   └── visualization.py # Visualizações
├── submission/          # Geração de submissões
│   └── generator.py     # Gerador de arquivos de submissão
├── utils/               # Utilitários
│   └── logger.py        # Sistema de logging
├── logs/                # Arquivos de log
├── submissions/         # Arquivos de submissão gerados
├── reports/            # Relatórios do pipeline
├── main.py             # Script principal
├── requirements.txt    # Dependências Python
└── pyproject.toml     # Configuração Poetry
```

## 🔧 Requisitos

- **Python**: >= 3.8 (recomendado 3.13+)
- **Kaggle CLI**: Para download de dados e submissões (opcional)
- **Git**: Para controle de versão

## 📦 Instalação

### 1. Clone o repositório

```bash
git clone <url-do-repositorio>
cd trajectory_prediction_project
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

**Opção B: Usando Poetry**
```bash
poetry install
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

O pipeline executa as seguintes etapas:

1. **Carregamento de Dados**
   - Verifica se os dados existem localmente
   - Faz download automático do Kaggle se necessário
   - Carrega dados de treino e teste

2. **Engenharia de Features**
   - Extrai features básicas (posição inicial, final, médias)
   - Calcula distâncias e velocidades
   - Gera features estatísticas da trajetória
   - Prepara dados para treinamento

3. **Preparação dos Dados**
   - Normalização e tratamento de valores faltantes
   - Separação de features e target
   - Preparação para validação cruzada

4. **Treinamento**
   - Treina múltiplos modelos com validação cruzada (5 folds)
   - Seleciona o melhor modelo baseado em métricas
   - Treina modelo final com todos os dados

5. **Predição**
   - Gera predições para dados de teste
   - Valida formato das predições

6. **Geração de Submissão**
   - Cria arquivo CSV no formato do Kaggle
   - Salva em `submissions/` com timestamp

7. **Submissão ao Kaggle** (opcional)
   - Envia automaticamente via Kaggle CLI

## 🤖 Modelos

O projeto suporta múltiplos algoritmos de Machine Learning:

### Modelos Prioritários (usados por padrão)
- **Random Forest**: Ensemble de árvores de decisão
- **XGBoost**: Gradient boosting otimizado
- **LightGBM**: Gradient boosting rápido e eficiente
- **Gradient Boosting**: Boosting tradicional
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

### Ensemble
O projeto também suporta ensemble de modelos usando Voting Regressor.

## 📊 Features

As features extraídas incluem:

### Features Básicas
- Posição inicial (`start_lat`, `start_lon`)
- Posição final (`end_lat`, `end_lon`)
- Médias de latitude e longitude
- Desvios padrão de latitude e longitude

### Features de Distância
- Distância total percorrida
- Distância Haversine entre pontos consecutivos
- Distância média entre pontos

### Features de Velocidade
- Velocidade média
- Velocidade máxima
- Aceleração média

### Features Estatísticas
- Número de pontos na trajetória
- Duração estimada
- Features derivadas de estatísticas descritivas

## 📂 Estrutura de Diretórios

- **`data/`**: Dados brutos e processados
  - `train.csv`: Dados de treino
  - `test.csv`: Dados de teste
  - `processed/`: Dados processados

- **`logs/`**: Arquivos de log do pipeline
  - `pipeline.log`: Log principal
  - Logs específicos por módulo

- **`submissions/`**: Arquivos de submissão gerados
  - Formato: `submission_<MODELO>_<TIMESTAMP>.csv`

- **`reports/`**: Relatórios gerados pelo pipeline
  - `pipeline_report.txt`: Relatório de execução

## 📈 Métricas

O projeto utiliza a métrica **Haversine Distance** (distância em quilômetros) para avaliar os modelos, que calcula a distância geodésica entre as coordenadas preditas e reais.

## 🔍 Logs

Os logs são salvos em `logs/` e incluem:
- Informações sobre carregamento de dados
- Progresso do treinamento
- Métricas de validação
- Erros e avisos

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

---

**Desenvolvido para a competição TE Aprendizado de Máquina - Kaggle**

