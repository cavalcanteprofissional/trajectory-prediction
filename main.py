# main.py
import warnings
warnings.filterwarnings('ignore')

import sys
import traceback
from pathlib import Path
import pandas as pd
import numpy as np
import subprocess
import os
import argparse

def initialize_project():
    """Inicializa o projeto"""
    ROOT_DIR = Path(__file__).parent
    sys.path.insert(0, str(ROOT_DIR))
    
    try:
        from config.settings import config
        print(f"Diretorio raiz: {config.ROOT_DIR}")
        print(f"Dados: {config.DATA_DIR}")
        print(f"Competicao: {config.KAGGLE_COMPETITION}")
        return config
        
    except ImportError as e:
        print(f"Erro ao carregar configuracoes: {e}")
        
        # Configuração básica como fallback
        ROOT_DIR = Path(__file__).parent
        
        directories = ['data', 'logs', 'models', 'submissions', 'reports']
        for dir_name in directories:
            dir_path = ROOT_DIR / dir_name
            dir_path.mkdir(exist_ok=True)
            print(f"Criado: {dir_path}")
        
        return type('Config', (), {
            'ROOT_DIR': ROOT_DIR, 
            'DATA_DIR': ROOT_DIR / 'data',
            'LOGS_DIR': ROOT_DIR / 'logs',
            'SUBMISSIONS_DIR': ROOT_DIR / 'submissions',
            'KAGGLE_COMPETITION': 'te-aprendizado-de-maquina'
        })()

def setup_logging():
    """Configura logging sem emojis"""
    import logging
    
    # Criar diretório de logs
    logs_dir = Path('logs')
    logs_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(logs_dir / 'pipeline.log', encoding='utf-8')
        ]
    )
    
    # Reduzir verbosidade
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)

def check_kaggle_cli():
    """Verifica se o Kaggle CLI está instalado e configurado"""
    try:
        result = subprocess.run(["kaggle", "--version"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Kaggle CLI detectado: {result.stdout.strip()}")
            return True
        else:
            print("⚠️  Kaggle CLI encontrado mas com erro:")
            print(result.stderr)
            return False
    except FileNotFoundError:
        print("❌ Kaggle CLI não encontrado.")
        print("   Instale com: pip install kaggle")
        print("   Configure com: kaggle configure")
        return False
    except Exception as e:
        print(f"❌ Erro ao verificar Kaggle CLI: {e}")
        return False

def submit_to_kaggle_automatically(submission_file, model_name, competition_name, logger):
    """Envia submissão automaticamente para Kaggle via linha de comando"""
    try:
        # Mensagem de commit
        commit_message = f"Submissao automatica - Modelo: {model_name} - {competition_name}"
        
        # Verificar se Kaggle CLI está instalado
        if not check_kaggle_cli():
            logger.warning("Kaggle CLI não está disponível para submissão automática")
            return False
        
        # Comando para submissão
        cmd = [
            "kaggle", "competitions", "submit",
            "-c", competition_name,
            "-f", str(submission_file),
            "-m", commit_message
        ]
        
        logger.info(f"Enviando submissão para Kaggle...")
        logger.info(f"Comando: {' '.join(cmd)}")
        
        # Executar submissão
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("✅ Submissão enviada com sucesso!")
            logger.info(f"Arquivo: {submission_file}")
            logger.info(f"Mensagem: {commit_message}")
            
            # Verificar status
            try:
                status_cmd = ["kaggle", "competitions", "submissions", "-c", competition_name]
                status_result = subprocess.run(status_cmd, capture_output=True, text=True)
                if status_result.returncode == 0:
                    logger.info("Últimas submissões:")
                    # Pegar apenas as primeiras linhas para não poluir o log
                    lines = status_result.stdout.strip().split('\n')
                    for line in lines[:3]:
                        logger.info(f"  {line}")
            except Exception as status_error:
                logger.warning(f"Não foi possível verificar status: {status_error}")
            
            return True
        else:
            logger.error(f"❌ Erro ao enviar submissão:")
            logger.error(f"Stderr: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Erro inesperado na submissão automática: {e}")
        return False

def run_data_pipeline(logger, config, auto_submit=False):
    """Executa o pipeline completo de dados"""
    logger.info("=" * 60)
    logger.info("PIPELINE DE DADOS - TE APRENDIZADO DE MAQUINA")
    logger.info("=" * 60)
    
    try:
        # 1. Carregar dados
        logger.info("\n1. CARREGANDO DADOS")
        
        from data.loader import DataLoader
        loader = DataLoader()
        
        # Verificar se dados existem
        data_exists = loader.ensure_data_exists(download_if_missing=True)
        
        if not data_exists:
            logger.warning("Nao foi possivel obter dados reais, usando dados de exemplo")
        
        # Carregar dados (tenta reais primeiro, depois exemplo)
        train_data, test_data = loader.load_data(use_sample_if_missing=True)
        
        summary = loader.get_data_summary()
        logger.info(f"Dados carregados:")
        logger.info(f"   • Train: {summary['train_samples']} amostras")
        logger.info(f"   • Test: {summary['test_samples']} amostras")
        logger.info(f"   • Tem target: {summary['has_target']}")
        
        # Mostrar primeiras linhas
        logger.info(f"\nPrimeira linha do train:")
        logger.info(f"   ID: {train_data.iloc[0]['trajectory_id']}")
        
        if 'dest_lat' in train_data.columns:
            logger.info(f"   Destino: ({train_data.iloc[0]['dest_lat']:.6f}, {train_data.iloc[0]['dest_lon']:.6f})")
        
        # 2. Engenharia de features
        logger.info("\n2. ENGENHARIA DE FEATURES")
        
        from features.engineering import FeatureEngineer
        feature_engineer = FeatureEngineer()
        
        train_features = feature_engineer.extract_all_features(train_data)
        test_features = feature_engineer.extract_all_features(test_data)
        
        if 'dest_lat' in train_data.columns:
            train_features['dest_lat'] = train_data['dest_lat'].values
            train_features['dest_lon'] = train_data['dest_lon'].values
        
        logger.info(f"Features extraidas: {train_features.shape[1]} features")
        
        # 3. Preparar dados para treinamento
        logger.info("\n3. PREPARANDO DADOS")
        
        prepared_data = feature_engineer.prepare_features_for_training(
            train_features, test_features
        )
        
        logger.info(f"Dados preparados:")
        logger.info(f"   • X_train: {prepared_data['X_train'].shape}")
        logger.info(f"   • X_test: {prepared_data['X_test'].shape}")
        
        if 'y_train' in prepared_data:
            logger.info(f"   • y_train: {prepared_data['y_train'].shape}")
        
        # 4. Treinar modelo
        logger.info("\n4. TREINANDO MODELO")

        from training.trainer import ModelTrainer
        trainer = ModelTrainer()

        # Criar modelos com parâmetros otimizados
        from models.model_factory import ModelFactory
        model_factory = ModelFactory(n_samples=len(prepared_data['X_train']))

        # Usar apenas modelos prioritários para velocidade
        models = model_factory.create_all_models(
            priority_only=True,  # Usa apenas modelos prioritários
            include_ensemble=True,  # Inclui ensemble
            n_features=prepared_data['X_train'].shape[1]
        )

        # Treinar com validação cruzada
        results = trainer.train_all_models(
            prepared_data['X_train'],
            prepared_data['y_train'],
            models,
            cv_folds=5
        )
        
        # Treinar modelo final
        final_model_info = trainer.train_final_model(
            prepared_data['X_train'],
            prepared_data['y_train']
        )
        
        logger.info(f"Modelo treinado: {final_model_info['model_name']}")
        
        # 5. Fazer predições
        logger.info("\n5. FAZENDO PREDICOES")
        
        final_model = final_model_info['model']
        predictions = final_model.predict(prepared_data['X_test'])
        
        logger.info(f"{len(predictions)} predicoes geradas")
        
        # 6. Salvar submissão
        logger.info("\n6. SALVANDO SUBMISSAO")
        
        from submission.generator import SubmissionGenerator
        submission_gen = SubmissionGenerator()
        
        submission_file = submission_gen.generate_submission(
            test_ids=test_data['trajectory_id'].values,
            predictions=predictions,
            model_name=final_model_info['model_name'],
            description=f"Modelo {final_model_info['model_name']} - {config.KAGGLE_COMPETITION}"
        )
        
        logger.info(f"Submissao salva: {submission_file}")
        
        # 7. Submissão automática ao Kaggle (se solicitado)
        submission_success = False
        if auto_submit:
            logger.info("\n7. ENVIANDO SUBMISSAO AO KAGGLE")
            submission_success = submit_to_kaggle_automatically(
                submission_file,
                final_model_info['model_name'],
                config.KAGGLE_COMPETITION,
                logger
            )
            
            if submission_success:
                logger.info("✅ Submissao enviada ao Kaggle com sucesso!")
            else:
                logger.warning("⚠️  Falha ao enviar submissao ao Kaggle")
        
        # 8. Mostrar estatísticas
        print("\n" + "=" * 60)
        print("ESTATISTICAS FINAIS")
        print("=" * 60)
        print(f"Modelo usado: {final_model_info['model_name']}")
        print(f"Total de predicoes: {len(predictions)}")
        print(f"Range Latitude: [{predictions[:, 0].min():.6f}, {predictions[:, 0].max():.6f}]")
        print(f"Range Longitude: [{predictions[:, 1].min():.6f}, {predictions[:, 1].max():.6f}]")
        
        if trainer.best_model_info:
            print(f"\nMelhor modelo na validacao: {trainer.best_model_info['model_name']}")
            print(f"   Erro medio: {trainer.best_model_info['mean_error']:.4f} km")
        
        print("\nPrimeiras 5 predicoes:")
        preview_df = pd.DataFrame({
            'trajectory_id': test_data['trajectory_id'].values[:5],
            'latitude': predictions[:5, 0],
            'longitude': predictions[:5, 1]
        })
        print(preview_df.to_string(index=False))
        
        # 9. Salvar relatório
        report_file = config.ROOT_DIR / 'reports' / 'pipeline_report.txt'
        report_file.parent.mkdir(exist_ok=True)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"Relatorio do Pipeline - {config.KAGGLE_COMPETITION}\n")
            f.write("=" * 50 + "\n")
            f.write(f"Data: {pd.Timestamp.now()}\n")
            f.write(f"Modelo final: {final_model_info['model_name']}\n")
            f.write(f"Amostras de treino: {len(train_data)}\n")
            f.write(f"Amostras de teste: {len(test_data)}\n")
            f.write(f"Features: {train_features.shape[1]}\n")
            f.write(f"Arquivo de submissao: {submission_file}\n")
            f.write(f"Submissao enviada: {'SIM' if submission_success else 'NAO'}\n")
            
            if trainer.best_model_info:
                f.write(f"Melhor erro na validacao: {trainer.best_model_info['mean_error']:.4f} km\n")
        
        logger.info(f"Relatorio salvo: {report_file}")
        
        return {
            'success': True,
            'submission_file': submission_file,
            'model_name': final_model_info['model_name'],
            'predictions': predictions,
            'submitted': submission_success,
            'best_model_error': trainer.best_model_info['mean_error'] if trainer.best_model_info else None
        }
        
    except Exception as e:
        logger.error(f"Erro no pipeline: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e)
        }

def get_latest_submission():
    """Obtém o arquivo de submissão mais recente"""
    submissions_dir = Path("submissions")
    if not submissions_dir.exists():
        return None
    
    csv_files = list(submissions_dir.glob("submission_*.csv"))
    if not csv_files:
        return None
    
    # Ordenar por data de modificação (mais recente primeiro)
    csv_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return str(csv_files[0])

def submit_only_mode(competition_name, message="", logger=None):
    """Modo apenas para submeter arquivo existente"""
    latest_file = get_latest_submission()
    
    if not latest_file:
        print("❌ Nenhum arquivo de submissão encontrado!")
        print(f"Procure em: submissions/")
        return False
    
    # Usar logger se disponível, senão print
    log_func = logger.info if logger else print
    
    if not message:
        file_name = os.path.basename(latest_file)
        # Extrair nome do modelo do arquivo
        if '_' in file_name:
            model_name = file_name.split('_')[1] if len(file_name.split('_')) > 1 else "Desconhecido"
            message = f"Submissao automatica - Modelo: {model_name}"
        else:
            message = "Submissao automatica"
    
    log_func(f"📤 Enviando arquivo mais recente: {os.path.basename(latest_file)}")
    log_func(f"📝 Mensagem: {message}")
    
    return submit_to_kaggle_automatically(latest_file, "Último modelo", competition_name, 
                                         logger or logging.getLogger(__name__))

def main():
    """Função principal"""
    
    # Configurar argumentos de linha de comando
    parser = argparse.ArgumentParser(
        description='Pipeline de predição de trajetórias - TE Aprendizado de Máquina',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  %(prog)s                    # Executa pipeline sem enviar
  %(prog)s --submit           # Executa pipeline e envia para Kaggle
  %(prog)s --submit-only      # Apenas envia o último arquivo
  %(prog)s --submit-only -m "Minha mensagem"  # Envia com mensagem customizada
        """
    )
    parser.add_argument('--submit', action='store_true', 
                       help='Executa pipeline completo e envia submissão para Kaggle')
    parser.add_argument('--submit-only', action='store_true',
                       help='Apenas envia o último arquivo de submissão (não executa pipeline)')
    parser.add_argument('-m', '--message', type=str, default='',
                       help='Mensagem customizada para submissão Kaggle')
    parser.add_argument('--model', type=str, default='',
                       help='Modelo específico para usar (opcional)')
    
    args = parser.parse_args()
    
    # Inicializar
    config = initialize_project()
    logger = setup_logging()
    
    print("=" * 60)
    print(f"TRAJECTORY PREDICTION - {config.KAGGLE_COMPETITION}")
    print("=" * 60)
    print(f"Dados: {config.DATA_DIR}")
    print(f"Logs: {config.LOGS_DIR}")
    print(f"Submissoes: {config.SUBMISSIONS_DIR}")
    
    if args.submit_only:
        print(f"Modo: APENAS SUBMISSÃO")
        print()
        
        # Verificar Kaggle CLI
        if not check_kaggle_cli():
            return 1
        
        # Executar apenas submissão
        success = submit_only_mode(
            config.KAGGLE_COMPETITION, 
            args.message if args.message else "",
            logger
        )
        
        if success:
            print("\n✅ Submissão concluída!")
            return 0
        else:
            print("\n❌ Falha na submissão!")
            return 1
    
    else:
        print(f"Submissao automatica: {'SIM' if args.submit else 'NAO'}")
        print()
        
        # Executar pipeline completo
        result = run_data_pipeline(logger, config, auto_submit=args.submit)
        
        if result['success']:
            print("\n" + "=" * 60)
            print("✅ PIPELINE CONCLUIDO COM SUCESSO!")
            print("=" * 60)
            print(f"📄 Submissao: {result['submission_file']}")
            print(f"🤖 Modelo: {result['model_name']}")
            
            if result.get('best_model_error'):
                print(f"📊 Erro medio: {result['best_model_error']:.4f} km")
            
            if result.get('submitted'):
                print("🚀 Submissão enviada automaticamente ao Kaggle!")
            else:
                # Instruções para envio manual
                if args.submit:
                    print("⚠️  Submissão NÃO foi enviada (verifique logs acima)")
                
                print("\nPARA ENVIAR AO KAGGLE:")
                print("Opção 1 - Execute novamente com --submit:")
                print(f"   python main.py --submit")
                print("\nOpção 2 - Apenas envie o último arquivo:")
                print(f"   python main.py --submit-only")
                print("\nOpção 3 - Comando manual:")
                print(f"   kaggle competitions submit -c {config.KAGGLE_COMPETITION} \\")
                print(f"     -f {result['submission_file']} \\")
                print(f"     -m \"Submissao automatica - {result['model_name']}\"")
            
            print("\n" + "=" * 60)
            return 0
        else:
            print("\n" + "=" * 60)
            print("❌ PIPELINE FALHOU")
            print("=" * 60)
            print(f"Erro: {result.get('error', 'Desconhecido')}")
            
            # Mostrar traceback se disponível
            if 'traceback' in result:
                print("\nTraceback:")
                print(result['traceback'])
            
            return 1

if __name__ == "__main__":
    sys.exit(main())