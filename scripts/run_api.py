"""
Script para ejecutar la API de predicción de ventas.

Uso:
    python run_api.py [--host HOST] [--port PORT] [--debug]
"""
import argparse
import sys
from pathlib import Path

# Agregar la carpeta padre al path para poder importar el paquete
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from product_development.api import run_api, set_model_metrics
from product_development.config import PIPELINE_FILE
from product_development.modeling.train import load_pipeline


def load_model_metrics_from_mlflow():
    """
    Intenta cargar las métricas del modelo desde MLflow.
    
    Retorna
    -------
    Dict[str, float]
        Métricas del modelo o valores por defecto.
    """
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        from product_development.config import (
            MLFLOW_MODEL_NAME,
            MLFLOW_CHAMPION_ALIAS,
            MLFLOW_TRACKING_URI,
        )
        
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()
        
        # Obtener versión del champion
        champion_version = client.get_model_version_by_alias(
            MLFLOW_MODEL_NAME, MLFLOW_CHAMPION_ALIAS
        )
        
        # Obtener métricas del run
        run = client.get_run(champion_version.run_id)
        metrics = run.data.metrics
        
        logger.info(f"Métricas cargadas desde MLflow: {metrics}")
        return metrics
        
    except Exception as e:
        logger.warning(f"No se pudieron cargar métricas desde MLflow: {e}")
        return None


def main(host: str = "0.0.0.0", port: int = 5000, debug: bool = False):
    """
    Ejecuta la API de predicción de ventas.
    
    Args:
        host: Host donde ejecutar la API
        port: Puerto donde ejecutar la API
        debug: Ejecutar en modo debug
    """
    logger.info("Iniciando API de predicción de ventas...")
    
    # Verificar que el pipeline existe
    if not PIPELINE_FILE.exists():
        logger.error(f"Pipeline no encontrado en {PIPELINE_FILE}")
        logger.error("Por favor, entrene el modelo primero ejecutando el notebook 04_model_tuning_training.ipynb")
        sys.exit(1)
    
    # Intentar cargar métricas desde MLflow
    metrics = load_model_metrics_from_mlflow()
    
    if metrics is None:
        # Usar métricas por defecto si no se pueden cargar
        logger.warning("Usando métricas por defecto. Las métricas reales se mostrarán si están disponibles en MLflow.")
        metrics = {
            "rmse": 0.0,
            "mae": 0.0,
            "r2": 0.0,
            "mse": 0.0
        }
    
    # Ejecutar API
    run_api(host=host, port=port, debug=debug, metrics=metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ejecuta la API de predicción de ventas")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host donde ejecutar la API")
    parser.add_argument("--port", type=int, default=5000, help="Puerto donde ejecutar la API")
    parser.add_argument("--debug", action="store_true", help="Ejecutar en modo debug")
    
    args = parser.parse_args()
    main(host=args.host, port=args.port, debug=args.debug)
