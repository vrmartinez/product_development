#!/usr/bin/env python
"""
Script de entrenamiento para el pipeline DVC.

Este script entrena y evalúa modelos, selecciona el mejor modelo,
y guarda el pipeline completo junto con las métricas.

Uso:
    python scripts/dvc_train.py --data-path data/processed/prepared_data.csv \
        --feature-pipeline-path models/feature_engineering_pipeline.pkl \
        --output-path models/sales_pipeline.pkl \
        --metrics-path reports/metrics.json
"""
import json
import sys
from pathlib import Path

# Agregar el root del proyecto al path para importar product_development
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
from loguru import logger
import typer
import yaml

from product_development.config import (
    FEATURES,
    TARGET,
    TRAIN_TEST_SPLIT_RATIO,
    CATEGORICAL_VARS,
)
from product_development.modeling.train import (
    create_full_pipeline,
    get_model_configurations,
    save_pipeline,
    train_and_evaluate_models,
    train_final_model,
)

app = typer.Typer()


def load_params() -> dict:
    """
    Carga parámetros desde params.yaml si existe.
    
    Retorna
    -------
    dict
        Diccionario con los parámetros o vacío si no existe el archivo.
    """
    # Buscar params.yaml en el root del proyecto (un nivel arriba de scripts/)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    params_file = project_root / "params.yaml"
    
    if params_file.exists():
        with open(params_file, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


# Rutas por defecto basadas en el root del proyecto
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "prepared_data.csv"
DEFAULT_FEATURE_PIPELINE_PATH = PROJECT_ROOT / "models" / "feature_engineering_pipeline.pkl"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "models" / "sales_pipeline.pkl"
DEFAULT_METRICS_PATH = PROJECT_ROOT / "reports" / "metrics.json"


@app.command()
def main(
    data_path: Path = typer.Option(DEFAULT_DATA_PATH, help="Ruta a los datos preparados"),
    feature_pipeline_path: Path = typer.Option(DEFAULT_FEATURE_PIPELINE_PATH, help="Ruta al pipeline de características"),
    output_path: Path = typer.Option(DEFAULT_OUTPUT_PATH, help="Ruta para guardar el pipeline entrenado"),
    metrics_path: Path = typer.Option(DEFAULT_METRICS_PATH, help="Ruta para guardar las métricas"),
):
    """
    Entrena el modelo y guarda el pipeline junto con las métricas.
    
    Parámetros
    ----------
    data_path : Path
        Ruta al archivo de datos preparados.
    feature_pipeline_path : Path
        Ruta al pipeline de ingeniería de características.
    output_path : Path
        Ruta para guardar el pipeline completo.
    metrics_path : Path
        Ruta para guardar las métricas en formato JSON.
    """
    logger.info("=" * 60)
    logger.info("DVC PIPELINE: ENTRENAMIENTO DE MODELO")
    logger.info("=" * 60)
    
    # Cargar parámetros
    params = load_params()
    train_ratio = params.get("data", {}).get("train_test_split_ratio", TRAIN_TEST_SPLIT_RATIO)
    training_mode = params.get("training", {}).get("mode", "fast")
    use_mlflow = params.get("training", {}).get("use_mlflow", True)
    
    logger.info(f"Modo de entrenamiento: {training_mode}")
    logger.info(f"Ratio entrenamiento/prueba: {train_ratio}")
    logger.info(f"Usar MLflow: {use_mlflow}")
    
    # Cargar datos preparados
    logger.info(f"Cargando datos desde {data_path}")
    data = pd.read_csv(data_path, parse_dates=["date"])
    
    # Convertir tipos categóricos
    for col in CATEGORICAL_VARS:
        if col in data.columns:
            data[col] = data[col].astype("O")
    
    # Cargar pipeline de características
    logger.info(f"Cargando pipeline de características desde {feature_pipeline_path}")
    feature_pipeline = joblib.load(feature_pipeline_path)
    
    # Dividir datos temporalmente
    logger.info("Dividiendo datos para entrenamiento y validación")
    df = data.sort_values("date").reset_index(drop=True)
    n_samples = len(df)
    n_train = int(n_samples * train_ratio)
    
    X_train = df.iloc[:n_train][FEATURES].copy()
    y_train = df.iloc[:n_train][TARGET].copy()
    X_val = df.iloc[n_train:][FEATURES].copy()
    y_val = df.iloc[n_train:][TARGET].copy()
    
    logger.info(f"Datos de entrenamiento: {len(X_train)} muestras")
    logger.info(f"Datos de validación: {len(X_val)} muestras")
    
    # Transformar características
    logger.info("Transformando características")
    X_train_transformed = feature_pipeline.transform(X_train)
    X_val_transformed = feature_pipeline.transform(X_val)
    
    # Obtener configuraciones de modelos
    models = get_model_configurations(mode=training_mode)
    
    # Entrenar y evaluar modelos
    logger.info("Iniciando entrenamiento y evaluación de modelos")
    results, best_model_name, best_model = train_and_evaluate_models(
        X_train_transformed, y_train,
        X_val_transformed, y_val,
        models,
        use_mlflow=use_mlflow
    )
    
    # Mostrar resultados
    logger.info("Resultados de evaluación:")
    for model_name, metrics in sorted(results.items(), key=lambda x: x[1]["rmse"]):
        logger.info(f"  {model_name}: RMSE={metrics['rmse']}, R2={metrics['r2']}")
    
    # Crear pipeline completo con el mejor modelo
    logger.info(f"Creando pipeline completo con el mejor modelo: {best_model_name}")
    full_pipeline = create_full_pipeline(feature_pipeline, best_model)
    
    # Obtener dataset completo para entrenamiento final
    X_full = data[FEATURES].copy()
    y_full = data[TARGET].copy()
    
    # Entrenar modelo final con dataset completo
    logger.info("Entrenando modelo final con dataset completo")
    full_pipeline = train_final_model(full_pipeline, X_full, y_full)
    
    # Guardar pipeline
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_pipeline(full_pipeline, output_path)
    logger.info(f"Pipeline guardado en {output_path}")
    
    # Preparar y guardar métricas
    best_metrics = results[best_model_name]
    metrics_output = {
        "best_model": best_model_name,
        "rmse": best_metrics["rmse"],
        "mse": best_metrics["mse"],
        "mae": best_metrics["mae"],
        "r2": best_metrics["r2"],
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "total_samples": len(data),
        "all_models": {
            name: {
                "rmse": m["rmse"],
                "r2": m["r2"]
            } for name, m in results.items()
        }
    }
    
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_output, f, indent=2)
    
    logger.info(f"Métricas guardadas en {metrics_path}")
    
    logger.info("=" * 60)
    logger.info("ENTRENAMIENTO COMPLETADO")
    logger.info("=" * 60)
    logger.info(f"Mejor modelo: {best_model_name}")
    logger.info(f"RMSE: {best_metrics['rmse']}")
    logger.info(f"R2: {best_metrics['r2']}")
    logger.success("¡Pipeline de entrenamiento ejecutado exitosamente!")


if __name__ == "__main__":
    app()
