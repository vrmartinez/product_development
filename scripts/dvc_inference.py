#!/usr/bin/env python
"""
Script de inferencia para el pipeline DVC.

Este script carga el pipeline entrenado y genera predicciones
sobre los datos de prueba.

Uso:
    python scripts/dvc_inference.py --data-path data/processed/prepared_data.csv \
        --pipeline-path models/sales_pipeline.pkl \
        --output-path data/processed/test_predictions.csv
"""
import sys
from pathlib import Path

# Agregar el root del proyecto al path para importar product_development
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from loguru import logger
import typer
import yaml

from product_development.config import (
    CATEGORICAL_VARS,
    TRAIN_TEST_SPLIT_RATIO,
)
from product_development.modeling.predict import (
    prepare_inference_data,
    predict,
    save_predictions,
)
from product_development.modeling.train import load_pipeline

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
DEFAULT_PIPELINE_PATH = PROJECT_ROOT / "models" / "sales_pipeline.pkl"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "test_predictions.csv"


@app.command()
def main(
    data_path: Path = typer.Option(DEFAULT_DATA_PATH, help="Ruta a los datos preparados"),
    pipeline_path: Path = typer.Option(DEFAULT_PIPELINE_PATH, help="Ruta al pipeline entrenado"),
    output_path: Path = typer.Option(DEFAULT_OUTPUT_PATH, help="Ruta para guardar las predicciones"),
):
    """
    Genera predicciones usando el pipeline entrenado.
    
    Parámetros
    ----------
    data_path : Path
        Ruta al archivo de datos preparados.
    pipeline_path : Path
        Ruta al pipeline completo entrenado.
    output_path : Path
        Ruta para guardar las predicciones en formato CSV.
    """
    logger.info("=" * 60)
    logger.info("DVC PIPELINE: INFERENCIA")
    logger.info("=" * 60)
    
    # Cargar parámetros
    params = load_params()
    train_ratio = params.get("data", {}).get("train_test_split_ratio", TRAIN_TEST_SPLIT_RATIO)
    
    # Cargar pipeline
    logger.info(f"Cargando pipeline desde {pipeline_path}")
    pipeline = load_pipeline(pipeline_path)
    
    # Cargar datos
    logger.info(f"Cargando datos desde {data_path}")
    data = pd.read_csv(data_path, parse_dates=["date"])
    
    # Convertir tipos categóricos
    for col in CATEGORICAL_VARS:
        if col in data.columns:
            data[col] = data[col].astype("O")
    
    # Obtener datos de prueba (después del split temporal)
    df = data.sort_values("date").reset_index(drop=True)
    n_samples = len(df)
    n_train = int(n_samples * train_ratio)
    test_data = df.iloc[n_train:].copy()
    
    logger.info(f"Datos de prueba: {len(test_data)} muestras")
    
    # Preparar datos para inferencia
    X_test = prepare_inference_data(test_data)
    
    # Generar predicciones
    logger.info("Generando predicciones...")
    predictions = predict(pipeline, X_test)
    
    # Guardar predicciones
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = save_predictions(test_data, predictions, output_path)
    
    logger.info("=" * 60)
    logger.info("INFERENCIA COMPLETADA")
    logger.info("=" * 60)
    logger.info(f"Predicciones generadas: {len(predictions)}")
    logger.info(f"Archivo guardado en: {output_path}")
    logger.success("¡Pipeline de inferencia ejecutado exitosamente!")


if __name__ == "__main__":
    app()
