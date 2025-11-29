"""
Módulo de inferencia para el pipeline de predicción de ventas.

Este módulo proporciona funciones para generar predicciones usando
el pipeline de ventas entrenado.
"""
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.pipeline import Pipeline
import typer

from product_development.config import (
    FEATURES,
    MODELS_DIR,
    PIPELINE_FILE,
    PREDICTIONS_FILE,
    PROCESSED_DATA_DIR,
)
from product_development.modeling.train import load_pipeline

app = typer.Typer()


def predict(
    pipeline: Pipeline,
    X: pd.DataFrame
) -> np.ndarray:
    """
    Genera predicciones usando el pipeline entrenado.

    Parámetros
    ----------
    pipeline : Pipeline
        Pipeline entrenado.
    X : pd.DataFrame
        Características para predicción.

    Retorna
    -------
    np.ndarray
        Valores predichos.
    """
    logger.info(f"Generando predicciones para {len(X)} muestras")
    predictions = pipeline.predict(X)
    logger.info("Predicciones generadas exitosamente")
    return predictions


def prepare_inference_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara datos para inferencia extrayendo las características requeridas.

    Parámetros
    ----------
    data : pd.DataFrame
        DataFrame de entrada con datos crudos.

    Retorna
    -------
    pd.DataFrame
        DataFrame con solo las características requeridas.
    """
    logger.info("Preparando datos para inferencia")

    df = data.copy()

    # Asegurar que date sea datetime
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

        # Agregar características temporales si no están presentes
        if "year" not in df.columns:
            df["year"] = df["date"].dt.year
        if "month" not in df.columns:
            df["month"] = df["date"].dt.month
        if "day_of_week_name" not in df.columns:
            df["day_of_week_name"] = df["date"].dt.day_name()

    # Convertir tipos categóricos
    for col in ["store", "item", "day_of_week_name"]:
        if col in df.columns:
            df[col] = df[col].astype("O")

    # Extraer características
    X = df[FEATURES].copy()
    return X


def save_predictions(
    data: pd.DataFrame,
    predictions: np.ndarray,
    output_path: Path = PREDICTIONS_FILE
) -> pd.DataFrame:
    """
    Guarda predicciones en archivo CSV.

    Parámetros
    ----------
    data : pd.DataFrame
        Datos originales.
    predictions : np.ndarray
        Valores predichos.
    output_path : Path
        Ruta para guardar predicciones.

    Retorna
    -------
    pd.DataFrame
        DataFrame con predicciones.
    """
    logger.info(f"Guardando predicciones en {output_path}")

    output_df = data.copy()
    output_df["sales_pred"] = predictions

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    logger.info(f"Predicciones guardadas: {len(predictions)} registros")
    return output_df


def run_inference(
    pipeline: Pipeline,
    data: pd.DataFrame,
    output_path: Path = PREDICTIONS_FILE
) -> pd.DataFrame:
    """
    Ejecuta el pipeline completo de inferencia.

    Parámetros
    ----------
    pipeline : Pipeline
        Pipeline entrenado.
    data : pd.DataFrame
        Datos de entrada para inferencia.
    output_path : Path
        Ruta para guardar predicciones.

    Retorna
    -------
    pd.DataFrame
        DataFrame con predicciones.
    """
    X = prepare_inference_data(data)
    predictions = predict(pipeline, X)
    result = save_predictions(data, predictions, output_path)
    return result


def load_and_predict(
    data: Union[pd.DataFrame, Path],
    pipeline_path: Path = PIPELINE_FILE,
    output_path: Path = PREDICTIONS_FILE
) -> pd.DataFrame:
    """
    Carga pipeline y genera predicciones.

    Parámetros
    ----------
    data : Union[pd.DataFrame, Path]
        Datos de entrada o ruta al archivo de datos.
    pipeline_path : Path
        Ruta al pipeline guardado.
    output_path : Path
        Ruta para guardar predicciones.

    Retorna
    -------
    pd.DataFrame
        DataFrame con predicciones.
    """
    # Cargar pipeline
    pipeline = load_pipeline(pipeline_path)

    # Cargar datos si se proporciona ruta
    if isinstance(data, Path):
        logger.info(f"Cargando datos desde {data}")
        data = pd.read_csv(data, parse_dates=["date"])

    # Ejecutar inferencia
    result = run_inference(pipeline, data, output_path)
    return result


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    model_path: Path = MODELS_DIR / "sales_pipeline.pkl",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
):
    """
    Función principal para realizar inferencia.

    Parámetros
    ----------
    features_path : Path
        Ruta al archivo de características.
    model_path : Path
        Ruta al pipeline guardado.
    predictions_path : Path
        Ruta para guardar predicciones.
    """
    logger.info("Realizando inferencia del modelo...")

    # Cargar datos
    data = pd.read_csv(features_path, parse_dates=["date"])

    # Ejecutar inferencia
    load_and_predict(data, model_path, predictions_path)

    logger.success("Inferencia completada.")


if __name__ == "__main__":
    app()
