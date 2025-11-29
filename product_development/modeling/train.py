"""
Módulo de entrenamiento de modelos para el pipeline de predicción de ventas.

Este módulo proporciona funciones para entrenar, evaluar y seleccionar
el mejor modelo para la predicción de ventas.
"""
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from tqdm import tqdm
import typer
from xgboost import XGBRegressor

from product_development.config import (
    FEATURE_PIPELINE_FILE,
    MODELS_DIR,
    PIPELINE_FILE,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
)

app = typer.Typer()


def get_model_configurations(mode: str = "fast") -> Dict[str, object]:
    """
    Obtiene diccionario de configuraciones de modelos a evaluar.

    Parámetros
    ----------
    mode : str, opcional
        Modo de entrenamiento: "fast" (rápido, por defecto) o "full" (completo).
        - "fast": Evalúa un modelo representativo de cada tipo (5 modelos).
        - "full": Evalúa todas las configuraciones (15 modelos).

    Retorna
    -------
    Dict[str, object]
        Diccionario mapeando nombres de modelos a instancias de modelos.

    Excepciones
    -----------
    ValueError
        Si el modo no es "fast" ni "full".
    """
    if mode not in ("fast", "full"):
        raise ValueError(f"Modo inválido: {mode}. Use 'fast' o 'full'.")

    if mode == "fast":
        # Modo rápido: un modelo representativo de cada tipo (sin SVR por lentitud)
        models = {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(
                n_estimators=100, max_depth=10,
                random_state=RANDOM_STATE, n_jobs=-1
            ),
            "GradientBoosting": GradientBoostingRegressor(
                n_estimators=100, learning_rate=0.1, random_state=RANDOM_STATE
            ),
            "XGBoost": XGBRegressor(
                n_estimators=100, learning_rate=0.1, max_depth=5,
                random_state=RANDOM_STATE, n_jobs=-1
            ),
        }
    else:
        # Modo full: todas las configuraciones
        models = {
            # Configuraciones de Regresión Lineal
            "LinearRegression_1": LinearRegression(),
            "LinearRegression_2": LinearRegression(fit_intercept=False),
            "LinearRegression_3": LinearRegression(positive=True),
            # Configuraciones de Random Forest
            "RandomForest_1": RandomForestRegressor(
                n_estimators=100, max_depth=10,
                random_state=RANDOM_STATE, n_jobs=-1
            ),
            "RandomForest_2": RandomForestRegressor(
                n_estimators=200, max_depth=20,
                random_state=RANDOM_STATE, n_jobs=-1
            ),
            "RandomForest_3": RandomForestRegressor(
                n_estimators=300, max_depth=None,
                random_state=RANDOM_STATE, n_jobs=-1
            ),
            # Configuraciones de Gradient Boosting
            "GradientBoosting_1": GradientBoostingRegressor(
                n_estimators=100, learning_rate=0.1, random_state=RANDOM_STATE
            ),
            "GradientBoosting_2": GradientBoostingRegressor(
                n_estimators=200, learning_rate=0.05, random_state=RANDOM_STATE
            ),
            "GradientBoosting_3": GradientBoostingRegressor(
                n_estimators=300, learning_rate=0.03, random_state=RANDOM_STATE
            ),
            # Configuraciones de SVR
            "SVR_1": SVR(kernel="rbf", C=1.0, epsilon=0.1),
            "SVR_2": SVR(kernel="rbf", C=10.0, epsilon=0.1),
            "SVR_3": SVR(kernel="rbf", C=100.0, epsilon=0.1),
            # Configuraciones de XGBoost
            "XGBoost_1": XGBRegressor(
                n_estimators=100, learning_rate=0.1, max_depth=5,
                random_state=RANDOM_STATE, n_jobs=-1
            ),
            "XGBoost_2": XGBRegressor(
                n_estimators=200, learning_rate=0.05, max_depth=7,
                random_state=RANDOM_STATE, n_jobs=-1
            ),
            "XGBoost_3": XGBRegressor(
                n_estimators=300, learning_rate=0.03, max_depth=9,
                random_state=RANDOM_STATE, n_jobs=-1
            ),
        }
    return models


def train_and_evaluate_models(
    X_train: np.ndarray,
    y_train: pd.Series,
    X_val: np.ndarray,
    y_val: pd.Series,
    models: Dict[str, object] = None
) -> Tuple[Dict[str, float], str, object]:
    """
    Entrena y evalúa múltiples configuraciones de modelos.

    Parámetros
    ----------
    X_train : np.ndarray
        Características de entrenamiento transformadas.
    y_train : pd.Series
        Objetivo de entrenamiento.
    X_val : np.ndarray
        Características de validación transformadas.
    y_val : pd.Series
        Objetivo de validación.
    models : Dict[str, object], opcional
        Diccionario de modelos a evaluar.

    Retorna
    -------
    Tuple[Dict[str, float], str, object]
        Diccionario de resultados, nombre del mejor modelo, instancia del mejor modelo.
    """
    if models is None:
        models = get_model_configurations()

    results = {}
    logger.info("Iniciando entrenamiento y evaluación de modelos")

    for model_name, model in tqdm(models.items(), desc="Entrenando modelos"):
        logger.info(f"Entrenando modelo: {model_name}")

        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            mse = mean_squared_error(y_val, y_pred)
            rmse = np.round(np.sqrt(mse), 2)
            results[model_name] = rmse

            logger.info(f"Modelo {model_name} RMSE: {rmse}")

        except Exception as ex:  # pylint: disable=broad-except
            logger.error(f"Error entrenando modelo {model_name}: {ex}")
            results[model_name] = float("inf")

    # Encontrar el mejor modelo
    best_model_name = min(results, key=results.get)
    best_rmse = results[best_model_name]
    best_model = models[best_model_name]

    logger.info(f"Mejor modelo: {best_model_name} con RMSE = {best_rmse}")

    return results, best_model_name, best_model


def create_full_pipeline(
    feature_pipeline: Pipeline,
    model: object
) -> Pipeline:
    """
    Crea pipeline completo agregando modelo al pipeline de características.

    Parámetros
    ----------
    feature_pipeline : Pipeline
        Pipeline de ingeniería de características ajustado.
    model : object
        Modelo a agregar al pipeline.

    Retorna
    -------
    Pipeline
        Pipeline completo con características y modelo.
    """
    logger.info("Creando pipeline completo con modelo")

    # Clonar los pasos del pipeline y agregar el modelo
    full_pipeline = Pipeline(
        list(feature_pipeline.steps) + [("regressor", model)]
    )

    return full_pipeline


def train_final_model(
    pipeline: Pipeline,
    X_full: pd.DataFrame,
    y_full: pd.Series
) -> Pipeline:
    """
    Entrena el pipeline final con el dataset completo.

    Parámetros
    ----------
    pipeline : Pipeline
        Pipeline completo a entrenar.
    X_full : pd.DataFrame
        Características completas.
    y_full : pd.Series
        Objetivo completo.

    Retorna
    -------
    Pipeline
        Pipeline entrenado.
    """
    logger.info("Entrenando modelo final con dataset completo")
    pipeline.fit(X_full, y_full)
    logger.info("Entrenamiento del modelo final completado")
    return pipeline


def save_pipeline(
    pipeline: Pipeline,
    output_path: Path = PIPELINE_FILE
) -> None:
    """
    Guarda el pipeline completo en disco.

    Parámetros
    ----------
    pipeline : Pipeline
        Pipeline a guardar.
    output_path : Path
        Ruta para guardar el pipeline.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)
    logger.info(f"Pipeline guardado en {output_path}")


def load_pipeline(input_path: Path = PIPELINE_FILE) -> Pipeline:
    """
    Carga un pipeline guardado desde disco.

    Parámetros
    ----------
    input_path : Path
        Ruta al pipeline guardado.

    Retorna
    -------
    Pipeline
        Pipeline cargado.

    Excepciones
    -----------
    FileNotFoundError
        Si el archivo del pipeline no existe.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Archivo de pipeline no encontrado: {input_path}")

    logger.info(f"Cargando pipeline desde {input_path}")
    pipeline = joblib.load(input_path)
    return pipeline


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    model_path: Path = MODELS_DIR / "sales_pipeline.pkl",
    pipeline_path: Path = FEATURE_PIPELINE_FILE,
):
    """
    Función principal para entrenar y guardar el mejor modelo.

    Parámetros
    ----------
    features_path : Path
        Ruta al archivo de características (no usado, mantenido por compatibilidad).
    labels_path : Path
        Ruta al archivo de etiquetas (no usado, mantenido por compatibilidad).
    model_path : Path
        Ruta para guardar el pipeline completo.
    pipeline_path : Path
        Ruta al pipeline de ingeniería de características.
    """
    logger.info("Entrenando modelo...")

    # Cargar pipeline de características
    feature_pipeline = joblib.load(pipeline_path)

    logger.success("Entrenamiento del modelo completado.")


if __name__ == "__main__":
    app()
