"""
Módulo de entrenamiento de modelos para el pipeline de predicción de ventas.

Este módulo proporciona funciones para entrenar, evaluar y seleccionar
el mejor modelo para la predicción de ventas, con tracking completo en MLflow.
"""
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from loguru import logger
from mlflow.tracking import MlflowClient
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from tqdm import tqdm
import typer
from xgboost import XGBRegressor

from product_development.config import (
    FEATURE_PIPELINE_FILE,
    MLFLOW_CHAMPION_ALIAS,
    MLFLOW_CHALLENGER_ALIAS,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_MODEL_NAME,
    MLFLOW_TRACKING_URI,
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


def setup_mlflow(
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
    tracking_uri: str = MLFLOW_TRACKING_URI
) -> str:
    """
    Configura MLflow para el tracking de experimentos.

    Parámetros
    ----------
    experiment_name : str
        Nombre del experimento MLflow.
    tracking_uri : str
        URI del servidor de tracking MLflow.

    Retorna
    -------
    str
        ID del experimento configurado.
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    logger.info(f"MLflow configurado - Experimento: {experiment_name}")
    return experiment.experiment_id


def get_model_hyperparameters(model: object) -> Dict:
    """
    Extrae los hiperparámetros de un modelo.

    Parámetros
    ----------
    model : object
        Instancia del modelo.

    Retorna
    -------
    Dict
        Diccionario con los hiperparámetros del modelo.
    """
    try:
        return model.get_params()
    except AttributeError:
        return {}


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calcula múltiples métricas de evaluación.

    Parámetros
    ----------
    y_true : np.ndarray
        Valores reales.
    y_pred : np.ndarray
        Valores predichos.

    Retorna
    -------
    Dict[str, float]
        Diccionario con las métricas calculadas.
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        "mse": round(mse, 4),
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2": round(r2, 4)
    }


def train_and_evaluate_models(
    X_train: np.ndarray,
    y_train: pd.Series,
    X_val: np.ndarray,
    y_val: pd.Series,
    models: Dict[str, object] = None,
    use_mlflow: bool = True
) -> Tuple[Dict[str, Dict], str, object]:
    """
    Entrena y evalúa múltiples configuraciones de modelos con tracking en MLflow.

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
    use_mlflow : bool, opcional
        Si True, registra los experimentos en MLflow. Por defecto True.

    Retorna
    -------
    Tuple[Dict[str, Dict], str, object]
        Diccionario de resultados (con todas las métricas), nombre del mejor modelo,
        instancia del mejor modelo.
    """
    if models is None:
        models = get_model_configurations()

    results = {}
    run_ids = {}
    logger.info("Iniciando entrenamiento y evaluación de modelos")

    # Configurar MLflow si está habilitado
    if use_mlflow:
        setup_mlflow()

    for model_name, model in tqdm(models.items(), desc="Entrenando modelos"):
        logger.info(f"Entrenando modelo: {model_name}")

        try:
            if use_mlflow:
                with mlflow.start_run(run_name=model_name) as run:
                    # Registrar hiperparámetros
                    hyperparams = get_model_hyperparameters(model)
                    mlflow.log_params(hyperparams)
                    mlflow.log_param("model_type", type(model).__name__)

                    # Entrenar modelo
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_val)

                    # Calcular y registrar métricas
                    metrics = calculate_metrics(y_val, y_pred)
                    mlflow.log_metrics(metrics)

                    # Registrar información adicional
                    mlflow.log_param("train_samples", len(X_train))
                    mlflow.log_param("val_samples", len(X_val))
                    mlflow.log_param("n_features", X_train.shape[1])

                    # Registrar modelo en MLflow con input_example para signature
                    # Usar una muestra pequeña para el input_example
                    input_example = X_train[:5] if len(X_train) >= 5 else X_train
                    mlflow.sklearn.log_model(
                        model,
                        name="model",
                        input_example=input_example
                    )

                    results[model_name] = metrics
                    run_ids[model_name] = run.info.run_id

                    logger.info(f"Modelo {model_name} - RMSE: {metrics['rmse']}, "
                                f"R2: {metrics['r2']}")
            else:
                # Entrenamiento sin MLflow
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                metrics = calculate_metrics(y_val, y_pred)
                results[model_name] = metrics
                logger.info(f"Modelo {model_name} - RMSE: {metrics['rmse']}")

        except Exception as ex:  # pylint: disable=broad-except
            logger.error(f"Error entrenando modelo {model_name}: {ex}")
            results[model_name] = {"rmse": float("inf"), "mse": float("inf"),
                                   "mae": float("inf"), "r2": float("-inf")}

    # Encontrar el mejor modelo basado en RMSE
    best_model_name = min(results, key=lambda x: results[x]["rmse"])
    best_metrics = results[best_model_name]
    best_model = models[best_model_name]

    logger.info(f"Mejor modelo: {best_model_name} con RMSE = {best_metrics['rmse']}")

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


def register_model_to_mlflow(
    pipeline: Pipeline,
    model_name: str,
    best_model_name: str,
    metrics: Dict[str, float],
    register_as_champion: bool = False,
    input_example: pd.DataFrame = None
) -> Optional[str]:
    """
    Registra un pipeline completo en MLflow Model Registry.

    Parámetros
    ----------
    pipeline : Pipeline
        Pipeline completo a registrar.
    model_name : str
        Nombre para el modelo en el registry.
    best_model_name : str
        Nombre del mejor modelo/algoritmo usado.
    metrics : Dict[str, float]
        Métricas del modelo.
    register_as_champion : bool
        Si True, registra el modelo como champion.
    input_example : pd.DataFrame, opcional
        Ejemplo de entrada para inferir la signature del modelo.

    Retorna
    -------
    Optional[str]
        Versión del modelo registrado o None si falla.
    """
    try:
        setup_mlflow()
        with mlflow.start_run(run_name=f"final_pipeline_{best_model_name}") as run:
            # Registrar métricas finales
            mlflow.log_metrics(metrics)
            mlflow.log_param("best_model_type", best_model_name)
            mlflow.log_param("pipeline_steps", [step[0] for step in pipeline.steps])

            # Registrar el pipeline completo con input_example para signature
            model_info = mlflow.sklearn.log_model(
                pipeline,
                name="pipeline",
                registered_model_name=model_name,
                input_example=input_example
            )

            logger.info(f"Pipeline registrado en MLflow: {model_info.model_uri}")

            # Obtener la versión del modelo registrado
            client = MlflowClient()
            latest_versions = client.get_latest_versions(model_name)

            if latest_versions:
                version = latest_versions[0].version
                logger.info(f"Modelo registrado como versión: {version}")

                if register_as_champion:
                    # Asignar alias de champion
                    client.set_registered_model_alias(
                        model_name, MLFLOW_CHAMPION_ALIAS, version
                    )
                    logger.info(f"Modelo versión {version} marcado como '{MLFLOW_CHAMPION_ALIAS}'")
                else:
                    # Asignar alias de challenger
                    client.set_registered_model_alias(
                        model_name, MLFLOW_CHALLENGER_ALIAS, version
                    )
                    logger.info(f"Modelo versión {version} marcado como '{MLFLOW_CHALLENGER_ALIAS}'")

                return version

    except Exception as ex:  # pylint: disable=broad-except
        logger.error(f"Error registrando modelo en MLflow: {ex}")

    return None


def get_champion_model(model_name: str = MLFLOW_MODEL_NAME) -> Optional[Pipeline]:
    """
    Carga el modelo champion desde MLflow Model Registry.

    Parámetros
    ----------
    model_name : str
        Nombre del modelo en el registry.

    Retorna
    -------
    Optional[Pipeline]
        Pipeline del modelo champion o None si no existe.
    """
    try:
        setup_mlflow()
        model_uri = f"models:/{model_name}@{MLFLOW_CHAMPION_ALIAS}"
        champion = mlflow.sklearn.load_model(model_uri)
        logger.info(f"Modelo champion cargado desde: {model_uri}")
        return champion
    except Exception as ex:  # pylint: disable=broad-except
        logger.warning(f"No se encontró modelo champion: {ex}")
        return None


def compare_with_champion(
    challenger_metrics: Dict[str, float],
    model_name: str = MLFLOW_MODEL_NAME,
    metric_key: str = "rmse"
) -> Tuple[bool, Optional[Dict[str, float]]]:
    """
    Compara las métricas de un modelo challenger con el champion actual.

    Parámetros
    ----------
    challenger_metrics : Dict[str, float]
        Métricas del modelo challenger.
    model_name : str
        Nombre del modelo en el registry.
    metric_key : str
        Métrica a usar para comparación (por defecto 'rmse').

    Retorna
    -------
    Tuple[bool, Optional[Dict[str, float]]]
        (True si challenger es mejor, métricas del champion o None).
    """
    try:
        setup_mlflow()
        client = MlflowClient()

        # Obtener información del modelo champion
        champion_version = client.get_model_version_by_alias(
            model_name, MLFLOW_CHAMPION_ALIAS
        )

        # Obtener el run asociado al champion
        run = client.get_run(champion_version.run_id)
        champion_metrics = run.data.metrics

        logger.info(f"Champion {metric_key}: {champion_metrics.get(metric_key, 'N/A')}")
        logger.info(f"Challenger {metric_key}: {challenger_metrics.get(metric_key, 'N/A')}")

        # Para RMSE/MSE/MAE menor es mejor, para R2 mayor es mejor
        if metric_key in ("rmse", "mse", "mae"):
            is_better = challenger_metrics[metric_key] < champion_metrics.get(metric_key, float("inf"))
        else:
            is_better = challenger_metrics[metric_key] > champion_metrics.get(metric_key, float("-inf"))

        return is_better, champion_metrics

    except Exception as ex:  # pylint: disable=broad-except
        logger.warning(f"No se pudo comparar con champion: {ex}")
        return True, None  # Si no hay champion, el challenger gana por defecto


def promote_challenger_to_champion(
    model_name: str = MLFLOW_MODEL_NAME
) -> bool:
    """
    Promueve el modelo challenger actual a champion.

    Parámetros
    ----------
    model_name : str
        Nombre del modelo en el registry.

    Retorna
    -------
    bool
        True si la promoción fue exitosa.
    """
    try:
        setup_mlflow()
        client = MlflowClient()

        # Obtener versión del challenger
        challenger_version = client.get_model_version_by_alias(
            model_name, MLFLOW_CHALLENGER_ALIAS
        )

        # Promover a champion
        client.set_registered_model_alias(
            model_name, MLFLOW_CHAMPION_ALIAS, challenger_version.version
        )

        logger.info(f"Modelo versión {challenger_version.version} promovido a champion")
        return True

    except Exception as ex:  # pylint: disable=broad-except
        logger.error(f"Error promoviendo challenger a champion: {ex}")
        return False


def get_experiment_summary(
    experiment_name: str = MLFLOW_EXPERIMENT_NAME
) -> pd.DataFrame:
    """
    Obtiene un resumen de todos los runs de un experimento.

    Parámetros
    ----------
    experiment_name : str
        Nombre del experimento.

    Retorna
    -------
    pd.DataFrame
        DataFrame con el resumen de todos los runs.
    """
    setup_mlflow()
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        logger.warning(f"Experimento '{experiment_name}' no encontrado")
        return pd.DataFrame()

    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

    if runs.empty:
        logger.info("No hay runs registrados en el experimento")
        return pd.DataFrame()

    # Seleccionar columnas relevantes
    metric_cols = [col for col in runs.columns if col.startswith("metrics.")]
    param_cols = [col for col in runs.columns if col.startswith("params.")]

    summary_cols = ["run_id", "run_name", "status", "start_time"] + metric_cols + param_cols
    available_cols = [col for col in summary_cols if col in runs.columns]

    return runs[available_cols].sort_values("start_time", ascending=False)


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
