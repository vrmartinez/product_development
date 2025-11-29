"""
Módulo de ingeniería de características para el pipeline de predicción de ventas.

Este módulo proporciona funciones para construir y aplicar el pipeline de
ingeniería de características para la predicción de ventas.
"""
from pathlib import Path

import joblib
import pandas as pd
from feature_engine.encoding import CountFrequencyEncoder
from feature_engine.imputation import MeanMedianImputer
from loguru import logger
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import typer

from product_development.config import (
    CATEGORICAL_VARS_FREQ,
    CATEGORICAL_VARS_IMPUTE,
    DAY_OF_WEEK_MAPPING,
    FEATURE_PIPELINE_FILE,
    MODELS_DIR,
    NUMERICAL_VARS,
    PROCESSED_DATA_DIR,
)
from product_development.transformers import Mapper, SimpleCategoricalImputer

app = typer.Typer()


def build_feature_pipeline() -> Pipeline:
    """
    Construye el pipeline de ingeniería de características.

    El pipeline incluye:
    1. Imputación categórica para valores faltantes
    2. Imputación numérica usando la mediana
    3. Codificación de frecuencia para variables categóricas
    4. Mapeo del día de la semana
    5. Escalado MinMax

    Retorna
    -------
    Pipeline
        Pipeline de sklearn para ingeniería de características.
    """
    logger.info("Construyendo pipeline de ingeniería de características")

    pipeline = Pipeline([
        # Paso 1: Imputar variables categóricas
        ("cat_missing_imputation", SimpleCategoricalImputer(
            variables=CATEGORICAL_VARS_IMPUTE,
            fill_value="Missing"
        )),
        # Paso 2: Imputar variables numéricas con mediana
        ("num_median_imputation", MeanMedianImputer(
            imputation_method="median",
            variables=NUMERICAL_VARS
        )),
        # Paso 3: Codificación de frecuencia para variables categóricas
        ("cat_freq_encoder", CountFrequencyEncoder(
            encoding_method="frequency",
            variables=CATEGORICAL_VARS_FREQ
        )),
        # Paso 4: Mapear nombres de días de la semana a enteros
        ("dayofweek_mapper", Mapper(
            mappings=DAY_OF_WEEK_MAPPING,
            variables=["day_of_week_name"]
        )),
        # Paso 5: Escalar todas las características
        ("feature_scaler", MinMaxScaler())
    ])

    logger.info("Pipeline de características construido exitosamente")
    return pipeline


def fit_feature_pipeline(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> Pipeline:
    """
    Ajusta el pipeline de ingeniería de características con datos de entrenamiento.

    Parámetros
    ----------
    pipeline : Pipeline
        Pipeline de sklearn a ajustar.
    X_train : pd.DataFrame
        Características de entrenamiento.
    y_train : pd.Series
        Variable objetivo de entrenamiento.

    Retorna
    -------
    Pipeline
        Pipeline ajustado.
    """
    logger.info("Ajustando pipeline de ingeniería de características")
    pipeline.fit(X_train, y_train)
    logger.info("Pipeline de características ajustado exitosamente")
    return pipeline


def transform_features(
    pipeline: Pipeline,
    X: pd.DataFrame
) -> pd.DataFrame:
    """
    Transforma características usando un pipeline ajustado.

    Parámetros
    ----------
    pipeline : Pipeline
        Pipeline de sklearn ajustado.
    X : pd.DataFrame
        Características a transformar.

    Retorna
    -------
    pd.DataFrame
        Características transformadas.
    """
    logger.info(f"Transformando {len(X)} registros")
    X_transformed = pipeline.transform(X)
    return X_transformed


def save_feature_pipeline(
    pipeline: Pipeline,
    output_path: Path = FEATURE_PIPELINE_FILE
) -> None:
    """
    Guarda el pipeline de características ajustado en disco.

    Parámetros
    ----------
    pipeline : Pipeline
        Pipeline ajustado a guardar.
    output_path : Path
        Ruta para guardar el pipeline.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)
    logger.info(f"Pipeline de características guardado en {output_path}")


def load_feature_pipeline(input_path: Path = FEATURE_PIPELINE_FILE) -> Pipeline:
    """
    Carga un pipeline de características ajustado desde disco.

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

    logger.info(f"Cargando pipeline de características desde {input_path}")
    pipeline = joblib.load(input_path)
    return pipeline


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "prepared_data.csv",
    output_path: Path = MODELS_DIR / "feature_engineering_pipeline.pkl",
):
    """
    Función principal para construir y guardar el pipeline de ingeniería de características.

    Parámetros
    ----------
    input_path : Path
        Ruta al archivo de datos preparados.
    output_path : Path
        Ruta para guardar el pipeline ajustado.
    """
    logger.info("Generando características del dataset...")

    # Cargar datos preparados
    data = pd.read_csv(input_path, parse_dates=["date"])

    # Construir y ajustar pipeline
    pipeline = build_feature_pipeline()
    X = data[["store", "item", "year", "month", "day_of_week_name"]].copy()
    y = data["sales"].copy()

    # Convertir tipos
    for col in ["store", "item", "day_of_week_name"]:
        X[col] = X[col].astype("O")

    pipeline.fit(X, y)

    # Guardar pipeline
    save_feature_pipeline(pipeline, output_path)

    logger.success("Generación de características completada.")


if __name__ == "__main__":
    app()
