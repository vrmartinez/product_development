"""
Módulo de dataset para carga y preprocesamiento de datos de ventas.

Este módulo proporciona funciones para cargar datos crudos y prepararlos
para el pipeline de ingeniería de características.
"""
from pathlib import Path
from typing import Tuple

import pandas as pd
from loguru import logger
import typer

from product_development.config import (
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    TRAIN_DATA_FILE,
    FEATURES,
    TARGET,
    CATEGORICAL_VARS,
    TRAIN_TEST_SPLIT_RATIO,
)

app = typer.Typer()


def load_raw_data(input_path: Path = TRAIN_DATA_FILE) -> pd.DataFrame:
    """
    Carga datos crudos de ventas desde un archivo CSV.

    Parámetros
    ----------
    input_path : Path
        Ruta al archivo CSV de datos crudos.

    Retorna
    -------
    pd.DataFrame
        DataFrame cargado con fechas parseadas.

    Excepciones
    -----------
    FileNotFoundError
        Si el archivo de entrada no existe.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Archivo de datos no encontrado: {input_path}")

    logger.info(f"Cargando datos crudos desde {input_path}")
    data = pd.read_csv(input_path, parse_dates=["date"])
    logger.info(f"Cargados {len(data)} registros")
    return data


def add_temporal_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega características temporales extraídas de la columna de fecha.

    Parámetros
    ----------
    data : pd.DataFrame
        DataFrame con una columna 'date'.

    Retorna
    -------
    pd.DataFrame
        DataFrame con características temporales agregadas.
    """
    logger.info("Agregando características temporales")
    df = data.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day_of_week_name"] = df["date"].dt.day_name()
    return df


def convert_categorical_types(data: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte columnas categóricas a tipo object.

    Parámetros
    ----------
    data : pd.DataFrame
        DataFrame de entrada.

    Retorna
    -------
    pd.DataFrame
        DataFrame con columnas categóricas como tipo object.
    """
    logger.info("Convirtiendo columnas categóricas a tipo object")
    df = data.copy()
    for col in CATEGORICAL_VARS:
        if col in df.columns:
            df[col] = df[col].astype("O")
    return df


def prepare_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preparación completa del dataset incluyendo características temporales
    y conversión de tipos.

    Parámetros
    ----------
    data : pd.DataFrame
        DataFrame crudo.

    Retorna
    -------
    pd.DataFrame
        DataFrame preparado listo para ingeniería de características.
    """
    df = add_temporal_features(data)
    df = convert_categorical_types(df)
    return df


def temporal_train_test_split(
    data: pd.DataFrame,
    train_ratio: float = TRAIN_TEST_SPLIT_RATIO
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Divide los datos temporalmente (primeros train_ratio para entrenamiento,
    resto para pruebas).

    Parámetros
    ----------
    data : pd.DataFrame
        DataFrame preparado ordenado por fecha.
    train_ratio : float
        Proporción de datos a usar para entrenamiento.

    Retorna
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        X_train, X_test, y_train, y_test
    """
    logger.info(f"Dividiendo datos con {train_ratio:.0%} para entrenamiento")

    df = data.sort_values("date").reset_index(drop=True)
    n_samples = len(df)
    n_train = int(n_samples * train_ratio)

    X_train = df.iloc[:n_train][FEATURES].copy()
    y_train = df.iloc[:n_train][TARGET].copy()
    X_test = df.iloc[n_train:][FEATURES].copy()
    y_test = df.iloc[n_train:][TARGET].copy()

    logger.info(f"Tamaño entrenamiento: {len(X_train)}, Tamaño prueba: {len(X_test)}")
    return X_train, X_test, y_train, y_test


def get_full_dataset(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Obtiene el dataset completo de características y objetivo.

    Parámetros
    ----------
    data : pd.DataFrame
        DataFrame preparado.

    Retorna
    -------
    Tuple[pd.DataFrame, pd.Series]
        X (características), y (objetivo)
    """
    X = data[FEATURES].copy()
    y = data[TARGET].copy()
    return X, y


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "train.csv",
    output_path: Path = PROCESSED_DATA_DIR / "prepared_data.csv",
):
    """
    Función principal para procesar el dataset crudo y guardar los datos preparados.

    Parámetros
    ----------
    input_path : Path
        Ruta al archivo de datos crudos.
    output_path : Path
        Ruta para guardar los datos procesados.
    """
    logger.info("Procesando dataset...")

    # Cargar y preparar datos
    raw_data = load_raw_data(input_path)
    prepared_data = prepare_dataset(raw_data)

    # Guardar datos preparados
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepared_data.to_csv(output_path, index=False)

    logger.success(f"Procesamiento de dataset completado. Guardado en {output_path}")


if __name__ == "__main__":
    app()
