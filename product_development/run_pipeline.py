#!/usr/bin/env python
"""
Script principal del pipeline para el sistema MLOps de predicción de ventas.

Este script ejecuta el pipeline completo de extremo a extremo incluyendo:
1. Carga y preprocesamiento de datos
2. Ingeniería de características
3. Entrenamiento y selección de modelos
4. Entrenamiento del modelo final
5. Generación de inferencias

Uso:
    python run_pipeline.py [OPCIONES]

Opciones:
    --input-path RUTA    Ruta a los datos crudos de entrenamiento
    --output-path RUTA   Ruta para guardar las predicciones
    --skip-training      Omitir entrenamiento del modelo (usar pipeline existente)
"""
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger
import typer

from product_development.config import (
    FEATURES,
    MODELS_DIR,
    PIPELINE_FILE,
    PREDICTIONS_FILE,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    TRAIN_DATA_FILE,
    TRAIN_TEST_SPLIT_RATIO,
)
from product_development.dataset import (
    load_raw_data,
    prepare_dataset,
    temporal_train_test_split,
    get_full_dataset,
)
from product_development.features import (
    build_feature_pipeline,
    fit_feature_pipeline,
    save_feature_pipeline,
    transform_features,
)
from product_development.modeling.train import (
    create_full_pipeline,
    get_model_configurations,
    load_pipeline,
    save_pipeline,
    train_and_evaluate_models,
    train_final_model,
)
from product_development.modeling.predict import (
    predict,
    prepare_inference_data,
    save_predictions,
)

app = typer.Typer()


def run_data_pipeline(input_path: Path = TRAIN_DATA_FILE) -> pd.DataFrame:
    """
    Ejecuta el pipeline de carga y preprocesamiento de datos.

    Parámetros
    ----------
    input_path : Path
        Ruta al archivo de datos crudos.

    Retorna
    -------
    pd.DataFrame
        Dataset preparado.
    """
    logger.info("=" * 60)
    logger.info("PASO 1: CARGA Y PREPROCESAMIENTO DE DATOS")
    logger.info("=" * 60)

    raw_data = load_raw_data(input_path)
    prepared_data = prepare_dataset(raw_data)

    logger.info(f"Dataset preparado: {len(prepared_data)} registros")
    return prepared_data


def run_feature_pipeline(
    prepared_data: pd.DataFrame,
    train_ratio: float = TRAIN_TEST_SPLIT_RATIO
) -> tuple:
    """
    Ejecuta el pipeline de ingeniería de características.

    Parámetros
    ----------
    prepared_data : pd.DataFrame
        Dataset preparado.
    train_ratio : float
        Ratio de división entrenamiento/prueba.

    Retorna
    -------
    tuple
        Pipeline de características, datos de entrenamiento/validación transformados.
    """
    logger.info("=" * 60)
    logger.info("PASO 2: INGENIERÍA DE CARACTERÍSTICAS")
    logger.info("=" * 60)

    # Dividir datos temporalmente
    X_train, X_val, y_train, y_val = temporal_train_test_split(
        prepared_data, train_ratio
    )

    # Construir y ajustar pipeline de características
    feature_pipeline = build_feature_pipeline()
    feature_pipeline = fit_feature_pipeline(feature_pipeline, X_train, y_train)

    # Guardar pipeline de características
    save_feature_pipeline(feature_pipeline)

    # Transformar datos
    X_train_transformed = transform_features(feature_pipeline, X_train)
    X_val_transformed = transform_features(feature_pipeline, X_val)

    logger.info("Ingeniería de características completada")

    return (
        feature_pipeline,
        X_train_transformed,
        X_val_transformed,
        y_train,
        y_val
    )


def run_training_pipeline(
    feature_pipeline,
    X_train_transformed,
    X_val_transformed,
    y_train,
    y_val,
    prepared_data: pd.DataFrame
) -> tuple:
    """
    Ejecuta el pipeline de entrenamiento de modelos.

    Parámetros
    ----------
    feature_pipeline : Pipeline
        Pipeline de ingeniería de características ajustado.
    X_train_transformed : np.ndarray
        Características de entrenamiento transformadas.
    X_val_transformed : np.ndarray
        Características de validación transformadas.
    y_train : pd.Series
        Objetivo de entrenamiento.
    y_val : pd.Series
        Objetivo de validación.
    prepared_data : pd.DataFrame
        Dataset completo preparado.

    Retorna
    -------
    tuple
        Diccionario de resultados, nombre del mejor modelo, pipeline entrenado.
    """
    logger.info("=" * 60)
    logger.info("PASO 3: ENTRENAMIENTO Y SELECCIÓN DE MODELOS")
    logger.info("=" * 60)

    # Obtener configuraciones de modelos
    models = get_model_configurations()

    # Entrenar y evaluar modelos
    results, best_model_name, best_model = train_and_evaluate_models(
        X_train_transformed, y_train,
        X_val_transformed, y_val,
        models
    )

    # Registrar resultados
    logger.info("Resultados de evaluación de modelos:")
    for model_name, rmse in sorted(results.items(), key=lambda x: x[1]):
        logger.info(f"  {model_name}: RMSE = {rmse}")

    logger.info("=" * 60)
    logger.info("PASO 4: ENTRENAMIENTO DEL MODELO FINAL")
    logger.info("=" * 60)

    # Crear pipeline completo con el mejor modelo
    full_pipeline = create_full_pipeline(feature_pipeline, best_model)

    # Obtener dataset completo
    X_full, y_full = get_full_dataset(prepared_data)

    # Entrenar con dataset completo
    full_pipeline = train_final_model(full_pipeline, X_full, y_full)

    # Guardar pipeline
    save_pipeline(full_pipeline)

    logger.info(f"Mejor modelo: {best_model_name}")

    return results, best_model_name, full_pipeline


def run_inference_pipeline(
    pipeline,
    data: pd.DataFrame,
    output_path: Path = PREDICTIONS_FILE
) -> pd.DataFrame:
    """
    Ejecuta el pipeline de inferencia.

    Parámetros
    ----------
    pipeline : Pipeline
        Pipeline entrenado.
    data : pd.DataFrame
        Datos para inferencia.
    output_path : Path
        Ruta para guardar predicciones.

    Retorna
    -------
    pd.DataFrame
        DataFrame con predicciones.
    """
    logger.info("=" * 60)
    logger.info("PASO 5: INFERENCIA")
    logger.info("=" * 60)

    # Preparar datos para inferencia
    X = prepare_inference_data(data)

    # Generar predicciones
    predictions = predict(pipeline, X)

    # Guardar predicciones
    result = save_predictions(data, predictions, output_path)

    logger.info(f"Predicciones guardadas en {output_path}")

    return result


@app.command()
def main(
    input_path: Path = TRAIN_DATA_FILE,
    output_path: Path = PREDICTIONS_FILE,
    skip_training: bool = False,
    inference_data_path: Optional[Path] = None,
):
    """
    Ejecuta el pipeline completo de MLOps.

    Este comando ejecuta el pipeline completo de extremo a extremo incluyendo:
    - Carga y preprocesamiento de datos
    - Ingeniería de características
    - Entrenamiento y selección de modelos
    - Entrenamiento del modelo final
    - Generación de inferencias

    Parámetros
    ----------
    input_path : Path
        Ruta a los datos crudos de entrenamiento.
    output_path : Path
        Ruta para guardar las predicciones.
    skip_training : bool
        Si es True, omite entrenamiento y usa pipeline existente.
    inference_data_path : Path, opcional
        Ruta a los datos para inferencia.
    """
    start_time = time.time()

    logger.info("=" * 60)
    logger.info("PIPELINE DE MLOPS PARA PREDICCIÓN DE VENTAS")
    logger.info("=" * 60)
    logger.info(f"Datos de entrada: {input_path}")
    logger.info(f"Predicciones de salida: {output_path}")

    if skip_training:
        logger.info("Omitiendo entrenamiento, usando pipeline existente")

        # Cargar pipeline existente
        pipeline = load_pipeline()

        # Cargar datos para inferencia
        if inference_data_path:
            inference_data = pd.read_csv(inference_data_path, parse_dates=["date"])
        else:
            raw_data = load_raw_data(input_path)
            prepared_data = prepare_dataset(raw_data)
            n_samples = len(prepared_data)
            n_train = int(n_samples * TRAIN_TEST_SPLIT_RATIO)
            inference_data = prepared_data.iloc[n_train:].copy()

        # Ejecutar inferencia
        run_inference_pipeline(pipeline, inference_data, output_path)

    else:
        # Ejecutar pipeline completo
        # Paso 1: Carga de datos
        prepared_data = run_data_pipeline(input_path)

        # Paso 2: Ingeniería de características
        (
            feature_pipeline,
            X_train_transformed,
            X_val_transformed,
            y_train,
            y_val
        ) = run_feature_pipeline(prepared_data)

        # Pasos 3 y 4: Entrenamiento de modelos
        results, best_model_name, full_pipeline = run_training_pipeline(
            feature_pipeline,
            X_train_transformed,
            X_val_transformed,
            y_train,
            y_val,
            prepared_data
        )

        # Paso 5: Inferencia en conjunto de prueba
        n_samples = len(prepared_data)
        n_train = int(n_samples * TRAIN_TEST_SPLIT_RATIO)
        test_data = prepared_data.iloc[n_train:].copy()

        run_inference_pipeline(full_pipeline, test_data, output_path)

    # Resumen
    elapsed_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETADO")
    logger.info("=" * 60)
    logger.info(f"Tiempo total de ejecución: {elapsed_time:.2f} segundos")
    logger.success("¡Ejecución del pipeline completada exitosamente!")


if __name__ == "__main__":
    app()
