"""
API Flask para predicción de ventas.

Este módulo proporciona una API REST para realizar predicciones
de ventas tanto individuales como por lote (batch).
"""
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from flask import Flask, jsonify, request
from loguru import logger

from product_development.config import (
    FEATURES,
    PIPELINE_FILE,
    MLFLOW_MODEL_NAME,
    MLFLOW_CHAMPION_ALIAS,
)
from product_development.modeling.train import (
    load_pipeline,
    calculate_metrics,
    get_model_hyperparameters,
)
from product_development.modeling.predict import prepare_inference_data

# Crear aplicación Flask
app = Flask(__name__)

# Variables globales para el pipeline y metadata
_pipeline = None
_model_metrics: Dict[str, float] = {}
_model_hyperparameters: Dict[str, Any] = {}


def get_pipeline():
    """
    Carga el pipeline de manera lazy (singleton).
    
    Retorna
    -------
    Pipeline
        Pipeline de predicción cargado.
    """
    global _pipeline, _model_hyperparameters
    
    if _pipeline is None:
        logger.info("Cargando pipeline de predicción...")
        _pipeline = load_pipeline(PIPELINE_FILE)
        
        # Extraer hiperparámetros del modelo (último paso del pipeline)
        if hasattr(_pipeline, 'steps'):
            model_step = _pipeline.steps[-1]
            if len(model_step) > 1:
                model = model_step[1]
                _model_hyperparameters = get_model_hyperparameters(model)
        
        logger.info("Pipeline cargado exitosamente")
    
    return _pipeline


def set_model_metrics(metrics: Dict[str, float]):
    """
    Establece las métricas del modelo.
    
    Parámetros
    ----------
    metrics : Dict[str, float]
        Diccionario con las métricas del modelo.
    """
    global _model_metrics
    _model_metrics = metrics


def get_model_metrics() -> Dict[str, float]:
    """
    Obtiene las métricas del modelo.
    
    Retorna
    -------
    Dict[str, float]
        Diccionario con las métricas del modelo.
    """
    return _model_metrics


def get_model_hyperparameters_cached() -> Dict[str, Any]:
    """
    Obtiene los hiperparámetros del modelo cacheados.
    
    Retorna
    -------
    Dict[str, Any]
        Diccionario con los hiperparámetros del modelo.
    """
    return _model_hyperparameters


def create_response(
    predictions: List[float],
    request_timestamp: str,
    additional_info: Optional[Dict] = None
) -> Dict:
    """
    Crea la respuesta JSON estandarizada.
    
    Parámetros
    ----------
    predictions : List[float]
        Lista de predicciones.
    request_timestamp : str
        Timestamp de la solicitud.
    additional_info : Dict, opcional
        Información adicional para incluir.
    
    Retorna
    -------
    Dict
        Respuesta JSON estandarizada.
    """
    response = {
        "predictions": predictions,
        "model_metrics": get_model_metrics(),
        "hyperparameters": get_model_hyperparameters_cached(),
        "timestamp": request_timestamp,
        "model_info": {
            "name": MLFLOW_MODEL_NAME,
            "alias": MLFLOW_CHAMPION_ALIAS,
            "features_used": FEATURES
        }
    }
    
    if additional_info:
        response.update(additional_info)
    
    return response


def validate_input_data(data: Dict) -> tuple:
    """
    Valida los datos de entrada para predicción.
    
    Parámetros
    ----------
    data : Dict
        Datos de entrada a validar.
    
    Retorna
    -------
    tuple
        (is_valid, error_message)
    """
    required_fields = ["store", "item", "date"]
    
    missing_fields = [field for field in required_fields if field not in data]
    
    if missing_fields:
        return False, f"Campos requeridos faltantes: {missing_fields}"
    
    return True, None


@app.route("/", methods=["GET"])
def home():
    """Endpoint principal con información de la API."""
    return jsonify({
        "name": "Sales Prediction API",
        "version": "1.0.0",
        "description": "API para predicción de ventas individuales y por lote",
        "endpoints": {
            "/": "Información de la API",
            "/health": "Estado de salud de la API",
            "/predict": "Predicción individual (POST)",
            "/predict/batch": "Predicción por lote (POST)",
            "/model/info": "Información del modelo (GET)"
        },
        "timestamp": datetime.now().isoformat()
    })


@app.route("/health", methods=["GET"])
def health_check():
    """Endpoint de health check."""
    try:
        # Verificar que el pipeline se puede cargar
        pipeline = get_pipeline()
        status = "healthy" if pipeline is not None else "unhealthy"
    except Exception as e:
        status = "unhealthy"
        logger.error(f"Health check failed: {e}")
    
    return jsonify({
        "status": status,
        "timestamp": datetime.now().isoformat()
    })


@app.route("/model/info", methods=["GET"])
def model_info():
    """Endpoint para obtener información del modelo."""
    try:
        pipeline = get_pipeline()
        
        # Obtener información de los pasos del pipeline
        pipeline_steps = []
        if hasattr(pipeline, 'steps'):
            for step_name, step in pipeline.steps:
                pipeline_steps.append({
                    "name": step_name,
                    "type": type(step).__name__
                })
        
        return jsonify({
            "model_name": MLFLOW_MODEL_NAME,
            "model_alias": MLFLOW_CHAMPION_ALIAS,
            "features": FEATURES,
            "pipeline_steps": pipeline_steps,
            "hyperparameters": get_model_hyperparameters_cached(),
            "metrics": get_model_metrics(),
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return jsonify({
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500


@app.route("/predict", methods=["POST"])
def predict_single():
    """
    Endpoint para predicción individual.
    
    Espera un JSON con los campos:
    - store: ID de la tienda
    - item: ID del item
    - date: Fecha en formato YYYY-MM-DD
    
    Ejemplo de request:
    {
        "store": 1,
        "item": 1,
        "date": "2018-01-01"
    }
    """
    request_timestamp = datetime.now().isoformat()
    
    try:
        # Obtener datos del request
        data = request.get_json()
        
        if data is None:
            return jsonify({
                "error": "No se proporcionaron datos JSON",
                "timestamp": request_timestamp
            }), 400
        
        # Validar datos de entrada
        is_valid, error_message = validate_input_data(data)
        if not is_valid:
            return jsonify({
                "error": error_message,
                "timestamp": request_timestamp
            }), 400
        
        # Crear DataFrame con un solo registro
        df = pd.DataFrame([data])
        
        # Preparar datos para inferencia
        X = prepare_inference_data(df)
        
        # Cargar pipeline y hacer predicción
        pipeline = get_pipeline()
        prediction = pipeline.predict(X)
        
        # Crear respuesta
        response = create_response(
            predictions=[round(float(prediction[0]), 2)],
            request_timestamp=request_timestamp,
            additional_info={
                "input_data": data,
                "prediction_count": 1
            }
        )
        
        logger.info(f"Predicción individual realizada: {prediction[0]:.2f}")
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error en predicción individual: {e}")
        return jsonify({
            "error": str(e),
            "timestamp": request_timestamp
        }), 500


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    """
    Endpoint para predicción por lote (batch).
    
    Espera un JSON con una lista de registros:
    {
        "data": [
            {"store": 1, "item": 1, "date": "2018-01-01"},
            {"store": 1, "item": 2, "date": "2018-01-01"},
            ...
        ]
    }
    """
    request_timestamp = datetime.now().isoformat()
    
    try:
        # Obtener datos del request
        request_data = request.get_json()
        
        if request_data is None:
            return jsonify({
                "error": "No se proporcionaron datos JSON",
                "timestamp": request_timestamp
            }), 400
        
        # Verificar que hay datos
        if "data" not in request_data:
            return jsonify({
                "error": "El campo 'data' es requerido con una lista de registros",
                "timestamp": request_timestamp
            }), 400
        
        data_list = request_data["data"]
        
        if not isinstance(data_list, list) or len(data_list) == 0:
            return jsonify({
                "error": "El campo 'data' debe ser una lista no vacía",
                "timestamp": request_timestamp
            }), 400
        
        # Validar cada registro
        for i, record in enumerate(data_list):
            is_valid, error_message = validate_input_data(record)
            if not is_valid:
                return jsonify({
                    "error": f"Error en registro {i}: {error_message}",
                    "timestamp": request_timestamp
                }), 400
        
        # Crear DataFrame con todos los registros
        df = pd.DataFrame(data_list)
        
        # Preparar datos para inferencia
        X = prepare_inference_data(df)
        
        # Cargar pipeline y hacer predicciones
        pipeline = get_pipeline()
        predictions = pipeline.predict(X)
        
        # Crear lista de predicciones con sus inputs
        predictions_with_inputs = []
        for i, (record, pred) in enumerate(zip(data_list, predictions)):
            predictions_with_inputs.append({
                "index": i,
                "input": record,
                "prediction": round(float(pred), 2)
            })
        
        # Crear respuesta
        response = create_response(
            predictions=[round(float(p), 2) for p in predictions],
            request_timestamp=request_timestamp,
            additional_info={
                "prediction_count": len(predictions),
                "predictions_detail": predictions_with_inputs
            }
        )
        
        logger.info(f"Predicción batch realizada: {len(predictions)} registros")
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error en predicción batch: {e}")
        return jsonify({
            "error": str(e),
            "timestamp": request_timestamp
        }), 500


def run_api(
    host: str = "0.0.0.0",
    port: int = 5000,
    debug: bool = False,
    metrics: Optional[Dict[str, float]] = None
):
    """
    Ejecuta la API Flask.
    
    Parámetros
    ----------
    host : str
        Host donde ejecutar la API.
    port : int
        Puerto donde ejecutar la API.
    debug : bool
        Si True, ejecuta en modo debug.
    metrics : Dict[str, float], opcional
        Métricas del modelo para mostrar en las respuestas.
    """
    if metrics:
        set_model_metrics(metrics)
    
    # Pre-cargar el pipeline
    logger.info("Pre-cargando pipeline...")
    get_pipeline()
    
    logger.info(f"Iniciando API en http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    # Métricas de ejemplo (deberían venir del entrenamiento)
    example_metrics = {
        "rmse": 0.0,
        "mae": 0.0,
        "r2": 0.0,
        "mse": 0.0
    }
    
    run_api(debug=True, metrics=example_metrics)
