"""
Script simple para probar la API de predicción de ventas.
Este script ejecuta la API sin dependencias adicionales.
"""
import sys
from pathlib import Path
from datetime import datetime
import json

# Agregar el directorio raíz al path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import joblib
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request

# Configuración
MODELS_DIR = project_root / "models"
PIPELINE_FILE = MODELS_DIR / "sales_pipeline.pkl"
FEATURES = ["store", "item", "year", "month", "day_of_week_name"]

# Crear aplicación Flask
app = Flask(__name__)

# Variable global para el pipeline
_pipeline = None
_hyperparameters = {}


def get_pipeline():
    """Carga el pipeline."""
    global _pipeline, _hyperparameters
    if _pipeline is None:
        print(f"Cargando pipeline desde {PIPELINE_FILE}...")
        _pipeline = joblib.load(PIPELINE_FILE)
        # Extraer hiperparámetros del modelo
        if hasattr(_pipeline, 'steps'):
            model = _pipeline.steps[-1][1]
            try:
                _hyperparameters = model.get_params()
            except:
                _hyperparameters = {}
        print("Pipeline cargado exitosamente!")
    return _pipeline


def prepare_data(data):
    """Prepara los datos para predicción."""
    df = pd.DataFrame(data if isinstance(data, list) else [data])
    
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["day_of_week_name"] = df["date"].dt.day_name()
    
    for col in ["store", "item", "day_of_week_name"]:
        if col in df.columns:
            df[col] = df[col].astype("O")
    
    return df[FEATURES].copy()


def create_response(predictions, timestamp, extra=None):
    """Crea respuesta JSON estandarizada."""
    response = {
        "predictions": predictions,
        "model_metrics": {
            "rmse": 13.0867,
            "mae": 10.2541,
            "r2": 0.9142,
            "mse": 171.2617
        },
        "hyperparameters": _hyperparameters,
        "timestamp": timestamp
    }
    if extra:
        response.update(extra)
    return response


@app.route("/", methods=["GET"])
def home():
    """Endpoint principal."""
    return jsonify({
        "name": "Sales Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/": "Información de la API",
            "/health": "Health check",
            "/predict": "Predicción individual (POST)",
            "/predict/batch": "Predicción batch (POST)"
        },
        "timestamp": datetime.now().isoformat()
    })


@app.route("/health", methods=["GET"])
def health():
    """Health check."""
    try:
        get_pipeline()
        return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict_single():
    """Predicción individual."""
    timestamp = datetime.now().isoformat()
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided", "timestamp": timestamp}), 400
        
        X = prepare_data(data)
        pipeline = get_pipeline()
        prediction = pipeline.predict(X)
        
        return jsonify(create_response(
            predictions=[round(float(prediction[0]), 2)],
            timestamp=timestamp,
            extra={"input": data, "count": 1}
        ))
    except Exception as e:
        return jsonify({"error": str(e), "timestamp": timestamp}), 500


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    """Predicción por lote."""
    timestamp = datetime.now().isoformat()
    try:
        request_data = request.get_json()
        if not request_data or "data" not in request_data:
            return jsonify({"error": "Field 'data' required", "timestamp": timestamp}), 400
        
        data_list = request_data["data"]
        X = prepare_data(data_list)
        pipeline = get_pipeline()
        predictions = pipeline.predict(X)
        
        details = [
            {"index": i, "input": rec, "prediction": round(float(pred), 2)}
            for i, (rec, pred) in enumerate(zip(data_list, predictions))
        ]
        
        return jsonify(create_response(
            predictions=[round(float(p), 2) for p in predictions],
            timestamp=timestamp,
            extra={"count": len(predictions), "details": details}
        ))
    except Exception as e:
        return jsonify({"error": str(e), "timestamp": timestamp}), 500


if __name__ == "__main__":
    print("=" * 50)
    print("SALES PREDICTION API")
    print("=" * 50)
    get_pipeline()  # Pre-cargar el pipeline
    print("\nIniciando servidor en http://localhost:5000")
    print("Presiona Ctrl+C para detener\n")
    app.run(host="0.0.0.0", port=5000, debug=True)
