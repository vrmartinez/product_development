"""
Módulo de configuración para el pipeline de MLOps de product_development.

Este módulo contiene todas las rutas, constantes y parámetros de configuración
utilizados a lo largo del pipeline.
"""
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Cargar variables de entorno desde archivo .env si existe
load_dotenv()

# Rutas del proyecto
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Archivos del dataset
TRAIN_DATA_FILE = RAW_DATA_DIR / "train.csv"
PIPELINE_FILE = MODELS_DIR / "sales_pipeline.pkl"
FEATURE_PIPELINE_FILE = MODELS_DIR / "feature_engineering_pipeline.pkl"
PREDICTIONS_FILE = PROCESSED_DATA_DIR / "test_predictions.csv"

# Configuración de características
TARGET = "sales"
FEATURES = ["store", "item", "year", "month", "day_of_week_name"]
CATEGORICAL_VARS = ["store", "item", "day_of_week_name"]
CATEGORICAL_VARS_IMPUTE = ["store", "item"]
CATEGORICAL_VARS_FREQ = ["store", "item"]
NUMERICAL_VARS = ["year", "month"]

# Mapeo de días de la semana
DAY_OF_WEEK_MAPPING = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6
}

# Configuración de entrenamiento del modelo
TRAIN_TEST_SPLIT_RATIO = 0.8
RANDOM_STATE = 2025

# Configuración de MLflow
MLFLOW_TRACKING_URI = "mlruns"  # URI local para tracking (puede ser un servidor remoto)
MLFLOW_EXPERIMENT_NAME = "sales_prediction"
MLFLOW_MODEL_NAME = "sales_prediction_model"  # Nombre para el Model Registry
MLFLOW_CHAMPION_ALIAS = "champion"  # Alias para el modelo campeón
MLFLOW_CHALLENGER_ALIAS = "challenger"  # Alias para modelos challenger

# Si tqdm está instalado, configurar loguru con tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
