# Product Development - Predicción de Ventas

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DVC](https://img.shields.io/badge/DVC-Pipeline-945DD6?logo=dvc)](https://dvc.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?logo=mlflow)](https://mlflow.org/)

**Proyecto del Curso - Pipeline de MLOps para Predicción de Ventas**

---

## 📋 Descripción

Este proyecto implementa un **pipeline completo de MLOps** para la predicción de ventas utilizando técnicas de machine learning. El sistema está diseñado siguiendo las mejores prácticas de ciencia de datos e ingeniería de software, incluyendo:

- ✅ Análisis exploratorio de datos (EDA)
- ✅ Ingeniería de características automatizada
- ✅ Entrenamiento y selección de modelos
- ✅ Pipeline de inferencia reproducible
- ✅ Arquitectura modular y escalable
- ✅ **API REST** para predicciones en tiempo real
- ✅ **MLflow** para tracking de experimentos y model registry
- ✅ **DVC** para versionado de datos y pipelines reproducibles

---

## 🎯 Objetivo del Proyecto

Desarrollar un sistema de predicción de ventas que permita:
1. Procesar datos históricos de ventas por tienda y artículo
2. Generar características predictivas automáticamente
3. Entrenar y evaluar múltiples modelos de machine learning
4. Producir predicciones confiables para la planificación de inventario

---

## 📁 Organización del Proyecto

```
product_development/
│
├── 📄 LICENSE                 <- Licencia de código abierto (MIT)
├── 📄 Makefile                <- Comandos útiles (make data, make train, etc.)
├── 📄 README.md               <- Documentación principal del proyecto
├── 📄 pyproject.toml          <- Configuración del proyecto y dependencias
├── 📄 environment.yml         <- Entorno conda con todas las dependencias
├── 📄 dvc.yaml                <- Definición del pipeline DVC
├── 📄 params.yaml             <- Parámetros configurables del pipeline
│
├── 📂 data/                   <- Datos del proyecto
│   ├── external/              <- Datos de fuentes externas
│   ├── processed/             <- Datos procesados listos para modelado
│   │   ├── prepared_data.csv  <- Dataset con características temporales
│   │   ├── preproc_train.csv  <- Dataset preprocesado
│   │   └── test_predictions.csv <- Predicciones generadas
│   └── raw/                   <- Datos originales (inmutables)
│       └── train.csv          <- Dataset de entrenamiento (date, store, item, sales)
│
├── 📂 docs/                   <- Documentación adicional del proyecto
│   └── DVC_PIPELINE.md        <- Documentación del pipeline DVC
│
├── 📂 models/                 <- Modelos entrenados serializados
│   ├── feature_engineering_pipeline.pkl  <- Pipeline de ingeniería de características
│   └── sales_pipeline.pkl                <- Pipeline completo de predicción
│
├── 📂 mlruns/                 <- Directorio de MLflow para tracking
│   └── ...                    <- Experimentos, métricas y artefactos
│
├── 📂 notebooks/              <- Jupyter notebooks del flujo de trabajo
│   ├── 01_Data_Exploration.ipynb      <- EDA: análisis y visualizaciones
│   ├── 02_feature_exploration.ipynb   <- Exploración de características
│   ├── 03_feature_creation.ipynb      <- Creación de features con sklearn
│   ├── 04_model_tuning_training.ipynb <- Ajuste y entrenamiento de modelos
│   ├── 05_inference_calculation.ipynb <- Cálculo de predicciones
│   └── operators.py                   <- Transformadores para notebooks
│
├── 📂 references/             <- Diccionarios de datos y materiales de referencia
│
├── 📂 reports/                <- Reportes y análisis generados
│   ├── metrics.json           <- Métricas del modelo (generado por DVC)
│   └── figures/               <- Gráficos y figuras para reportes
│
├── 📂 tests/                  <- Pruebas unitarias
│   ├── test_data.py           <- Tests de validación de datos
│   └── test_api_examples.py   <- Ejemplos de consumo de la API
│
├── 📂 scripts/                <- Scripts auxiliares
│   ├── dvc_train.py           <- Script de entrenamiento para DVC
│   ├── dvc_inference.py       <- Script de inferencia para DVC
│   ├── run_api.py             <- Script para ejecutar la API
│   └── run_pipeline.py        <- Script principal del pipeline MLOps
│
└── 📂 product_development/    <- 📦 Código fuente del paquete
    │
    ├── __init__.py            <- Inicialización del módulo Python
    ├── config.py              <- Configuración de rutas y constantes
    ├── dataset.py             <- Funciones de carga y preparación de datos
    ├── features.py            <- Pipeline de ingeniería de características
    ├── plots.py               <- Funciones de visualización
    ├── transformers.py        <- Transformadores personalizados de sklearn
    ├── api.py                 <- 🌐 API REST Flask para predicciones
    │
    └── modeling/              <- Submódulo de modelado
        ├── __init__.py
        ├── train.py           <- Entrenamiento y evaluación de modelos
        └── predict.py         <- Inferencia y generación de predicciones
```

---

## 🔄 Flujo de Trabajo

El proyecto sigue un flujo de trabajo estructurado en 5 etapas:

### 1️⃣ Exploración de Datos
**Notebook:** `01_Data_Exploration.ipynb`

- Análisis exploratorio del dataset de ventas
- Estadísticas descriptivas por tienda y artículo
- Visualizaciones de series temporales
- Identificación de patrones y tendencias

### 2️⃣ Exploración de Características
**Notebook:** `02_feature_exploration.ipynb`

- Análisis de correlaciones
- Evaluación de variables candidatas
- Selección de características relevantes

### 3️⃣ Creación de Características
**Notebook:** `03_feature_creation.ipynb`

Pipeline de ingeniería de características que incluye:
- 📊 **Features de Lag**: 1, 7, 14, 28 días
- 📈 **Medias Móviles**: ventanas de 7 y 28 días
- 🏷️ **Codificación de Frecuencia**: para tiendas e items
- 📅 **Features Temporales**: año, mes, día de la semana
- ⚖️ **Escalado MinMax**: normalización de características

### 4️⃣ Entrenamiento del Modelo
**Notebook:** `04_model_tuning_training.ipynb`

Evaluación de múltiples algoritmos:
- Regresión Lineal
- Random Forest
- Gradient Boosting
- Support Vector Regression (SVR)
- XGBoost

### 5️⃣ Inferencia
**Notebook:** `05_inference_calculation.ipynb`

- Carga del pipeline entrenado
- Generación de predicciones
- Evaluación de métricas (RMSE)

---

## 🔄 Pipeline DVC

El proyecto incluye un **pipeline DVC** para automatizar y reproducir el flujo de trabajo completo.

### Estructura del Pipeline

```
prepare_data → feature_engineering → train_model → inference
```

| Etapa | Entrada | Salida | Descripción |
|-------|---------|--------|-------------|
| `prepare_data` | `data/raw/train.csv` | `data/processed/prepared_data.csv` | Carga datos y agrega features temporales |
| `feature_engineering` | `prepared_data.csv` | `feature_engineering_pipeline.pkl` | Construye pipeline de características |
| `train_model` | `prepared_data.csv`, `pipeline.pkl` | `sales_pipeline.pkl`, `metrics.json` | Entrena y evalúa modelos |
| `inference` | `prepared_data.csv`, `sales_pipeline.pkl` | `test_predictions.csv` | Genera predicciones |

### Comandos DVC

```bash
# Ejecutar todo el pipeline
dvc repro

# Ejecutar una etapa específica
dvc repro train_model

# Ver estado del pipeline
dvc status

# Ver grafo de dependencias
dvc dag

# Ver métricas
dvc metrics show

# Comparar métricas entre experimentos
dvc metrics diff
```

### Parámetros del Pipeline (`params.yaml`)

```yaml
# Configuración de división de datos
data:
  train_test_split_ratio: 0.8    # Proporción train/test
  random_state: 2025             # Semilla para reproducibilidad

# Configuración de características
features:
  target: "sales"
  feature_columns:
    - "store"
    - "item"
    - "year"
    - "month"
    - "day_of_week_name"
  categorical_vars:
    - "store"
    - "item"
    - "day_of_week_name"
  numerical_vars:
    - "year"
    - "month"

# Configuración de entrenamiento
training:
  mode: "fast"                   # "fast" o "full"
  use_mlflow: true               # Registrar en MLflow

# Configuración de MLflow
mlflow:
  tracking_uri: "mlruns"
  experiment_name: "sales_prediction"
  model_name: "sales_prediction_model"
```

> 📖 Para más detalles, consulta [docs/DVC_PIPELINE.md](docs/DVC_PIPELINE.md)

---

## 🌐 API REST

El proyecto incluye una **API REST** construida con Flask para realizar predicciones en tiempo real.

### Iniciar la API

```bash
# Opción 1: Usando el script de scripts/
python scripts/run_api.py

# Opción 2: Con opciones personalizadas
python scripts/run_api.py --host 0.0.0.0 --port 5000 --debug

# Opción 3: Ejecutando directamente el módulo api
python -m product_development.api
```

### Endpoints Disponibles

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/` | GET | Información de la API |
| `/health` | GET | Health check del servicio |
| `/model/info` | GET | Información del modelo (métricas, hiperparámetros) |
| `/predict` | POST | Predicción individual |
| `/predict/batch` | POST | Predicción por lote |

### Ejemplos de Uso

#### Predicción Individual

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"store": 1, "item": 1, "date": "2018-01-15"}'
```

**Respuesta:**
```json
{
  "predictions": [42.35],
  "model_metrics": {"rmse": 13.08, "mae": 10.25, "r2": 0.91},
  "timestamp": "2024-01-15T10:30:00",
  "prediction_count": 1
}
```

#### Predicción por Lote (Batch)

```bash
curl -X POST http://localhost:5000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {"store": 1, "item": 1, "date": "2018-01-15"},
      {"store": 2, "item": 3, "date": "2018-01-16"},
      {"store": 5, "item": 10, "date": "2018-02-01"}
    ]
  }'
```

#### Usando Python

```python
import requests

# Predicción individual
response = requests.post(
    "http://localhost:5000/predict",
    json={"store": 1, "item": 1, "date": "2018-01-15"}
)
print(response.json())

# Predicción batch
response = requests.post(
    "http://localhost:5000/predict/batch",
    json={
        "data": [
            {"store": 1, "item": 1, "date": "2018-01-15"},
            {"store": 2, "item": 3, "date": "2018-01-16"}
        ]
    }
)
print(response.json())
```

### Probar la API

```bash
# Ejecutar ejemplos de prueba
python tests/test_api_examples.py
```

---

## 🚀 Inicio Rápido

```bash
# 1. Clonar el repositorio
git clone https://github.com/franciscogonzalez-gal/product_development.git
cd product_development

# 2. Crear entorno e instalar dependencias
conda env create -f environment.yml
conda activate product_development
pip install -e .

# 3. Ejecutar el pipeline DVC
dvc repro

# 4. Ver métricas
dvc metrics show

# 5. Iniciar la API
python scripts/run_api.py
```

---

## 🚀 Instalación

### Prerrequisitos
- Python 3.11
- Conda (recomendado) o pip

### Opción 1: Usando Conda (Recomendado)

```bash
# Clonar el repositorio
git clone https://github.com/franciscogonzalez-gal/product_development.git
cd product_development

# Crear entorno conda desde environment.yml
conda env create -f environment.yml

# Activar entorno
conda activate product_development

# Instalar el paquete en modo desarrollo
pip install -e .
```

### Opción 2: Usando pip

```bash
# Clonar el repositorio
git clone https://github.com/franciscogonzalez-gal/product_development.git
cd product_development

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar el paquete en modo desarrollo
pip install -e .
```

---

## 💻 Uso

### Ejecutar el Pipeline Completo

```bash
# Ejecutar pipeline completo (entrenamiento + inferencia)
python scripts/run_pipeline.py

# Solo inferencia (usando modelo existente)
python scripts/run_pipeline.py --skip-training

# Especificar rutas personalizadas
python scripts/run_pipeline.py \
    --input-path data/raw/train.csv \
    --output-path data/processed/predictions.csv
```

### Opciones del Pipeline

| Opción | Descripción |
|--------|-------------|
| `--input-path`, `-i` | Ruta a los datos crudos de entrenamiento |
| `--output-path`, `-o` | Ruta para guardar las predicciones |
| `--skip-training`, `-s` | Omitir entrenamiento y usar pipeline existente |
| `--inference-data`, `-d` | Ruta a datos para inferencia |

### Usar como Biblioteca

```python
from product_development.dataset import load_raw_data, prepare_dataset
from product_development.features import build_feature_pipeline
from product_development.modeling.train import train_and_evaluate_models
from product_development.modeling.predict import load_and_predict

# Cargar y preparar datos
data = load_raw_data()
prepared_data = prepare_dataset(data)

# Generar predicciones con modelo existente
predictions = load_and_predict(prepared_data)
```

---

## 📊 MLflow - Tracking de Experimentos

El proyecto utiliza **MLflow** para el seguimiento de experimentos y registro de modelos.

### Configuración de MLflow

```python
# En config.py
MLFLOW_TRACKING_URI = "mlruns"           # URI del servidor de tracking
MLFLOW_EXPERIMENT_NAME = "sales_prediction"
MLFLOW_MODEL_NAME = "sales_prediction_model"
MLFLOW_CHAMPION_ALIAS = "champion"       # Alias del modelo en producción
```

### Ver Experimentos

```bash
# Iniciar la UI de MLflow
mlflow ui --backend-store-uri mlruns

# Abrir en el navegador: http://localhost:5000
```

### Características de MLflow en el Proyecto

- 📈 **Tracking de métricas**: RMSE, MAE, R², MSE
- 🔧 **Registro de hiperparámetros**: Parámetros del modelo
- 📦 **Model Registry**: Gestión de versiones de modelos
- 🏷️ **Aliases**: Champion/Challenger para promoción de modelos

---

## 🛠️ Tecnologías Utilizadas

| Categoría | Tecnologías |
|-----------|-------------|
| **Lenguaje** | Python 3.11 |
| **Manipulación de Datos** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn, XGBoost |
| **Ingeniería de Características** | Feature-engine |
| **Visualización** | Matplotlib, Seaborn |
| **Análisis Estadístico** | Statsmodels |
| **API REST** | Flask |
| **MLOps** | MLflow, DVC |
| **CLI** | Typer |
| **Logging** | Loguru |
| **Serialización** | Joblib |
| **Configuración** | python-dotenv, PyYAML |

---

## 📊 Estructura de Datos

### Dataset de Entrada (`train.csv`)

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `date` | datetime | Fecha de la venta |
| `store` | int | Identificador de la tienda |
| `item` | int | Identificador del artículo |
| `sales` | int | Cantidad de ventas |

### Características Generadas

| Característica | Descripción |
|----------------|-------------|
| `year` | Año extraído de la fecha |
| `month` | Mes extraído de la fecha |
| `day_of_week_name` | Nombre del día de la semana |
| `store` (encoded) | Tienda codificada por frecuencia |
| `item` (encoded) | Artículo codificado por frecuencia |

---

## 🧪 Pruebas

```bash
# Ejecutar todas las pruebas
pytest tests/

# Ejecutar con cobertura
pytest tests/ --cov=product_development

# Ejecutar pruebas de datos
pytest tests/test_data.py

# Probar ejemplos de la API (requiere que la API esté corriendo)
python tests/test_api_examples.py
```

---

## 🛠️ Comandos Makefile

El proyecto incluye un `Makefile` con comandos útiles:

```bash
# Ver todos los comandos disponibles
make help

# Instalar dependencias
make requirements

# Ejecutar pipeline completo
make pipeline

# Solo inferencia (usando modelo existente)
make inference

# Entrenar modelo
make train

# Ejecutar pruebas
make test

# Formatear código
make format

# Linting
make lint

# Ejecutar pylint
make pylint

# Limpiar archivos compilados
make clean

# Crear entorno conda
make create_environment
```

---

## 📈 Métricas de Evaluación

El modelo se evalúa utilizando:
- **RMSE** (Root Mean Square Error): Métrica principal de evaluación
- **MAE** (Mean Absolute Error): Error absoluto promedio
- **R²** (Coeficiente de determinación): Varianza explicada
- **MSE** (Mean Square Error): Error cuadrático medio

### Modelos Evaluados

El pipeline evalúa automáticamente los siguientes modelos:
- Regresión Lineal
- Random Forest
- Gradient Boosting
- XGBoost

El mejor modelo se selecciona automáticamente basándose en RMSE y se registra en MLflow.

---

## 👥 Autores

- **Francisco González** - [franciscogonzalez-gal](https://github.com/franciscogonzalez-gal) - Universidad Galileo

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

---

## 🙏 Agradecimientos

- [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/) por la plantilla del proyecto
- Universidad Galileo por el soporte académico

---

<p align="center">
  <i>Desarrollado con ❤️ para el curso de Desarrollo de Producto</i>
</p>