# Product Development - Predicción de Ventas

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Proyecto del Curso - Pipeline de MLOps para Predicción de Ventas**

---

## 📋 Descripción

Este proyecto implementa un **pipeline completo de MLOps** para la predicción de ventas utilizando técnicas de machine learning. El sistema está diseñado siguiendo las mejores prácticas de ciencia de datos e ingeniería de software, incluyendo:

- ✅ Análisis exploratorio de datos (EDA)
- ✅ Ingeniería de características automatizada
- ✅ Entrenamiento y selección de modelos
- ✅ Pipeline de inferencia reproducible
- ✅ Arquitectura modular y escalable

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
│
├── 📂 data/                   <- Datos del proyecto
│   ├── external/              <- Datos de fuentes externas
│   ├── processed/             <- Datos procesados listos para modelado
│   │   ├── preproc_train.csv  <- Dataset preprocesado
│   │   └── test_predictions.csv <- Predicciones generadas
│   └── raw/                   <- Datos originales (inmutables)
│       └── train.csv          <- Dataset de entrenamiento (date, store, item, sales)
│
├── 📂 docs/                   <- Documentación adicional del proyecto
│
├── 📂 models/                 <- Modelos entrenados serializados
│   ├── feature_engineering_pipeline.pkl  <- Pipeline de ingeniería de características
│   └── sales_pipeline.pkl                <- Pipeline completo de predicción
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
│   └── figures/               <- Gráficos y figuras para reportes
│
├── 📂 tests/                  <- Pruebas unitarias
│   └── test_data.py           <- Tests de validación de datos
│
└── 📂 product_development/    <- 📦 Código fuente del paquete
    │
    ├── __init__.py            <- Inicialización del módulo Python
    ├── config.py              <- Configuración de rutas y constantes
    ├── dataset.py             <- Funciones de carga y preparación de datos
    ├── features.py            <- Pipeline de ingeniería de características
    ├── plots.py               <- Funciones de visualización
    ├── transformers.py        <- Transformadores personalizados de sklearn
    ├── run_pipeline.py        <- Script principal del pipeline MLOps
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
python -m product_development.run_pipeline

# Solo inferencia (usando modelo existente)
python -m product_development.run_pipeline --skip-training

# Especificar rutas personalizadas
python -m product_development.run_pipeline \
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

## 🛠️ Tecnologías Utilizadas

| Categoría | Tecnologías |
|-----------|-------------|
| **Lenguaje** | Python 3.11 |
| **Manipulación de Datos** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn, XGBoost |
| **Ingeniería de Características** | Feature-engine |
| **Visualización** | Matplotlib, Seaborn |
| **Análisis Estadístico** | Statsmodels |
| **CLI** | Typer |
| **Logging** | Loguru |
| **Serialización** | Joblib |

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
```

---

## 📈 Métricas de Evaluación

El modelo se evalúa utilizando:
- **RMSE** (Root Mean Square Error): Métrica principal de evaluación
- Comparación de predicciones vs valores reales

---

## 👥 Autores

- **Galileo Team** - Universidad Galileo

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