# product_development

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Proyecto del Curso - Predicción de Ventas

## Descripción

Este proyecto implementa un pipeline de ciencia de datos para la predicción de ventas utilizando técnicas de machine learning. Incluye análisis exploratorio de datos (EDA), ingeniería de características, entrenamiento de modelos y generación de predicciones.

## Organización del Proyecto

```
├── LICENSE            <- Licencia de código abierto
├── Makefile           <- Makefile con comandos útiles como `make data` o `make train`
├── README.md          <- README principal para desarrolladores que usen este proyecto.
├── data
│   ├── external       <- Datos de fuentes externas.
│   ├── processed      <- Conjuntos de datos finales y canónicos para modelado.
│   └── raw            <- Datos originales e inmutables.
│       ├── train.csv            <- Dataset de entrenamiento (date, store, item, sales)
│       ├── preproc_train.csv    <- Dataset preprocesado
│       └── test_predictions.csv <- Predicciones del modelo
│
├── docs               <- Documentación del proyecto
│
├── models             <- Modelos entrenados y serializados
│   ├── feature_engineering_pipeline.pkl  <- Pipeline de ingeniería de características
│   └── sales_pipeline.pkl                <- Pipeline de predicción de ventas
│
├── notebooks          <- Jupyter notebooks del flujo de trabajo
│   ├── 01_Data_Exploration.ipynb      <- EDA de ventas: análisis de datos, visualizaciones
│   ├── 02_feature_exploration.ipynb   <- Exploración y selección de características
│   ├── 03_feature_creation.ipynb      <- Creación de features con pipelines de sklearn
│   ├── 04_model_tuning_training.ipynb <- Ajuste de hiperparámetros y entrenamiento
│   ├── 05_inference_calculation.ipynb <- Cálculo de inferencias/predicciones
│   └── operators.py                   <- Transformadores personalizados (Mapper, SimpleCategoricalImputer)
│
├── pyproject.toml     <- Archivo de configuración del proyecto con metadatos del paquete
│
├── references         <- Diccionarios de datos, manuales y materiales explicativos.
│
├── reports            <- Análisis generados como HTML, PDF, LaTeX, etc.
│   └── figures        <- Gráficos y figuras generadas para reportes
│
├── requirements.txt   <- Dependencias de conda para reproducir el entorno
│
├── tests              <- Pruebas unitarias
│   └── test_data.py   <- Tests de validación de datos
│
└── product_development   <- Código fuente del proyecto
    │
    ├── __init__.py     <- Hace que product_development sea un módulo de Python
    ├── config.py       <- Configuración de rutas del proyecto (DATA_DIR, MODELS_DIR, etc.)
    ├── dataset.py      <- Scripts para descargar o generar datos
    ├── features.py     <- Código para generación de características
    ├── plots.py        <- Código para crear visualizaciones
    │
    └── modeling                
        ├── __init__.py 
        ├── predict.py  <- Código para ejecutar inferencia con modelos entrenados          
        └── train.py    <- Código para entrenar modelos
```

## Flujo de Trabajo

1. **Exploración de Datos** (`01_Data_Exploration.ipynb`): Análisis exploratorio del dataset de ventas con visualizaciones y estadísticas descriptivas.

2. **Exploración de Características** (`02_feature_exploration.ipynb`): Análisis de variables candidatas para el modelo.

3. **Creación de Características** (`03_feature_creation.ipynb`): 
   - Features de lag (1, 7, 14, 28 días)
   - Medias móviles (7 y 28 días)
   - Codificación de frecuencia para items
   - One-hot encoding para tiendas
   - Tratamiento de outliers y transformaciones logarítmicas
   - Escalado de características

4. **Entrenamiento del Modelo** (`04_model_tuning_training.ipynb`): Ajuste de hiperparámetros y entrenamiento del modelo de predicción.

5. **Inferencia** (`05_inference_calculation.ipynb`): Generación de predicciones usando los modelos entrenados.

## Instalación

```bash
# Crear entorno conda
conda create --name product_dev --file requirements.txt

# Activar entorno
conda activate product_dev
```

## Tecnologías Utilizadas

- **Python 3.11**
- **Pandas & NumPy**: Manipulación de datos
- **Scikit-learn**: Pipelines y modelado
- **Feature-engine**: Transformadores de características
- **Matplotlib & Seaborn**: Visualizaciones
- **Statsmodels**: Análisis estadístico
- **Joblib**: Serialización de modelos

--------

