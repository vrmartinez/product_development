# Pipeline DVC para MLOps - Predicción de Ventas

Este documento describe cómo usar el pipeline DVC para automatizar el flujo de entrenamiento del modelo de predicción de ventas.

## Estructura del Pipeline

El pipeline DVC está compuesto por 4 etapas secuenciales:

```
prepare_data → feature_engineering → train_model → inference
```

### 1. prepare_data
- **Entrada**: `data/raw/train.csv`
- **Salida**: `data/processed/prepared_data.csv`
- **Descripción**: Carga los datos crudos y agrega características temporales (year, month, day_of_week_name)

### 2. feature_engineering
- **Entrada**: `data/processed/prepared_data.csv`
- **Salida**: `models/feature_engineering_pipeline.pkl`
- **Descripción**: Construye y ajusta el pipeline de ingeniería de características

### 3. train_model
- **Entrada**: `data/processed/prepared_data.csv`, `models/feature_engineering_pipeline.pkl`
- **Salida**: `models/sales_pipeline.pkl`, `reports/metrics.json`
- **Descripción**: Entrena y evalúa múltiples modelos, selecciona el mejor y guarda el pipeline completo

### 4. inference
- **Entrada**: `data/processed/prepared_data.csv`, `models/sales_pipeline.pkl`
- **Salida**: `data/processed/test_predictions.csv`
- **Descripción**: Genera predicciones sobre los datos de prueba

## Requisitos

1. Instalar DVC:
```bash
pip install dvc
```

2. O instalar todas las dependencias del proyecto:
```bash
pip install -e .
```

## Uso del Pipeline

### Inicializar DVC (solo la primera vez)
```bash
dvc init
```

### Ejecutar todo el pipeline
```bash
dvc repro
```

### Ejecutar una etapa específica
```bash
# Solo preparación de datos
dvc repro prepare_data

# Solo entrenamiento
dvc repro train_model
```

### Ver el estado del pipeline
```bash
dvc status
```

### Ver el grafo de dependencias
```bash
dvc dag
```

### Ver métricas
```bash
dvc metrics show
```

### Comparar métricas entre experimentos
```bash
dvc metrics diff
```

## Parámetros

Los parámetros del pipeline se configuran en `params.yaml`:

```yaml
data:
  train_test_split_ratio: 0.8  # Proporción de datos para entrenamiento
  random_state: 2025           # Semilla para reproducibilidad

training:
  mode: "fast"                 # "fast" o "full" (más modelos)
  use_mlflow: true             # Registrar experimentos en MLflow

mlflow:
  tracking_uri: "mlruns"
  experiment_name: "sales_prediction"
  model_name: "sales_prediction_model"
```

### Modificar parámetros

Para cambiar el modo de entrenamiento a "full" (evalúa más configuraciones de modelos):

```yaml
training:
  mode: "full"
```

Luego ejecutar:
```bash
dvc repro
```

## Integración con Git

### Agregar archivos al control de versiones
```bash
git add dvc.yaml dvc.lock params.yaml .dvc/
git commit -m "Agregar pipeline DVC"
```

### Versionado de datos y modelos
```bash
# Agregar datos al tracking de DVC
dvc add data/raw/train.csv

# Commit del archivo .dvc generado
git add data/raw/train.csv.dvc
git commit -m "Agregar datos de entrenamiento"
```

## Flujo de trabajo típico

1. **Modificar código o parámetros**
2. **Ejecutar pipeline**: `dvc repro`
3. **Revisar métricas**: `dvc metrics show`
4. **Commit de cambios**: `git add . && git commit -m "Descripción"`
5. **Push de datos/modelos**: `dvc push` (si configurado remote)

## Archivos generados

| Archivo | Descripción |
|---------|-------------|
| `dvc.yaml` | Definición del pipeline |
| `dvc.lock` | Estado actual del pipeline (generado automáticamente) |
| `params.yaml` | Parámetros configurables |
| `reports/metrics.json` | Métricas del modelo |

## Métricas de ejemplo

Después de ejecutar el pipeline, `reports/metrics.json` contendrá:

```json
{
  "best_model": "XGBoost",
  "rmse": 12.34,
  "mse": 152.28,
  "mae": 9.87,
  "r2": 0.95,
  "train_samples": 729600,
  "val_samples": 182400
}
```
