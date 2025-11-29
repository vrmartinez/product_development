# 📊 Sales Prediction API Documentation

## Descripción General

La **Sales Prediction API** es una API REST desarrollada con Flask que proporciona predicciones de ventas basadas en un modelo de Machine Learning. La API permite realizar predicciones tanto individuales como por lote (batch), y proporciona información detallada sobre el modelo y sus métricas.

---

## 🚀 Inicio Rápido

### Requisitos Previos

- Python 3.8+
- Pipeline de modelo entrenado (`models/sales_pipeline.pkl`)
- Dependencias instaladas (ver `environment.yml`)

### Ejecución de la API

```bash
# Desde la raíz del proyecto
python scripts/run_api.py

# Con opciones personalizadas
python scripts/run_api.py --host 0.0.0.0 --port 5000 --debug
```

### Parámetros de Configuración

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `--host` | string | `0.0.0.0` | Host donde ejecutar la API |
| `--port` | int | `5000` | Puerto donde ejecutar la API |
| `--debug` | flag | `False` | Habilita el modo debug de Flask |

---

## 📡 Endpoints

### 1. Información de la API

**GET** `/`

Retorna información general sobre la API y sus endpoints disponibles.

#### Respuesta Exitosa (200 OK)

```json
{
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
    "timestamp": "2025-01-15T10:30:00.123456"
}
```

---

### 2. Health Check

**GET** `/health`

Verifica el estado de salud de la API y la disponibilidad del modelo.

#### Respuesta Exitosa (200 OK)

```json
{
    "status": "healthy",
    "timestamp": "2025-01-15T10:30:00.123456"
}
```

#### Respuesta de Error

```json
{
    "status": "unhealthy",
    "timestamp": "2025-01-15T10:30:00.123456"
}
```

---

### 3. Información del Modelo

**GET** `/model/info`

Retorna información detallada sobre el modelo de predicción, incluyendo características utilizadas, pasos del pipeline, hiperparámetros y métricas.

#### Respuesta Exitosa (200 OK)

```json
{
    "model_name": "sales_prediction_model",
    "model_alias": "champion",
    "features": ["store", "item", "year", "month", "day_of_week_name"],
    "pipeline_steps": [
        {"name": "preprocessor", "type": "ColumnTransformer"},
        {"name": "regressor", "type": "RandomForestRegressor"}
    ],
    "hyperparameters": {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 2
    },
    "metrics": {
        "rmse": 5.23,
        "mae": 3.45,
        "r2": 0.89,
        "mse": 27.35
    },
    "timestamp": "2025-01-15T10:30:00.123456"
}
```

---

### 4. Predicción Individual

**POST** `/predict`

Realiza una predicción de ventas para un único registro.

#### Request Body

| Campo | Tipo | Requerido | Descripción |
|-------|------|-----------|-------------|
| `store` | integer | ✅ | ID de la tienda (1-10) |
| `item` | integer | ✅ | ID del producto (1-50) |
| `date` | string | ✅ | Fecha en formato `YYYY-MM-DD` |

#### Ejemplo de Request

```json
{
    "store": 1,
    "item": 1,
    "date": "2018-01-01"
}
```

#### cURL

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"store": 1, "item": 1, "date": "2018-01-01"}'
```

#### Respuesta Exitosa (200 OK)

```json
{
    "predictions": [42.35],
    "model_metrics": {
        "rmse": 5.23,
        "mae": 3.45,
        "r2": 0.89,
        "mse": 27.35
    },
    "hyperparameters": {
        "n_estimators": 100,
        "max_depth": 10
    },
    "timestamp": "2025-01-15T10:30:00.123456",
    "model_info": {
        "name": "sales_prediction_model",
        "alias": "champion",
        "features_used": ["store", "item", "year", "month", "day_of_week_name"]
    },
    "input_data": {
        "store": 1,
        "item": 1,
        "date": "2018-01-01"
    },
    "prediction_count": 1
}
```

#### Errores Posibles

**400 Bad Request** - Datos inválidos o faltantes

```json
{
    "error": "Campos requeridos faltantes: ['date']",
    "timestamp": "2025-01-15T10:30:00.123456"
}
```

**500 Internal Server Error** - Error del servidor

```json
{
    "error": "Error procesando la predicción",
    "timestamp": "2025-01-15T10:30:00.123456"
}
```

---

### 5. Predicción por Lote (Batch)

**POST** `/predict/batch`

Realiza predicciones de ventas para múltiples registros en una sola llamada.

#### Request Body

| Campo | Tipo | Requerido | Descripción |
|-------|------|-----------|-------------|
| `data` | array | ✅ | Lista de objetos con `store`, `item` y `date` |

#### Ejemplo de Request

```json
{
    "data": [
        {"store": 1, "item": 1, "date": "2018-01-01"},
        {"store": 1, "item": 2, "date": "2018-01-01"},
        {"store": 2, "item": 1, "date": "2018-01-02"},
        {"store": 3, "item": 5, "date": "2018-01-03"}
    ]
}
```

#### cURL

```bash
curl -X POST http://localhost:5000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
        {"store": 1, "item": 1, "date": "2018-01-01"},
        {"store": 1, "item": 2, "date": "2018-01-01"},
        {"store": 2, "item": 1, "date": "2018-01-02"}
    ]
  }'
```

#### Respuesta Exitosa (200 OK)

```json
{
    "predictions": [42.35, 38.21, 45.67],
    "model_metrics": {
        "rmse": 5.23,
        "mae": 3.45,
        "r2": 0.89,
        "mse": 27.35
    },
    "hyperparameters": {
        "n_estimators": 100,
        "max_depth": 10
    },
    "timestamp": "2025-01-15T10:30:00.123456",
    "model_info": {
        "name": "sales_prediction_model",
        "alias": "champion",
        "features_used": ["store", "item", "year", "month", "day_of_week_name"]
    },
    "prediction_count": 3,
    "predictions_detail": [
        {
            "index": 0,
            "input": {"store": 1, "item": 1, "date": "2018-01-01"},
            "prediction": 42.35
        },
        {
            "index": 1,
            "input": {"store": 1, "item": 2, "date": "2018-01-01"},
            "prediction": 38.21
        },
        {
            "index": 2,
            "input": {"store": 2, "item": 1, "date": "2018-01-02"},
            "prediction": 45.67
        }
    ]
}
```

#### Errores Posibles

**400 Bad Request** - Lista vacía o campo `data` faltante

```json
{
    "error": "El campo 'data' es requerido con una lista de registros",
    "timestamp": "2025-01-15T10:30:00.123456"
}
```

**400 Bad Request** - Error en un registro específico

```json
{
    "error": "Error en registro 2: Campos requeridos faltantes: ['item']",
    "timestamp": "2025-01-15T10:30:00.123456"
}
```

---

## 📋 Características del Modelo

### Features de Entrada

El modelo utiliza las siguientes características para realizar predicciones:

| Feature | Tipo | Descripción |
|---------|------|-------------|
| `store` | Categórico | Identificador de la tienda |
| `item` | Categórico | Identificador del producto |
| `year` | Numérico | Año extraído de la fecha |
| `month` | Numérico | Mes extraído de la fecha |
| `day_of_week_name` | Categórico | Nombre del día de la semana |

> **Nota:** Las características `year`, `month` y `day_of_week_name` se extraen automáticamente de la fecha proporcionada.

### Métricas del Modelo

| Métrica | Descripción |
|---------|-------------|
| `rmse` | Root Mean Square Error - Error cuadrático medio |
| `mae` | Mean Absolute Error - Error absoluto medio |
| `r2` | Coeficiente de determinación R² |
| `mse` | Mean Square Error - Error cuadrático medio |

---

## 🔧 Ejemplos de Uso

### Python (requests)

```python
import requests
import json

BASE_URL = "http://localhost:5000"

# Predicción individual
response = requests.post(
    f"{BASE_URL}/predict",
    json={
        "store": 1,
        "item": 1,
        "date": "2018-01-01"
    }
)
print(response.json())

# Predicción batch
response = requests.post(
    f"{BASE_URL}/predict/batch",
    json={
        "data": [
            {"store": 1, "item": 1, "date": "2018-01-01"},
            {"store": 2, "item": 3, "date": "2018-01-02"},
        ]
    }
)
print(response.json())
```

### JavaScript (fetch)

```javascript
// Predicción individual
const response = await fetch('http://localhost:5000/predict', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify({
        store: 1,
        item: 1,
        date: '2018-01-01'
    })
});
const data = await response.json();
console.log(data);
```

### HTTPie

```bash
# Health check
http GET localhost:5000/health

# Predicción individual
http POST localhost:5000/predict store:=1 item:=1 date="2018-01-01"

# Info del modelo
http GET localhost:5000/model/info
```

---

## 🏗️ Arquitectura

```
┌─────────────────────────────────────────────────────────────────┐
│                        Flask Application                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────┐  ┌─────────────┐  ┌──────────────┐  ┌───────────┐ │
│  │    /    │  │   /health   │  │   /predict   │  │/predict/  │ │
│  │  (GET)  │  │    (GET)    │  │    (POST)    │  │  batch    │ │
│  │         │  │             │  │              │  │  (POST)   │ │
│  └─────────┘  └─────────────┘  └──────────────┘  └───────────┘ │
│       │             │                 │                 │       │
│       └─────────────┴─────────────────┴─────────────────┘       │
│                              │                                   │
│                    ┌─────────▼─────────┐                        │
│                    │  Pipeline (Lazy)  │                        │
│                    │     Singleton     │                        │
│                    └─────────┬─────────┘                        │
│                              │                                   │
├──────────────────────────────┼──────────────────────────────────┤
│                              │                                   │
│                    ┌─────────▼─────────┐                        │
│                    │ sales_pipeline.pkl│                        │
│                    │   (Modelo ML)     │                        │
│                    └───────────────────┘                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Configuración

### Variables de Configuración (config.py)

| Variable | Valor Default | Descripción |
|----------|---------------|-------------|
| `PIPELINE_FILE` | `models/sales_pipeline.pkl` | Ruta al pipeline entrenado |
| `MLFLOW_MODEL_NAME` | `sales_prediction_model` | Nombre del modelo en MLflow |
| `MLFLOW_CHAMPION_ALIAS` | `champion` | Alias del modelo campeón |
| `FEATURES` | `["store", "item", "year", "month", "day_of_week_name"]` | Features del modelo |

### Integración con MLflow

La API se integra automáticamente con MLflow para:

1. **Cargar métricas** del modelo campeón desde el Model Registry
2. **Obtener hiperparámetros** del modelo entrenado
3. **Tracking** de experimentos y versiones

---

## 📊 Códigos de Estado HTTP

| Código | Descripción |
|--------|-------------|
| `200` | OK - Solicitud exitosa |
| `400` | Bad Request - Datos de entrada inválidos |
| `500` | Internal Server Error - Error del servidor |

---

## 🔍 Logging

La API utiliza `loguru` para logging estructurado. Los logs incluyen:

- Carga del pipeline
- Predicciones realizadas (individuales y batch)
- Errores y excepciones
- Health checks

### Ejemplo de Logs

```
2025-01-15 10:30:00.123 | INFO     | Cargando pipeline de predicción...
2025-01-15 10:30:00.456 | INFO     | Pipeline cargado exitosamente
2025-01-15 10:30:01.789 | INFO     | Predicción individual realizada: 42.35
2025-01-15 10:30:02.123 | INFO     | Predicción batch realizada: 100 registros
```

---

## 🛡️ Manejo de Errores

La API implementa manejo robusto de errores:

1. **Validación de entrada**: Verifica campos requeridos antes de procesar
2. **Errores de parsing JSON**: Retorna mensaje descriptivo
3. **Errores del modelo**: Captura excepciones del pipeline
4. **Logging de errores**: Todos los errores se registran para debugging

---

## 📝 Notas Adicionales

### Rendimiento

- El pipeline se carga de manera **lazy** (singleton pattern)
- Una vez cargado, permanece en memoria para predicciones subsecuentes
- Las predicciones batch son más eficientes que múltiples predicciones individuales

### Buenas Prácticas

1. Use `/health` para verificar disponibilidad antes de enviar predicciones
2. Para múltiples predicciones, use `/predict/batch` en lugar de llamadas individuales
3. Verifique los códigos de estado HTTP de las respuestas
4. Maneje errores apropiadamente en el cliente

---

## 📞 Soporte

Para problemas o preguntas:

1. Revise los logs del servidor
2. Verifique que el pipeline existe en `models/sales_pipeline.pkl`
3. Asegúrese de que MLflow esté configurado correctamente
4. Verifique el formato de los datos de entrada

---

**Versión:** 1.0.0  
**Última actualización:** Noviembre 2025
