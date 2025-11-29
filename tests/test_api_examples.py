"""
Ejemplos de uso de la API de predicción de ventas.

Este archivo contiene ejemplos de cómo consumir la API usando requests.
"""
import requests
import json

# URL base de la API (cambiar según configuración)
BASE_URL = "http://localhost:5000"


def test_home():
    """Prueba el endpoint principal."""
    response = requests.get(f"{BASE_URL}/")
    print("=== Home ===")
    print(json.dumps(response.json(), indent=2))
    print()


def test_health():
    """Prueba el endpoint de health check."""
    response = requests.get(f"{BASE_URL}/health")
    print("=== Health Check ===")
    print(json.dumps(response.json(), indent=2))
    print()


def test_model_info():
    """Prueba el endpoint de información del modelo."""
    response = requests.get(f"{BASE_URL}/model/info")
    print("=== Model Info ===")
    print(json.dumps(response.json(), indent=2))
    print()


def test_predict_single():
    """Prueba la predicción individual."""
    data = {
        "store": 1,
        "item": 1,
        "date": "2018-01-15"
    }
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json=data,
        headers={"Content-Type": "application/json"}
    )
    
    print("=== Predicción Individual ===")
    print(f"Input: {data}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()


def test_predict_batch():
    """Prueba la predicción por lote."""
    data = {
        "data": [
            {"store": 1, "item": 1, "date": "2018-01-15"},
            {"store": 1, "item": 2, "date": "2018-01-15"},
            {"store": 2, "item": 1, "date": "2018-01-16"},
            {"store": 3, "item": 5, "date": "2018-02-01"},
            {"store": 5, "item": 10, "date": "2018-03-15"},
        ]
    }
    
    response = requests.post(
        f"{BASE_URL}/predict/batch",
        json=data,
        headers={"Content-Type": "application/json"}
    )
    
    print("=== Predicción Batch ===")
    print(f"Input: {len(data['data'])} registros")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()


def run_all_tests():
    """Ejecuta todas las pruebas."""
    print("=" * 60)
    print("PRUEBAS DE LA API DE PREDICCIÓN DE VENTAS")
    print("=" * 60)
    print()
    
    try:
        test_home()
        test_health()
        test_model_info()
        test_predict_single()
        test_predict_batch()
        
        print("=" * 60)
        print("TODAS LAS PRUEBAS COMPLETADAS")
        print("=" * 60)
        
    except requests.exceptions.ConnectionError:
        print("ERROR: No se puede conectar a la API.")
        print(f"Asegúrese de que la API está corriendo en {BASE_URL}")
        print()
        print("Para iniciar la API, ejecute:")
        print("  python -m product_development.run_api")


if __name__ == "__main__":
    run_all_tests()
