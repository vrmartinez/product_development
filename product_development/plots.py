"""
Módulo de visualización para el pipeline de predicción de ventas.

Este módulo proporciona funciones para generar visualizaciones
para el análisis de predicción de ventas.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from loguru import logger
import typer

from product_development.config import FIGURES_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

# Configurar estilo de gráficos
sns.set(style="whitegrid", context="notebook")


def plot_time_series(
    series: pd.Series,
    title: str = "",
    ylabel: str = "Ventas",
    output_path: Path = None
) -> None:
    """
    Grafica una serie temporal.

    Parámetros
    ----------
    series : pd.Series
        Datos de serie temporal con índice datetime.
    title : str
        Título del gráfico.
    ylabel : str
        Etiqueta del eje Y.
    output_path : Path, opcional
        Ruta para guardar el gráfico.
    """
    plt.figure(figsize=(12, 4))
    plt.plot(series.index, series.values)
    plt.title(title)
    plt.xlabel("Fecha")
    plt.ylabel(ylabel)
    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        logger.info(f"Gráfico guardado en {output_path}")

    plt.show()


def plot_predictions_vs_actual(
    y_true: pd.Series,
    y_pred: pd.Series,
    title: str = "Predicciones vs Valores Reales",
    output_path: Path = None
) -> None:
    """
    Grafica predicciones contra valores reales.

    Parámetros
    ----------
    y_true : pd.Series
        Valores reales.
    y_pred : pd.Series
        Valores predichos.
    title : str
        Título del gráfico.
    output_path : Path, opcional
        Ruta para guardar el gráfico.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, s=1)
    plt.plot([y_true.min(), y_true.max()],
             [y_true.min(), y_true.max()],
             'r--', lw=2, label='Predicción perfecta')
    plt.xlabel("Real")
    plt.ylabel("Predicho")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        logger.info(f"Gráfico guardado en {output_path}")

    plt.show()


def plot_model_comparison(
    results: dict,
    title: str = "Comparación de Modelos (RMSE)",
    output_path: Path = None
) -> None:
    """
    Grafica comparación del rendimiento de modelos.

    Parámetros
    ----------
    results : dict
        Diccionario mapeando nombres de modelos a valores RMSE.
    title : str
        Título del gráfico.
    output_path : Path, opcional
        Ruta para guardar el gráfico.
    """
    sorted_results = dict(sorted(results.items(), key=lambda x: x[1]))

    plt.figure(figsize=(12, 6))
    bars = plt.barh(list(sorted_results.keys()), list(sorted_results.values()))
    plt.xlabel("RMSE")
    plt.title(title)

    # Agregar etiquetas de valores
    for bar, value in zip(bars, sorted_results.values()):
        plt.text(value + 0.1, bar.get_y() + bar.get_height()/2,
                 f'{value:.2f}', va='center')

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        logger.info(f"Gráfico guardado en {output_path}")

    plt.show()


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
    output_path: Path = FIGURES_DIR / "predictions_plot.png",
):
    """
    Función principal para generar gráficos a partir de datos de predicción.

    Parámetros
    ----------
    input_path : Path
        Ruta al archivo CSV de predicciones.
    output_path : Path
        Ruta para guardar el gráfico generado.
    """
    logger.info("Generando gráfico a partir de los datos...")

    # Cargar predicciones
    data = pd.read_csv(input_path)

    if "sales" in data.columns and "sales_pred" in data.columns:
        plot_predictions_vs_actual(
            data["sales"],
            data["sales_pred"],
            output_path=output_path
        )

    logger.success("Generación de gráfico completada.")


if __name__ == "__main__":
    app()
