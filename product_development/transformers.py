"""
Módulo de transformadores personalizados para el pipeline de predicción de ventas.

Este módulo contiene transformadores personalizados compatibles con sklearn
utilizados en el pipeline de ingeniería de características.
"""
from sklearn.base import BaseEstimator, TransformerMixin


class Mapper(BaseEstimator, TransformerMixin):
    """
    Transformador que mapea valores en columnas especificadas usando un diccionario.

    Parámetros
    ----------
    mappings : dict
        Diccionario que contiene el mapeo de valores originales a nuevos valores.
    variables : list
        Lista de nombres de columnas a las que aplicar el mapeo.
    """

    def __init__(self, mappings: dict, variables: list):
        """
        Inicializa el transformador Mapper.

        Parámetros
        ----------
        mappings : dict
            Diccionario con mapeos de valores.
        variables : list
            Lista de variables a transformar.
        """
        self.mappings = mappings
        self.variables = variables

    def fit(self, X, y=None):  # pylint: disable=unused-argument
        """
        Método fit (no realiza operaciones para este transformador).

        Parámetros
        ----------
        X : pandas.DataFrame
            Características de entrada.
        y : array-like, opcional
            Variable objetivo (no utilizada).

        Retorna
        -------
        self : Mapper
            Retorna self.
        """
        return self

    def transform(self, X):
        """
        Aplica el mapeo a las columnas especificadas.

        Parámetros
        ----------
        X : pandas.DataFrame
            Características de entrada.

        Retorna
        -------
        pandas.DataFrame
            Características transformadas con valores mapeados.
        """
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].map(self.mappings)
        return X


class SimpleCategoricalImputer(BaseEstimator, TransformerMixin):
    """
    Imputador categórico simple que rellena valores NaN con un valor especificado.

    Parámetros
    ----------
    variables : list
        Lista de nombres de columnas a imputar.
    fill_value : str, default='Missing'
        Valor a usar para rellenar valores NaN.
    """

    def __init__(self, variables: list, fill_value: str = "Missing"):
        """
        Inicializa el SimpleCategoricalImputer.

        Parámetros
        ----------
        variables : list
            Lista de variables a imputar.
        fill_value : str, default='Missing'
            Valor para rellenar NaN.
        """
        self.variables = variables
        self.fill_value = fill_value

    def fit(self, X, y=None):  # pylint: disable=unused-argument
        """
        Método fit (no realiza operaciones para este transformador).

        Parámetros
        ----------
        X : pandas.DataFrame
            Características de entrada.
        y : array-like, opcional
            Variable objetivo (no utilizada).

        Retorna
        -------
        self : SimpleCategoricalImputer
            Retorna self.
        """
        return self

    def transform(self, X):
        """
        Rellena valores NaN en las columnas especificadas.

        Parámetros
        ----------
        X : pandas.DataFrame
            Características de entrada.

        Retorna
        -------
        pandas.DataFrame
            Características transformadas con valores imputados.
        """
        X = X.copy()
        for var in self.variables:
            if var in X.columns:
                X[var] = X[var].fillna(self.fill_value).astype(str)
        return X
