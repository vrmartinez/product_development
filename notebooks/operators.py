from sklearn.base import BaseEstimator, TransformerMixin

class Mapper(BaseEstimator, TransformerMixin):
    def __init__(self, mappings, variables):
        self.mappings = mappings
        self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].map(self.mappings)
        return X


class SimpleCategoricalImputer(BaseEstimator, TransformerMixin):
    """
    Imputador categórico sencillo:
    - Rellena NaNs con un string (por defecto 'Missing').
    """
    def __init__(self, variables, fill_value="Missing"):
        self.variables = variables
        self.fill_value = fill_value

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for var in self.variables:
            if var in X.columns:
                X[var] = X[var].fillna(self.fill_value).astype(str)
        return X
