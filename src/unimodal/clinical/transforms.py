from src.unimodal.rna.transforms import Scale
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

def to_int_list(x):
    # Если это pandas Series → .values
    if hasattr(x, "values"):
        x = x.values

    # Если это numpy array → .tolist()
    if hasattr(x, "tolist"):
        x = x.tolist()

    # Теперь x должен быть list/tuple/iterable
    return [int(v) for v in x]

class StandardScalerWithoutCategorical:
    """
    Препроцессор, который:
    - нормализует числовые колонки StandardScaler'ом
    - категориальные оставляет "как есть"
    - принимает индексы числовых и категориальных колонок
    """

    def __init__(self, categorical_cols, numerical_cols):
        self.categorical_cols = to_int_list(categorical_cols)
        self.numerical_cols = to_int_list(numerical_cols)
        print("self.categorical_cols:", self.categorical_cols)
        print("self.numerical_cols:", self.numerical_cols)

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('cat', 'passthrough', self.categorical_cols),
                ('num', StandardScaler(), self.numerical_cols)
                
            ],
            remainder='drop'
        )

    def fit(self, X, y=None):
        """Запоминает статистики StandardScaler."""
        self.preprocessor.fit(X)
        return self

    def transform(self, X):
        """Преобразует данные (числовые нормализует, категориальные нет)."""
        return self.preprocessor.transform(X)

    def fit_transform(self, X):
        """fit + transform."""
        return self.preprocessor.fit_transform(X)

def base_scaling(standart_scaler):
    return Scale(standart_scaler)