"""Small, framework-independent helpers for model monitoring."""

import pandas as pd


def infer_task_type(target: pd.Series) -> str:
    """Infer classification for labels and low-cardinality numeric categories."""
    values = target.dropna()
    if values.empty:
        return "Classification"
    categorical_limit = max(20, int(len(values) * 0.05))
    is_numeric = pd.api.types.is_numeric_dtype(values)
    return "Classification" if not is_numeric or values.nunique() <= categorical_limit else "Regression"
