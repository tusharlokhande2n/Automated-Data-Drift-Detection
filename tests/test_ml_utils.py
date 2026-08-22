import pandas as pd

from ml_utils import infer_task_type


def test_text_target_is_classification():
    assert infer_task_type(pd.Series(["Existing Customer", "Attrited Customer"])) == "Classification"


def test_low_cardinality_numeric_target_is_classification():
    assert infer_task_type(pd.Series([0, 1, 0, 1, 1])) == "Classification"


def test_continuous_numeric_target_is_regression():
    assert infer_task_type(pd.Series(range(100))) == "Regression"


def test_empty_target_has_safe_default():
    assert infer_task_type(pd.Series([], dtype="float64")) == "Classification"
