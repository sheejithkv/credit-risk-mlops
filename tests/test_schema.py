import pandas as pd
import pytest

from src.credit_risk.config import load_config
from src.credit_risk.data.schema import SchemaValidationError, infer_schema


def test_infer_schema_detects_columns() -> None:
    config = load_config("params.yaml")
    df = pd.DataFrame(
        {
            "age": [30, 40],
            "amount": [1000.0, 2000.0],
            "job": ["skilled", "unskilled"],
            "target_bad": [0, 1],
        }
    )

    schema = infer_schema(df, config)

    assert schema["target_column"] == "target_bad"
    assert schema["numerical_columns"] == ["age", "amount"]
    assert schema["categorical_columns"] == ["job"]


def test_infer_schema_rejects_missing_target() -> None:
    config = load_config("params.yaml")
    df = pd.DataFrame({"age": [30, 40]})

    with pytest.raises(SchemaValidationError, match="Target column missing"):
        infer_schema(df, config)
