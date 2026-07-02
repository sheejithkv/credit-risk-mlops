import pandas as pd
import pytest

from src.credit_risk.config import load_config
from src.credit_risk.data.preprocess import PreprocessingError, preprocess_dataframe


def test_preprocess_encodes_categoricals_and_keeps_numeric_target() -> None:
    config = load_config("params.yaml")
    df = pd.DataFrame(
        {
            "application_id": [1, 2],
            "timestamp": ["2025-01-01 00:00:00", "2025-01-02 00:00:00"],
            "age": [30, 40],
            "job": ["skilled", "unskilled"],
            "target_bad": [0, 1],
        }
    )

    result = preprocess_dataframe(df, config)

    assert "application_id" not in result.columns
    assert "timestamp" not in result.columns
    assert "target_bad" in result.columns
    assert result["target_bad"].tolist() == [0, 1]
    assert "job_skilled" in result.columns
    assert "job_unskilled" in result.columns


def test_preprocess_rejects_missing_target() -> None:
    config = load_config("params.yaml")
    df = pd.DataFrame({"age": [30], "job": ["skilled"]})

    with pytest.raises(PreprocessingError, match="Target column missing"):
        preprocess_dataframe(df, config)
