import pandas as pd
import pytest

from src.credit_risk.config import load_config
from src.credit_risk.data.preprocess import PreprocessingError, preprocess_dataframe


def test_preprocess_encodes_target_and_categoricals() -> None:
    config = load_config("params.yaml")
    df = pd.DataFrame(
        {
            "age": [30, 40],
            "job": ["skilled", "unskilled"],
            "credit_risk": ["good", "bad"],
        }
    )

    result = preprocess_dataframe(df, config)

    assert "credit_risk" in result.columns
    assert result["credit_risk"].tolist() == [0, 1]
    assert "job_skilled" in result.columns
    assert "job_unskilled" in result.columns


def test_preprocess_rejects_missing_target() -> None:
    config = load_config("params.yaml")
    df = pd.DataFrame({"age": [30], "job": ["skilled"]})

    with pytest.raises(PreprocessingError, match="Target column missing"):
        preprocess_dataframe(df, config)
