import pandas as pd
import pytest

from src.credit_risk.config import load_config
from src.credit_risk.data.validate import DatasetValidationError, validate_dataframe


def make_valid_dataframe(rows: int = 100) -> pd.DataFrame:
    data = {f"feature_{idx}": [idx] * rows for idx in range(1, 21)}
    data["credit_risk"] = ["good"] * rows
    return pd.DataFrame(data)


def test_validate_dataframe_success() -> None:
    config = load_config("params.yaml")
    df = make_valid_dataframe()

    result = validate_dataframe(df, config)

    assert result.shape == (100, 21)


def test_validate_dataframe_rejects_invalid_target() -> None:
    config = load_config("params.yaml")
    df = make_valid_dataframe()
    df.loc[0, "credit_risk"] = "unknown"

    with pytest.raises(DatasetValidationError, match="Invalid target values"):
        validate_dataframe(df, config)
