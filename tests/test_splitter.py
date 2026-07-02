import pandas as pd
import pytest

from src.credit_risk.config import load_config
from src.credit_risk.data.splitter import DatasetSplitError, split_dataframe


def make_split_dataframe(rows: int = 100) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": list(range(rows)),
            "amount": list(range(1000, 1000 + rows)),
            "target_bad": [0, 1] * (rows // 2),
        }
    )


def test_split_dataframe_ratios() -> None:
    config = load_config("params.yaml")
    df = make_split_dataframe(100)

    train_df, validation_df, test_df = split_dataframe(df, config)

    assert len(train_df) == 70
    assert len(validation_df) == 15
    assert len(test_df) == 15


def test_split_dataframe_is_reproducible() -> None:
    config = load_config("params.yaml")
    df = make_split_dataframe(100)

    first_train, first_validation, first_test = split_dataframe(df, config)
    second_train, second_validation, second_test = split_dataframe(df, config)

    pd.testing.assert_frame_equal(first_train, second_train)
    pd.testing.assert_frame_equal(first_validation, second_validation)
    pd.testing.assert_frame_equal(first_test, second_test)


def test_split_dataframe_rejects_missing_target() -> None:
    config = load_config("params.yaml")
    df = pd.DataFrame({"age": [30, 40]})

    with pytest.raises(DatasetSplitError, match="Target column missing"):
        split_dataframe(df, config)
