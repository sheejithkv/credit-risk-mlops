import pandas as pd
import pytest

from src.credit_risk.config import load_config
from src.credit_risk.modeling.train import ModelTrainingError, build_model, split_features_target, train_model


def make_training_dataframe(rows: int = 80) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": list(range(rows)),
            "amount": list(range(1000, 1000 + rows)),
            "duration": list(range(10, 10 + rows)),
            "target_bad": [0, 1] * (rows // 2),
        }
    )


def test_build_model_returns_pipeline() -> None:
    config = load_config("params.yaml")

    model = build_model(config)

    assert hasattr(model, "fit")
    assert hasattr(model, "predict")


def test_split_features_target() -> None:
    config = load_config("params.yaml")
    df = make_training_dataframe()

    x, y = split_features_target(df, config.dataset.target_column)

    assert "target_bad" not in x.columns
    assert len(y) == len(df)


def test_train_model_returns_metrics() -> None:
    config = load_config("params.yaml")
    train_df = make_training_dataframe(80)
    validation_df = make_training_dataframe(20)

    model, metrics = train_model(train_df, validation_df, config)

    assert hasattr(model, "predict")
    assert "accuracy" in metrics
    assert "f1" in metrics


def test_split_features_target_rejects_empty_dataframe() -> None:
    config = load_config("params.yaml")
    df = pd.DataFrame(columns=["age", "target_bad"])

    with pytest.raises(ModelTrainingError, match="Training dataset is empty"):
        split_features_target(df, config.dataset.target_column)
