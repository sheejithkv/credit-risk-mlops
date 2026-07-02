import json

import pandas as pd
import pytest

from src.credit_risk.config import load_config
from src.credit_risk.modeling.train import (
    ModelTrainingError,
    build_model,
    get_random_forest_params,
    load_best_params,
    split_features_target,
    train_model,
)


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


def test_load_best_params_returns_none_when_missing(tmp_path) -> None:
    missing_path = tmp_path / "missing.json"

    assert load_best_params(missing_path) is None


def test_get_random_forest_params_uses_best_params(tmp_path, monkeypatch) -> None:
    config = load_config("params.yaml")
    best_params_path = tmp_path / "best_params.json"
    best_params = {
        "n_estimators": 50,
        "max_depth": 5,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
    }
    best_params_path.write_text(json.dumps(best_params), encoding="utf-8")

    monkeypatch.setattr(config.model, "best_params_path", best_params_path)

    params = get_random_forest_params(config)

    assert params == best_params
