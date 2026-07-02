import pandas as pd
from optuna.trial import FixedTrial

from src.credit_risk.config import load_config
from src.credit_risk.modeling.optimize import build_trial_model, suggest_random_forest_params


def test_suggest_random_forest_params() -> None:
    config = load_config("params.yaml")
    trial = FixedTrial(
        {
            "n_estimators": 100,
            "max_depth": 8,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
        }
    )

    params = suggest_random_forest_params(trial, config)

    assert params["n_estimators"] == 100
    assert params["max_depth"] == 8
    assert params["min_samples_split"] == 2
    assert params["min_samples_leaf"] == 1
    assert params["max_features"] == "sqrt"
    assert params["class_weight"] == "balanced"


def test_build_trial_model() -> None:
    params = {
        "n_estimators": 10,
        "max_depth": 4,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
    }

    model = build_trial_model(params)

    assert hasattr(model, "fit")
    assert hasattr(model, "predict")
