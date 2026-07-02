from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.credit_risk.config import AppConfig, load_config
from src.credit_risk.modeling.metrics import calculate_classification_metrics
from src.credit_risk.utils.logging import configure_logging

LOGGER = logging.getLogger(__name__)


class ModelTrainingError(ValueError):
    """Raised when model training cannot be completed."""


def load_best_params(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as file_obj:
        params = json.load(file_obj)

    if not isinstance(params, dict):
        raise ModelTrainingError(f"Invalid best params format: {path}")

    return params


def get_random_forest_params(config: AppConfig) -> dict[str, Any]:
    optimized_params = load_best_params(config.model.best_params_path)

    if optimized_params:
        LOGGER.info("Using optimized Random Forest params from %s", config.model.best_params_path)
        return optimized_params

    LOGGER.info("Using configured Random Forest params from params.yaml")
    return {
        "n_estimators": config.model.random_forest.n_estimators,
        "max_depth": config.model.random_forest.max_depth,
        "min_samples_split": config.model.random_forest.min_samples_split,
        "min_samples_leaf": config.model.random_forest.min_samples_leaf,
        "max_features": config.model.random_forest.max_features,
        "class_weight": config.model.random_forest.class_weight,
        "random_state": config.model.random_state,
        "n_jobs": -1,
    }


def build_model(config: AppConfig) -> Pipeline:
    algorithm = config.model.algorithm

    if algorithm == "logistic_regression":
        classifier = LogisticRegression(
            max_iter=config.model.logistic_regression.max_iter,
            class_weight=config.model.logistic_regression.class_weight,
            random_state=config.model.random_state,
        )
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("classifier", classifier),
            ]
        )

    if algorithm == "random_forest":
        classifier = RandomForestClassifier(**get_random_forest_params(config))
        return Pipeline(steps=[("classifier", classifier)])

    raise ModelTrainingError(f"Unsupported algorithm: {algorithm}")


def split_features_target(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    if df.empty:
        raise ModelTrainingError("Training dataset is empty")

    if target_column not in df.columns:
        raise ModelTrainingError(f"Target column missing: {target_column}")

    x = df.drop(columns=[target_column])
    y = df[target_column].astype(int)

    if x.empty:
        raise ModelTrainingError("No feature columns available for training")

    return x, y


def predict_probability(model: Pipeline, features: pd.DataFrame) -> Any:
    classifier = model.named_steps["classifier"]
    if hasattr(classifier, "predict_proba"):
        return model.predict_proba(features)[:, 1]
    return None


def train_model(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    config: AppConfig,
) -> tuple[Pipeline, dict[str, Any]]:
    target_column = config.dataset.target_column

    x_train, y_train = split_features_target(train_df, target_column)
    x_validation, y_validation = split_features_target(validation_df, target_column)

    model = build_model(config)

    LOGGER.info("Training model algorithm=%s train_shape=%s", config.model.algorithm, x_train.shape)
    model.fit(x_train, y_train)

    predictions = model.predict(x_validation)
    probabilities = predict_probability(model, x_validation)

    metrics = calculate_classification_metrics(
        y_true=y_validation.to_numpy(),
        y_pred=predictions,
        y_probability=probabilities,
    )

    return model, metrics


def save_pickle(model: Pipeline, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as file_obj:
        pickle.dump(model, file_obj)


def save_metrics(metrics: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")


def log_to_mlflow(model: Pipeline, metrics: dict[str, Any], config: AppConfig) -> None:
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment_name)

    with mlflow.start_run(run_name=f"train-{config.model.algorithm}"):
        mlflow.log_params(
            {
                "algorithm": config.model.algorithm,
                "random_state": config.model.random_state,
            }
        )

        if config.model.algorithm == "random_forest":
            mlflow.log_params(
                {f"model_{key}": value for key, value in get_random_forest_params(config).items()}
            )

        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, float(metric_value))

        mlflow.sklearn.log_model(model, artifact_path="model")


def main() -> None:
    configure_logging()
    config = load_config()

    train_path = config.dataset.train_path
    validation_path = config.dataset.validation_path

    if not train_path.exists():
        raise FileNotFoundError(f"Training dataset not found: {train_path}")

    if not validation_path.exists():
        raise FileNotFoundError(f"Validation dataset not found: {validation_path}")

    train_df = pd.read_csv(train_path)
    validation_df = pd.read_csv(validation_path)

    model, metrics = train_model(train_df, validation_df, config)

    save_pickle(model, config.model.output_path)
    save_metrics(metrics, config.model.metrics_path)
    log_to_mlflow(model, metrics, config)

    LOGGER.info("Model saved to %s", config.model.output_path)
    LOGGER.info("Metrics saved to %s", config.model.metrics_path)
    LOGGER.info("Metrics=%s", metrics)


if __name__ == "__main__":
    main()
