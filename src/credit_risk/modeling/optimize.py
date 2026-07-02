from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import mlflow
import optuna
import pandas as pd
from optuna.trial import Trial
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.credit_risk.config import AppConfig, load_config
from src.credit_risk.modeling.metrics import calculate_classification_metrics
from src.credit_risk.modeling.train import predict_probability, split_features_target
from src.credit_risk.utils.logging import configure_logging

LOGGER = logging.getLogger(__name__)


class OptimizationError(ValueError):
    """Raised when hyperparameter optimization cannot be completed."""


def suggest_random_forest_params(trial: Trial, config: AppConfig) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
        "max_depth": trial.suggest_int("max_depth", 4, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        "class_weight": config.model.random_forest.class_weight,
        "random_state": config.model.random_state,
        "n_jobs": -1,
    }


def build_trial_model(params: dict[str, Any]) -> Pipeline:
    classifier = RandomForestClassifier(**params)
    return Pipeline(steps=[("classifier", classifier)])


def objective_factory(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    config: AppConfig,
):
    target_column = config.dataset.target_column
    x_train, y_train = split_features_target(train_df, target_column)
    x_validation, y_validation = split_features_target(validation_df, target_column)

    def objective(trial: Trial) -> float:
        params = suggest_random_forest_params(trial, config)
        model = build_trial_model(params)

        with mlflow.start_run(run_name=f"optuna-trial-{trial.number}", nested=True):
            mlflow.log_params(params)

            model.fit(x_train, y_train)

            predictions = model.predict(x_validation)
            probabilities = predict_probability(model, x_validation)

            metrics = calculate_classification_metrics(
                y_true=y_validation.to_numpy(),
                y_pred=predictions,
                y_probability=probabilities,
            )

            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, float(metric_value))

            score = metrics.get(config.optuna.metric_name)
            if score is None:
                raise OptimizationError(f"Metric not available: {config.optuna.metric_name}")

            return float(score)

    return objective


def save_best_params(best_params: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(best_params, indent=2, sort_keys=True), encoding="utf-8")


def optimize_hyperparameters(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    config: AppConfig,
) -> dict[str, Any]:
    if config.model.algorithm != "random_forest":
        raise OptimizationError("Optuna optimization currently supports random_forest only")

    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment_name)

    sampler = optuna.samplers.TPESampler(seed=config.optuna.sampler_seed)

    study = optuna.create_study(
        study_name=config.optuna.study_name,
        direction=config.optuna.direction,
        sampler=sampler,
        storage=config.optuna.storage,
        load_if_exists=True,
    )

    objective = objective_factory(train_df, validation_df, config)

    with mlflow.start_run(run_name="optuna-random-forest-optimization"):
        mlflow.log_params(
            {
                "study_name": config.optuna.study_name,
                "n_trials": config.optuna.n_trials,
                "direction": config.optuna.direction,
                "metric_name": config.optuna.metric_name,
            }
        )

        study.optimize(
            objective,
            n_trials=config.optuna.n_trials,
            timeout=config.optuna.timeout,
            show_progress_bar=False,
        )

        best_params = dict(study.best_params)
        best_params["class_weight"] = config.model.random_forest.class_weight
        best_params["random_state"] = config.model.random_state
        best_params["n_jobs"] = -1

        mlflow.log_params({f"best_{key}": value for key, value in best_params.items()})
        mlflow.log_metric(f"best_{config.optuna.metric_name}", float(study.best_value))

    return best_params


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

    LOGGER.info(
        "Starting Optuna optimization with n_trials=%s metric=%s",
        config.optuna.n_trials,
        config.optuna.metric_name,
    )

    best_params = optimize_hyperparameters(train_df, validation_df, config)
    save_best_params(best_params, config.model.best_params_path)

    LOGGER.info("Best params written to %s", config.model.best_params_path)
    LOGGER.info("Best params=%s", best_params)


if __name__ == "__main__":
    main()
