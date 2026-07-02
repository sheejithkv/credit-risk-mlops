from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class DatasetConfig(BaseModel):
    raw_path: Path
    processed_path: Path
    final_path: Path
    train_path: Path
    validation_path: Path
    test_path: Path
    schema_path: Path
    target_column: str
    allow_missing_raw: bool = False


class ValidationConfig(BaseModel):
    min_rows: int = Field(gt=0)
    expected_columns: int = Field(gt=0)
    allowed_targets: list[str]


class PreprocessingConfig(BaseModel):
    drop_duplicates: bool = True
    encode_target: bool = True
    drop_columns: list[str] = Field(default_factory=list)
    target_mapping: dict[str, int]

class SplitConfig(BaseModel):
    train_size: float = Field(gt=0.0, lt=1.0)
    validation_size: float = Field(gt=0.0, lt=1.0)
    test_size: float = Field(gt=0.0, lt=1.0)
    stratify: bool = True
    random_state: int = 42

    @model_validator(mode="after")
    def validate_split_sum(self) -> "SplitConfig":
        total = self.train_size + self.validation_size + self.test_size
        if round(total, 6) != 1.0:
            raise ValueError("train_size + validation_size + test_size must equal 1.0")
        return self


class LogisticRegressionConfig(BaseModel):
    max_iter: int = Field(gt=0)
    class_weight: str | None = None


class RandomForestConfig(BaseModel):
    n_estimators: int = Field(gt=0)
    max_depth: int | None = Field(default=None, gt=0)
    min_samples_split: int = Field(default=2, ge=2)
    min_samples_leaf: int = Field(default=1, ge=1)
    class_weight: str | None = None


class ModelConfig(BaseModel):
    output_path: Path
    metrics_path: Path
    algorithm: Literal["logistic_regression", "random_forest"]
    random_state: int = 42
    test_size: float = Field(gt=0.0, lt=1.0)
    logistic_regression: LogisticRegressionConfig
    random_forest: RandomForestConfig


class MlflowConfig(BaseModel):
    tracking_uri: str
    experiment_name: str
    registered_model_name: str


class AppConfig(BaseModel):
    dataset: DatasetConfig
    validation: ValidationConfig
    preprocessing: PreprocessingConfig
    split: SplitConfig
    model: ModelConfig
    mlflow: MlflowConfig


def load_config(path: str | Path = "params.yaml") -> AppConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file_obj:
        raw_config: dict[str, Any] = yaml.safe_load(file_obj) or {}

    return AppConfig.model_validate(raw_config)
