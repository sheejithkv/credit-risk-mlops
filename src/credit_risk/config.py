from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class DatasetConfig(BaseModel):
    raw_path: Path
    processed_path: Path
    final_path: Path
    target_column: str
    allow_missing_raw: bool = False


class ValidationConfig(BaseModel):
    min_rows: int = Field(gt=0)
    expected_columns: int = Field(gt=0)
    allowed_targets: list[str]


class PreprocessingConfig(BaseModel):
    drop_duplicates: bool = True
    encode_target: bool = True
    target_mapping: dict[str, int]


class ModelConfig(BaseModel):
    random_state: int = 42
    test_size: float = Field(gt=0.0, lt=1.0)


class MlflowConfig(BaseModel):
    tracking_uri: str
    experiment_name: str
    registered_model_name: str


class AppConfig(BaseModel):
    dataset: DatasetConfig
    validation: ValidationConfig
    preprocessing: PreprocessingConfig
    model: ModelConfig
    mlflow: MlflowConfig


def load_config(path: str | Path = "params.yaml") -> AppConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file_obj:
        raw_config: dict[str, Any] = yaml.safe_load(file_obj) or {}

    return AppConfig.model_validate(raw_config)
