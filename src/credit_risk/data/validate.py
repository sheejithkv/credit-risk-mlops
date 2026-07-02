from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.credit_risk.config import AppConfig, load_config
from src.credit_risk.utils.logging import configure_logging

LOGGER = logging.getLogger(__name__)


class DatasetValidationError(ValueError):
    """Raised when the input dataset violates validation rules."""


def validate_dataframe(df: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
    if df.empty:
        raise DatasetValidationError("Dataset is empty")

    if len(df) < config.validation.min_rows:
        raise DatasetValidationError(
            f"Dataset has {len(df)} rows; expected at least {config.validation.min_rows}"
        )

    if df.shape[1] != config.validation.expected_columns:
        raise DatasetValidationError(
            f"Dataset has {df.shape[1]} columns; expected {config.validation.expected_columns}"
        )

    target_column = config.dataset.target_column
    if target_column not in df.columns:
        raise DatasetValidationError(f"Target column missing: {target_column}")

    missing_count = int(df.isna().sum().sum())
    if missing_count > 0:
        raise DatasetValidationError(f"Dataset contains {missing_count} missing values")

    observed_targets = set(df[target_column].astype(str).str.lower().unique())
    allowed_targets = set(config.validation.allowed_targets)
    invalid_targets = observed_targets - allowed_targets
    if invalid_targets:
        raise DatasetValidationError(f"Invalid target values: {sorted(invalid_targets)}")

    return df


def write_bootstrap_dataset(output_path: Path, config: AppConfig) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [f"feature_{idx}" for idx in range(1, config.validation.expected_columns)]
    columns.append(config.dataset.target_column)
    pd.DataFrame(columns=columns).to_csv(output_path, index=False)


def main() -> None:
    configure_logging()
    config = load_config()

    raw_path = config.dataset.raw_path
    processed_path = config.dataset.processed_path

    if not raw_path.exists():
        if config.dataset.allow_missing_raw:
            LOGGER.warning("Raw dataset not found at %s. Writing bootstrap processed dataset.", raw_path)
            write_bootstrap_dataset(processed_path, config)
            return
        raise FileNotFoundError(f"Raw dataset not found: {raw_path}")

    LOGGER.info("Loading raw dataset from %s", raw_path)
    df = pd.read_csv(raw_path)

    LOGGER.info("Validating dataset with shape=%s", df.shape)
    validated_df = validate_dataframe(df, config)

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    validated_df.to_csv(processed_path, index=False)
    LOGGER.info("Validated dataset written to %s", processed_path)


if __name__ == "__main__":
    main()
