from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.credit_risk.config import AppConfig, load_config
from src.credit_risk.utils.logging import configure_logging

LOGGER = logging.getLogger(__name__)


class DatasetSplitError(ValueError):
    """Raised when dataset splitting cannot be performed safely."""


def split_dataframe(df: pd.DataFrame, config: AppConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target_column = config.dataset.target_column

    if df.empty:
        LOGGER.warning("Input dataframe is empty. Writing empty train/validation/test files.")
        return df.copy(), df.copy(), df.copy()

    if target_column not in df.columns:
        raise DatasetSplitError(f"Target column missing: {target_column}")

    if config.split.stratify:
        class_counts = df[target_column].value_counts()
        if class_counts.min() < 2:
            raise DatasetSplitError("Each target class must have at least 2 rows for stratified split")
        stratify_values = df[target_column]
    else:
        stratify_values = None

    train_df, temp_df = train_test_split(
        df,
        train_size=config.split.train_size,
        random_state=config.split.random_state,
        stratify=stratify_values,
    )

    relative_validation_size = config.split.validation_size / (
        config.split.validation_size + config.split.test_size
    )

    temp_stratify = temp_df[target_column] if config.split.stratify else None

    validation_df, test_df = train_test_split(
        temp_df,
        train_size=relative_validation_size,
        random_state=config.split.random_state,
        stratify=temp_stratify,
    )

    return (
        train_df.reset_index(drop=True),
        validation_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def write_dataset(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def main() -> None:
    configure_logging()
    config = load_config()

    input_path = config.dataset.final_path

    if not input_path.exists():
        raise FileNotFoundError(f"Preprocessed dataset not found: {input_path}")

    LOGGER.info("Loading preprocessed dataset from %s", input_path)
    df = pd.read_csv(input_path)

    train_df, validation_df, test_df = split_dataframe(df, config)

    write_dataset(train_df, config.dataset.train_path)
    write_dataset(validation_df, config.dataset.validation_path)
    write_dataset(test_df, config.dataset.test_path)

    LOGGER.info(
        "Split complete: train=%s validation=%s test=%s",
        train_df.shape,
        validation_df.shape,
        test_df.shape,
    )


if __name__ == "__main__":
    main()
