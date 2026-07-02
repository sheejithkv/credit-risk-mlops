from __future__ import annotations

import logging

import pandas as pd

from src.credit_risk.config import AppConfig, load_config
from src.credit_risk.utils.logging import configure_logging

LOGGER = logging.getLogger(__name__)


class PreprocessingError(ValueError):
    """Raised when preprocessing cannot be completed safely."""


def preprocess_dataframe(df: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
    target_column = config.dataset.target_column

    if df.empty:
        LOGGER.warning("Input dataframe is empty. Returning empty dataframe.")
        return df.copy()

    if target_column not in df.columns:
        raise PreprocessingError(f"Target column missing: {target_column}")

    processed_df = df.copy()

    if config.preprocessing.drop_duplicates:
        before = len(processed_df)
        processed_df = processed_df.drop_duplicates()
        LOGGER.info("Dropped %s duplicate rows", before - len(processed_df))

    if config.preprocessing.encode_target:
        mapping = config.preprocessing.target_mapping
        processed_df[target_column] = (
            processed_df[target_column].astype(str).str.lower().map(mapping)
        )

        if processed_df[target_column].isna().any():
            raise PreprocessingError("Target encoding produced null values")

        processed_df[target_column] = processed_df[target_column].astype(int)

    categorical_columns = [
        column
        for column in processed_df.columns
        if column != target_column and processed_df[column].dtype == "object"
    ]

    if categorical_columns:
        processed_df = pd.get_dummies(
            processed_df,
            columns=categorical_columns,
            drop_first=False,
            dtype=int,
        )

    return processed_df


def main() -> None:
    configure_logging()
    config = load_config()

    input_path = config.dataset.processed_path
    output_path = config.dataset.final_path

    if not input_path.exists():
        raise FileNotFoundError(f"Validated dataset not found: {input_path}")

    LOGGER.info("Loading validated dataset from %s", input_path)
    df = pd.read_csv(input_path)

    LOGGER.info("Preprocessing dataset with shape=%s", df.shape)
    processed_df = preprocess_dataframe(df, config)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(output_path, index=False)

    LOGGER.info("Preprocessed dataset written to %s with shape=%s", output_path, processed_df.shape)


if __name__ == "__main__":
    main()
