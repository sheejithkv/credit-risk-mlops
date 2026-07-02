from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.credit_risk.config import AppConfig, load_config
from src.credit_risk.utils.logging import configure_logging

LOGGER = logging.getLogger(__name__)


class SchemaValidationError(ValueError):
    """Raised when feature schema validation fails."""


def infer_schema(df: pd.DataFrame, config: AppConfig) -> dict[str, Any]:
    target_column = config.dataset.target_column

    if df.empty:
        LOGGER.warning("Input dataframe is empty. Writing empty schema.")
        return {
            "target_column": target_column,
            "columns": list(df.columns),
            "numerical_columns": [],
            "categorical_columns": [],
            "row_count": 0,
            "column_count": len(df.columns),
        }

    if df.columns.duplicated().any():
        duplicates = df.columns[df.columns.duplicated()].tolist()
        raise SchemaValidationError(f"Duplicate columns found: {duplicates}")

    if target_column not in df.columns:
        raise SchemaValidationError(f"Target column missing: {target_column}")

    unsupported_columns = [
        column
        for column in df.columns
        if column != target_column
        and not (
            pd.api.types.is_numeric_dtype(df[column])
            or pd.api.types.is_bool_dtype(df[column])
            or pd.api.types.is_object_dtype(df[column])
            or pd.api.types.is_categorical_dtype(df[column])
        )
    ]

    if unsupported_columns:
        raise SchemaValidationError(f"Unsupported column dtypes: {unsupported_columns}")

    feature_columns = [column for column in df.columns if column != target_column]

    numerical_columns = [
        column
        for column in feature_columns
        if pd.api.types.is_numeric_dtype(df[column]) or pd.api.types.is_bool_dtype(df[column])
    ]

    categorical_columns = [
        column
        for column in feature_columns
        if pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_categorical_dtype(df[column])
    ]

    return {
        "target_column": target_column,
        "columns": list(df.columns),
        "feature_columns": feature_columns,
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
        "row_count": int(len(df)),
        "column_count": int(df.shape[1]),
    }


def write_schema(schema: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(schema, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    configure_logging()
    config = load_config()

    input_path = config.dataset.final_path
    output_path = config.dataset.schema_path

    if not input_path.exists():
        raise FileNotFoundError(f"Preprocessed dataset not found: {input_path}")

    LOGGER.info("Loading preprocessed dataset from %s", input_path)
    df = pd.read_csv(input_path)

    schema = infer_schema(df, config)
    write_schema(schema, output_path)

    LOGGER.info("Schema written to %s", output_path)


if __name__ == "__main__":
    main()
