from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from api.schemas import CreditRiskRequest


class InferencePreprocessingError(RuntimeError):
    pass


def load_schema(schema_path: str | Path = "data/processed/schema.json") -> dict[str, Any]:
    path = Path(schema_path)

    if not path.exists():
        raise InferencePreprocessingError(f"Schema file not found: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def prepare_features(
    request: CreditRiskRequest,
    schema_path: str | Path = "data/processed/schema.json",
) -> pd.DataFrame:
    schema = load_schema(schema_path)

    model_columns = [
        column
        for column in schema["columns"]
        if column != schema["target_column"]
    ]

    raw_df = pd.DataFrame([request.model_dump()])
    encoded_df = pd.get_dummies(raw_df, dtype="int8")

    for column in model_columns:
        if column not in encoded_df.columns:
            encoded_df[column] = 0

    return encoded_df[model_columns]
