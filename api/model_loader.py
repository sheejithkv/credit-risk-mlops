from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from sklearn.pipeline import Pipeline


class ModelLoadError(RuntimeError):
    pass


def load_model(model_path: str | Path = "models/model.pkl") -> Pipeline:
    path = Path(model_path)

    if not path.exists():
        raise ModelLoadError(f"Model file not found: {path}")

    with path.open("rb") as file_obj:
        model: Any = pickle.load(file_obj)

    if not hasattr(model, "predict"):
        raise ModelLoadError(f"Invalid model artifact: {path}")

    return model
