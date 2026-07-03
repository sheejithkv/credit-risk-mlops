from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

TARGET_COLUMN = "target_bad"


def train(train_path: Path, validation_path: Path, model_dir: Path) -> None:
    train_df = pd.read_csv(train_path)
    validation_df = pd.read_csv(validation_path)

    x_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN].astype(int)
    x_val = validation_df.drop(columns=[TARGET_COLUMN])
    y_val = validation_df[TARGET_COLUMN].astype(int)

    model = RandomForestClassifier(
        n_estimators=350,
        max_depth=7,
        min_samples_split=3,
        min_samples_leaf=1,
        max_features="log2",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_val)
    probabilities = model.predict_proba(x_val)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_val, predictions)),
        "precision": float(precision_score(y_val, predictions, zero_division=0)),
        "recall": float(recall_score(y_val, predictions, zero_division=0)),
        "f1": float(f1_score(y_val, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_val, probabilities)),
    }

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "model.joblib")
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (model_dir / "feature_columns.json").write_text(json.dumps(list(x_train.columns), indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="/opt/ml/input/data/train/train.csv")
    parser.add_argument("--validation", default="/opt/ml/input/data/validation/validation.csv")
    parser.add_argument("--model-dir", default="/opt/ml/model")
    args = parser.parse_args()
    train(Path(args.train), Path(args.validation), Path(args.model_dir))


if __name__ == "__main__":
    main()
