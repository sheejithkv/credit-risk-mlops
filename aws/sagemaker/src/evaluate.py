from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

TARGET_COLUMN = "target_bad"


def extract_model(model_artifact: Path, extract_dir: Path) -> Path:
    extract_dir.mkdir(parents=True, exist_ok=True)
    if model_artifact.is_dir():
        tar_path = model_artifact / "model.tar.gz"
    else:
        tar_path = model_artifact
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(extract_dir)
    model_path = extract_dir / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"model.joblib not found after extraction: {model_path}")
    return model_path


def evaluate(model_artifact: Path, test_path: Path, output_dir: Path) -> None:
    model = joblib.load(extract_model(model_artifact, Path("/tmp/credit-risk-model")))
    test_df = pd.read_csv(test_path)

    x_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN].astype(int)
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]

    report = {
        "classification_metrics": {
            "accuracy": {"value": float(accuracy_score(y_test, predictions))},
            "precision": {"value": float(precision_score(y_test, predictions, zero_division=0))},
            "recall": {"value": float(recall_score(y_test, predictions, zero_division=0))},
            "f1": {"value": float(f1_score(y_test, predictions, zero_division=0))},
            "roc_auc": {"value": float(roc_auc_score(y_test, probabilities))},
        }
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/opt/ml/processing/model/model.tar.gz")
    parser.add_argument("--test", default="/opt/ml/processing/test/test.csv")
    parser.add_argument("--output", default="/opt/ml/processing/evaluation")
    args = parser.parse_args()
    evaluate(Path(args.model), Path(args.test), Path(args.output))


if __name__ == "__main__":
    main()
