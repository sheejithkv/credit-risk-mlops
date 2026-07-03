from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

TARGET_COLUMN = "target_bad"
DROP_COLUMNS = ["application_id", "timestamp"]


def preprocess(input_path: Path, output_dir: Path) -> None:
    df = pd.read_csv(input_path)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column missing: {TARGET_COLUMN}")

    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    categorical = [c for c in df.columns if c != TARGET_COLUMN and df[c].dtype == "object"]
    df = pd.get_dummies(df, columns=categorical, dtype="int8")

    train = df.sample(frac=0.70, random_state=42)
    remaining = df.drop(train.index)
    validation = remaining.sample(frac=0.50, random_state=42)
    test = remaining.drop(validation.index)

    output_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(output_dir / "train.csv", index=False)
    validation.to_csv(output_dir / "validation.csv", index=False)
    test.to_csv(output_dir / "test.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/opt/ml/processing/input/german_credit.csv")
    parser.add_argument("--output", default="/opt/ml/processing/output")
    args = parser.parse_args()
    preprocess(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
