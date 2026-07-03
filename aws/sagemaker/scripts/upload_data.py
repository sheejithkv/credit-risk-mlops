from __future__ import annotations

import argparse
from pathlib import Path

import boto3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-path", default="data/raw/german_credit.csv")
    parser.add_argument("--bucket", default="credit-risk-mlops-539357031810-us-east-1")
    parser.add_argument("--key", default="credit-risk/data/german_credit.csv")
    args = parser.parse_args()

    local_path = Path(args.local_path)
    if not local_path.exists():
        raise FileNotFoundError(local_path)

    boto3.client("s3").upload_file(str(local_path), args.bucket, args.key)
    print(f"Uploaded {local_path} to s3://{args.bucket}/{args.key}")


if __name__ == "__main__":
    main()
