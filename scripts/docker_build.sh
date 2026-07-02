#!/bin/bash
set -euo pipefail

IMAGE_NAME="${1:-credit-risk-api}"
IMAGE_TAG="${2:-latest}"

if [ ! -f "models/model.pkl" ]; then
    echo "ERROR: models/model.pkl not found. Run training first."
    exit 1
fi

if [ ! -f "data/processed/schema.json" ]; then
    echo "ERROR: data/processed/schema.json not found. Run schema stage first."
    exit 1
fi

docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" .
