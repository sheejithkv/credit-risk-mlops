#!/bin/bash
set -euo pipefail

IMAGE_NAME="${1:-credit-risk-api}"
IMAGE_TAG="${2:-latest}"
PORT="${3:-8000}"

docker rm -f credit-risk-api >/dev/null 2>&1 || true

docker run \
  --name credit-risk-api \
  -p "${PORT}:8000" \
  "${IMAGE_NAME}:${IMAGE_TAG}"
