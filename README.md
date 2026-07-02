# Real-Time Credit Risk Prediction System

Production MLOps assignment skeleton for German Credit Risk prediction.

## Module 0 scope

This module establishes the repository structure, configuration loading, logging, dataset validation skeleton, test baseline, and DVC pipeline entrypoint.

## Quick validation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
python3 -m pytest tests -v
python3 -m src.credit_risk.data.validate
dvc repro
```

## Dataset

Place the German Credit dataset under:

```text
data/raw/german_credit.csv
```

For Module 0, validation supports missing dataset mode so the project can bootstrap cleanly.
