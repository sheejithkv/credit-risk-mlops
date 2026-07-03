# Credit Risk MLOps Pipeline

## Executive Summary
Production-style end-to-end MLOps pipeline for credit risk prediction.

## Features
- Data validation
- Preprocessing
- Schema inference
- Train/Validation/Test split
- Random Forest + Optuna
- MLflow
- DVC
- FastAPI
- Docker
- Kubernetes
- Prometheus & Grafana
- AWS SageMaker (Pipeline, Registry, S3, ECR, IAM)

## Commands
bash
pytest tests -v
dvc repro
uvicorn app.main:app --reload


## API
- GET /health
- GET /ready
- POST /predict
- GET /metrics

## AWS
Pipeline creation succeeded. Execution was blocked because the AWS account has zero SageMaker Processing Job quota.

## Results
- 22 tests passed
- API validated
- Monitoring validated
- SageMaker pipeline submitted


# AWS SageMaker Implementation

This directory implements the AWS part of the assignment.

## Requirements covered

- SageMaker Processing for data preparation
- SageMaker Training Jobs
- SageMaker Pipelines with conditional execution
- SageMaker Model Registry for version management
- Cost/performance comparison with self-hosted approach

## Files

text
aws/sagemaker/config/sagemaker_config.yaml
aws/sagemaker/src/preprocess.py
aws/sagemaker/src/train.py
aws/sagemaker/src/evaluate.py
aws/sagemaker/src/inference.py
aws/sagemaker/pipelines/credit_risk_sagemaker_pipeline.py
aws/sagemaker/scripts/upload_data.py
aws/sagemaker/scripts/compare_costs.py


## Install AWS dependencies

bash
pip install -r requirements-aws.txt


## Configure

Edit:

text
aws/sagemaker/config/sagemaker_config.yaml


Set:

yaml
aws:
  region: ap-south-1
  default_bucket: <your-s3-bucket>
  execution_role_arn: arn:aws:iam::<account-id>:role/<sagemaker-execution-role>


## Upload data

bash
python3 aws/sagemaker/scripts/upload_data.py


## Print pipeline definition

bash
python3 aws/sagemaker/pipelines/credit_risk_sagemaker_pipeline.py


## Create/update pipeline

bash
python3 aws/sagemaker/pipelines/credit_risk_sagemaker_pipeline.py --upsert


## Start pipeline

bash
python3 aws/sagemaker/pipelines/credit_risk_sagemaker_pipeline.py --start


## Cost comparison

bash
python3 aws/sagemaker/scripts/compare_costs.py

