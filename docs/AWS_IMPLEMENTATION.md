# AWS SageMaker Production Implementation

## Real AWS values used

text
AWS Account ID: 539357031810
AWS Region: us-east-1
S3 Bucket: credit-risk-mlops-539357031810-us-east-1
SageMaker Role ARN: arn:aws:iam::539357031810:role/SageMakerExecutionRole
ECR Repository: 539357031810.dkr.ecr.us-east-1.amazonaws.com/credit-risk-api
Pipeline Name: credit-risk-sagemaker-pipeline
Model Package Group: credit-risk-model-group
Endpoint Name: credit-risk-endpoint


## What this implements

- SageMaker Processing Job for preprocessing.
- SageMaker Training Job for RandomForest training.
- SageMaker Evaluation Processing Job.
- SageMaker Pipeline with conditional registration.
- SageMaker Model Registry with manual approval.
- Real-time endpoint helper.
- S3 artifact storage.
- Cost comparison versus self-hosted Kubernetes.

## Install

bash
pip install -r requirements-aws.txt


## Validate code locally

bash
python3 -m py_compile \
  aws/sagemaker/src/preprocess.py \
  aws/sagemaker/src/train.py \
  aws/sagemaker/src/evaluate.py \
  aws/sagemaker/src/inference.py \
  aws/sagemaker/pipelines/credit_risk_sagemaker_pipeline.py \
  aws/sagemaker/scripts/upload_data.py \
  aws/sagemaker/scripts/create_endpoint.py \
  aws/sagemaker/scripts/compare_costs.py


## Upload dataset

bash
python3 aws/sagemaker/scripts/upload_data.py


## Start pipeline

bash
python3 aws/sagemaker/pipelines/credit_risk_sagemaker_pipeline.py


## Check execution

bash
aws sagemaker list-pipeline-executions \
  --pipeline-name credit-risk-sagemaker-pipeline \
  --region us-east-1


## Check Model Registry

bash
aws sagemaker list-model-package-groups --region us-east-1

aws sagemaker list-model-packages \
  --model-package-group-name credit-risk-model-group \
  --region us-east-1


## Manual approval

bash
aws sagemaker update-model-package \
  --model-package-arn <MODEL_PACKAGE_ARN> \
  --model-approval-status Approved \
  --region us-east-1


## Endpoint deployment

Create a SageMaker model from the approved package or from a training artifact, then run:

bash
python3 aws/sagemaker/scripts/create_endpoint.py --model-name <SAGEMAKER_MODEL_NAME>


## Cost comparison

bash
python3 aws/sagemaker/scripts/compare_costs.py

