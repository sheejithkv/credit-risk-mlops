from __future__ import annotations

import argparse

import sagemaker
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.functions import Join, JsonGet
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.parameters import ParameterFloat, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import ProcessingStep, TrainingStep

REGION = "us-east-1"
ROLE_ARN = "arn:aws:iam::539357031810:role/SageMakerExecutionRole"
BUCKET = "credit-risk-mlops-539357031810-us-east-1"
PIPELINE_NAME = "credit-risk-sagemaker-pipeline"
MODEL_PACKAGE_GROUP_NAME = "credit-risk-model-group"


def sklearn_image(region: str) -> str:
    return sagemaker.image_uris.retrieve(
        framework="sklearn",
        region=region,
        version="1.2-1",
        py_version="py3",
        instance_type="ml.m5.xlarge",
    )


def build_pipeline(role_arn: str = ROLE_ARN, bucket: str = BUCKET, region: str = REGION) -> Pipeline:
    pipeline_session = PipelineSession()

    input_data = ParameterString(
        name="InputData",
        default_value=f"s3://{bucket}/credit-risk/data/german_credit.csv",
    )
    minimum_roc_auc = ParameterFloat(name="MinimumRocAuc", default_value=0.70)

    processor = ScriptProcessor(
        image_uri=sklearn_image(region),
        command=["python3"],
        role=role_arn,
        instance_type="ml.m5.xlarge",
        instance_count=1,
        base_job_name="credit-risk-preprocess",
        sagemaker_session=pipeline_session,
    )

    preprocess_step = ProcessingStep(
        name="PreprocessCreditRiskData",
        processor=processor,
        inputs=[
            ProcessingInput(
                source=input_data,
                destination="/opt/ml/processing/input/german_credit.csv",
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="train",
                source="/opt/ml/processing/output/train.csv",
                destination=f"s3://{bucket}/credit-risk/processing/train",
            ),
            ProcessingOutput(
                output_name="validation",
                source="/opt/ml/processing/output/validation.csv",
                destination=f"s3://{bucket}/credit-risk/processing/validation",
            ),
            ProcessingOutput(
                output_name="test",
                source="/opt/ml/processing/output/test.csv",
                destination=f"s3://{bucket}/credit-risk/processing/test",
            ),
        ],
        code="aws/sagemaker/src/preprocess.py",
    )

    estimator = SKLearn(
        entry_point="train.py",
        source_dir="aws/sagemaker/src",
        framework_version="1.2-1",
        py_version="py3",
        role=role_arn,
        instance_type="ml.m5.xlarge",
        instance_count=1,
        output_path=f"s3://{bucket}/credit-risk/models",
        base_job_name="credit-risk-train",
        sagemaker_session=pipeline_session,
    )

    train_s3 = preprocess_step.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri
    validation_s3 = preprocess_step.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri
    test_s3 = preprocess_step.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri

    training_step = TrainingStep(
        name="TrainCreditRiskModel",
        estimator=estimator,
        inputs={
            "train": train_s3,
            "validation": validation_s3,
        },
    )

    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )

    evaluator = ScriptProcessor(
        image_uri=sklearn_image(region),
        command=["python3"],
        role=role_arn,
        instance_type="ml.m5.xlarge",
        instance_count=1,
        base_job_name="credit-risk-evaluate",
        sagemaker_session=pipeline_session,
    )

    evaluation_step = ProcessingStep(
        name="EvaluateCreditRiskModel",
        processor=evaluator,
        inputs=[
            ProcessingInput(
                source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model/model.tar.gz",
            ),
            ProcessingInput(
                source=test_s3,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/evaluation",
                destination=f"s3://{bucket}/credit-risk/evaluation",
            )
        ],
        code="aws/sagemaker/src/evaluate.py",
        property_files=[evaluation_report],
    )

    model = SKLearnModel(
        model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
        role=role_arn,
        entry_point="inference.py",
        source_dir="aws/sagemaker/src",
        framework_version="1.2-1",
        py_version="py3",
        sagemaker_session=pipeline_session,
    )

    evaluation_json_s3 = Join(
        on="/",
        values=[
            evaluation_step.properties.ProcessingOutputConfig.Outputs["evaluation"].S3Output.S3Uri,
            "evaluation.json",
        ],
    )

    register_args = model.register(
        content_types=["application/json", "text/csv"],
        response_types=["application/json"],
        inference_instances=["ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=MODEL_PACKAGE_GROUP_NAME,
        approval_status="PendingManualApproval",
        model_metrics=ModelMetrics(
            model_statistics=MetricsSource(
                s3_uri=evaluation_json_s3,
                content_type="application/json",
            )
        ),
    )

    register_step = ModelStep(name="RegisterCreditRiskModel", step_args=register_args)

    condition_step = ConditionStep(
        name="CheckModelQuality",
        conditions=[
            ConditionGreaterThanOrEqualTo(
                left=JsonGet(
                    step_name=evaluation_step.name,
                    property_file=evaluation_report,
                    json_path="classification_metrics.roc_auc.value",
                ),
                right=minimum_roc_auc,
            )
        ],
        if_steps=[register_step],
        else_steps=[],
    )

    return Pipeline(
        name=PIPELINE_NAME,
        parameters=[input_data, minimum_roc_auc],
        steps=[preprocess_step, training_step, evaluation_step, condition_step],
        sagemaker_session=pipeline_session,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role-arn", default=ROLE_ARN)
    parser.add_argument("--bucket", default=BUCKET)
    parser.add_argument("--region", default=REGION)
    args = parser.parse_args()

    pipeline = build_pipeline(args.role_arn, args.bucket, args.region)
    pipeline.upsert(role_arn=args.role_arn)
    execution = pipeline.start()
    print(f"Started SageMaker pipeline execution: {execution.arn}")


if __name__ == "__main__":
    main()
