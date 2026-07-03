from __future__ import annotations


def main() -> None:
    sagemaker_monthly = {
        "processing_training_estimate": 20.0,
        "real_time_endpoint_ml_m5_large_730h": 83.95,
        "s3_10gb": 0.23,
        "cloudwatch_logs_estimate": 5.0,
    }
    kubernetes_monthly = {
        "worker_capacity_estimate": 240.0,
        "monitoring_storage": 30.0,
        "ops_overhead": 100.0,
    }

    print("Estimated SageMaker monthly USD:", round(sum(sagemaker_monthly.values()), 2))
    print("Estimated self-hosted Kubernetes monthly USD:", round(sum(kubernetes_monthly.values()), 2))
    print("SageMaker: managed training, registry, batch transform, endpoints.")
    print("Kubernetes: best when shared cluster capacity already exists.")


if __name__ == "__main__":
    main()
