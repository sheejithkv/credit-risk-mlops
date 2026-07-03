from __future__ import annotations

import argparse
import time

import boto3

REGION = "us-east-1"


def wait_for_endpoint(client, endpoint_name: str) -> None:
    while True:
        response = client.describe_endpoint(EndpointName=endpoint_name)
        status = response["EndpointStatus"]
        print(f"Endpoint status: {status}")
        if status == "InService":
            return
        if status in {"Failed", "OutOfService"}:
            raise RuntimeError(response)
        time.sleep(30)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--endpoint-name", default="credit-risk-endpoint")
    parser.add_argument("--instance-type", default="ml.m5.large")
    args = parser.parse_args()

    client = boto3.client("sagemaker", region_name=REGION)
    endpoint_config_name = f"{args.endpoint_name}-config"

    try:
        client.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
    except Exception:
        pass

    client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": args.model_name,
                "InitialInstanceCount": 1,
                "InstanceType": args.instance_type,
                "InitialVariantWeight": 1.0,
            }
        ],
    )

    endpoints = client.list_endpoints(NameContains=args.endpoint_name)["Endpoints"]
    if any(item["EndpointName"] == args.endpoint_name for item in endpoints):
        client.update_endpoint(EndpointName=args.endpoint_name, EndpointConfigName=endpoint_config_name)
    else:
        client.create_endpoint(EndpointName=args.endpoint_name, EndpointConfigName=endpoint_config_name)

    wait_for_endpoint(client, args.endpoint_name)


if __name__ == "__main__":
    main()
