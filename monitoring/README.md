# Corrected Monitoring Package

Meets assignment monitoring requirements:

- Prometheus metrics
- Prediction latency p95 target under 100ms
- Request rate
- Error rate
- Business cost metrics
- Grafana dashboard:
  - real-time system health
  - model performance trends
  - business impact visualization
- Alert rules:
  - API down
  - p95 latency > 100ms
  - error rate > 2%
  - model not ready
  - high business cost
  - pod restart spike
  - HPA max replicas

## Files

```text
api/main.py
monitoring/servicemonitor.yaml
monitoring/prometheus-rules.yaml
monitoring/alertmanager-config.yaml
monitoring/grafana/datasource.yaml
monitoring/grafana/dashboards/credit-risk-dashboard.json
```

## Apply

```bash
kubectl apply -f monitoring/servicemonitor.yaml
kubectl apply -f monitoring/prometheus-rules.yaml
kubectl apply -f monitoring/alertmanager-config.yaml
```

OpenShift:

```bash
oc apply -f monitoring/servicemonitor.yaml
oc apply -f monitoring/prometheus-rules.yaml
oc apply -f monitoring/alertmanager-config.yaml
```

## Local validation

```bash
python3 -m pytest tests -v
uvicorn api.main:app --host 0.0.0.0 --port 8000
curl -s http://localhost:8000/metrics | grep credit_risk
```

Expected metrics:

```text
credit_risk_requests_total
credit_risk_requests_failed_total
credit_risk_predictions_total
credit_risk_prediction_errors_total
credit_risk_prediction_latency_seconds
credit_risk_model_confidence
credit_risk_business_cost_usd_total
credit_risk_false_approvals_total
credit_risk_false_rejections_total
credit_risk_applications_processed_total
credit_risk_model_ready
```
