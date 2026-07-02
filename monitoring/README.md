# Monitoring

Apply:

```bash
kubectl apply -f monitoring/servicemonitor.yaml
kubectl apply -f monitoring/prometheus-rules.yaml
```

Verify:

```bash
kubectl get servicemonitor -A
kubectl get prometheusrule -A
```

Metrics endpoint:

```
/metrics
```

