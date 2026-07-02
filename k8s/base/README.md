# Corrected Kubernetes Deployment

Meets assignment deployment requirements:

- Deployment
- Service
- ConfigMap
- Secret
- Resource limits: 2Gi memory, 1 CPU per pod
- Liveness and readiness probes
- HPA: 2-10 replicas
- Blue/green deployment via blue and green deployments
- Canary support through green deployment and canary service
- Rollback hook placeholder for error-rate based rollback

## Validate

```bash
kubectl apply --dry-run=client -k k8s/base
```

OpenShift:

```bash
oc apply --dry-run=client -k k8s/base
```

## Deploy

```bash
kubectl apply -k k8s/base
kubectl -n credit-risk get deploy,po,svc,hpa,pdb
```

## Port forward

```bash
kubectl -n credit-risk port-forward svc/credit-risk-api 8000:80
curl http://localhost:8000/health
curl http://localhost:8000/ready
```

## Switch service from blue to green

```bash
kubectl -n credit-risk scale deploy credit-risk-api-green --replicas=2
kubectl -n credit-risk patch svc credit-risk-api -p '{"spec":{"selector":{"app":"credit-risk-api","version":"green"}}}'
```

## Rollback to blue

```bash
kubectl -n credit-risk patch svc credit-risk-api -p '{"spec":{"selector":{"app":"credit-risk-api","version":"blue"}}}'
```
