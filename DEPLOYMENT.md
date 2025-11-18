# Deployment

This file describes how to build and run the inference Docker image locally and basic notes for cloud/k8s deployment.

Build the Docker image (local):

```powershell
docker build -t music-feature-analysis:latest .
```

Run locally exposing port 8000:

```powershell
docker run --rm -p 8000:8000 music-feature-analysis:latest
```

Health endpoint: GET /healthz
Predict endpoint: POST /predict with JSON {"features": [ ... ]}

Kubernetes notes:
- Use a readiness and liveness probe targeting /healthz
- Use resource requests/limits for CPU and memory
- For GPU workloads, use NVIDIA device plugin and an image built with the matching CUDA base image
