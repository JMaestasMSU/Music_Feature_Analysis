# Copilot Instructions for Music Feature Analysis

## Project Overview

- **Architecture**: Multi-component system for music/audio feature analysis.
  - **Backend API** (`backend/`): Python FastAPI app, exposes endpoints for feature extraction, health, and analysis.
  - **Model Server** (`models/`): Handles ML model inference, typically via PyTorch.
  - **Tests** (`tests/`): Standalone scripts for FFT, audio processing, CNN, and integration.
  - **Docker** (`docker/`): Compose files for local orchestration of backend and model server.
  - **Kubernetes** (`kubernetes/`): Manifests for production deployment.
  - **Notebooks** (`notebooks/`): EDA and modeling, for grading and exploration.
  - **Presentation** (`presentation/`): Project summary and slides.

## Developer Workflows

- **Quick Tests**: Run `bash tests/run_all_tests.sh` to validate FFT, audio, and model components.
- **API Testing**: Use `python backend/test_api.py` or `curl` endpoints (see `/api/v1/health`, `/api/v1/analysis/features`).
- **Full Stack**: Use `docker-compose up -d` in `docker/` to start backend and model server together.
- **Kubernetes**: Apply manifests in `kubernetes/` for production-like deployment (`kubectl apply -f ...`).
- **Notebook Review**: Open `notebooks/01_EDA.ipynb` and `notebooks/02_Modeling.ipynb` for data exploration and modeling.

## Project-Specific Patterns

- **Backend**: FastAPI, config via `.env` and `backend/config.py`. Health endpoint at `/api/v1/health`.
- **Model Server**: Expects PyTorch models in `models/trained_models/`. Batch size and device config in `models/app/server.py`.
- **Testing**: Each test script is standalone; `run_all_tests.sh` orchestrates them. Some tests (e.g., Parseval's theorem) require careful normalization.
- **Docker**: Separate Dockerfiles for backend and model server. Compose file binds ports 8000 (backend) and 8001 (model).
- **Kubernetes**: Namespace `music-analysis`, config via `configmap.yaml`, deployments and services for each component.

## Integration & Dependencies

- **Python 3.9+** required.
- **PyTorch, NumPy, SciPy, librosa, scikit-learn** for models and audio processing.
- **Docker** for local orchestration; **Kubernetes** for production.
- **No hardcoded secrets**; use `.env` and ConfigMaps.

## Conventions

- **Branching**: Main development on `main` and `develop`. Workflows only trigger on these branches.
- **Tests**: Must pass all in `tests/` before deployment.
- **Logs**: Use `docker-compose logs -f` or `kubectl logs` for debugging.
- **Performance tuning**: Adjust `CACHE_TTL`, `BATCH_SIZE`, and Docker/K8s resource limits as needed.

## Key Files & Directories

- `backend/app.py`, `backend/config.py`
- `models/app/server.py`, `models/trained_models/`
- `tests/run_all_tests.sh`, `tests/quick_fft_test.py`, `tests/quick_cnn_test.py`
- `docker/docker-compose.yml`, `docker/Dockerfile.backend`, `docker/Dockerfile.model`
- `kubernetes/deployment.yaml`, `kubernetes/service.yaml`, `kubernetes/configmap.yaml`
- `notebooks/01_EDA.ipynb`, `notebooks/02_Modeling.ipynb`
- `DEPLOYMENT.md`, `README.md`, `ARCHITECTURE.md`

---

**For more details, see `DEPLOYMENT.md` and `README.md`.**
