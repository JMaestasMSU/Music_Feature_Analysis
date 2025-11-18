# Maintenance

Routine tasks for repository maintainers:

- Dependency updates: review `requirements.txt` and `environment-*.yml` quarterly. Use `pip-compile` or `conda-lock` to create reproducible artifacts.
- Security: run dependency vulnerability scans (e.g., GitHub Dependabot) and patch critical findings quickly.
- Tests: ensure CI passes on `main` and require PRs to run tests.
- Clean up large artifacts: keep `data/` and `outputs/` out of Git; use an artifacts store for large files.

Housekeeping commands
- Run linting and tests locally:
```powershell
python -m pip install -r scripts/dev-requirements.txt
python -m pytest -q
ruff check .
```
