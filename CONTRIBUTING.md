# Contributing

Thank you for your interest in contributing to the Music Feature Analysis project.

Basic workflow
- Fork the repo and create a feature branch: `git checkout -b feat/your-feature`
- Implement changes, add tests, and run `pytest` and `ruff` locally
- Push and open a Pull Request to `main`; include a short description and testing steps

Coding standards
- Follow PEP8 for Python code and use `ruff` for linting.
- Use `mypy` for type hints where helpful; CI runs `mypy --ignore-missing-imports`.

Testing
- Add unit tests under `tests/` and keep them fast. Integration tests that require models should use `scripts/create_sample_model.py` to prepare artifacts.

PR checklist
- [ ] Code builds and tests pass locally
- [ ] New code has unit tests or a clear justification
- [ ] Documentation updated (README, QUICKSTART, or other docs)
- [ ] No secrets or large data files committed
