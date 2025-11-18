# Reproducible lockfile guidance

Recommended approaches for reproducible environments in enterprise:

- Conda-lock (cross-platform):

```powershell
conda install -n base -c conda-forge conda-lock
conda-lock lock --file environment-cpu.yml
conda-lock render --file conda-lock.yml --platform win-64
```

- Pip-tools / pip-compile (Python wheel-based):

```powershell
# create requirements.in with top-level deps then:
pip-compile requirements.in --output-file requirements.txt
pip-sync requirements.txt
```

Choose the strategy that matches your deployment stack (conda for heavy native deps, pip for lightweight wheel-only deployments).