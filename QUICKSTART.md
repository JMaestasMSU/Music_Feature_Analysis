# Quick start (scaffold)

These instructions set up the scaffolded Python package for development. They assume a Python 3.8+ environment.

1. Create and activate a virtual environment (Windows PowerShell):

```powershell
python -m venv venv; .\venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

Instead of manually installing, you can run the project bootstrap which will create the conda env if available or fall back to pip:

```powershell
.\scripts\install_requirements.ps1
```

3. Run tests (pytest):

```powershell
python -m pip install pytest; python -m pytest -q
```

4. Try the CLI stub:

```powershell
python cli.py dummy.wav
```

Contributing & Maintenance
--------------------------
See `CONTRIBUTING.md` for contribution workflow and `MAINTENANCE.md` for routine maintainer tasks.


Windows curl / PowerShell examples:

PowerShell (recommended):
```powershell
Invoke-RestMethod -Uri http://localhost:8000/predict -Method Post -ContentType 'application/json' -Body '{"features":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]}'
```

CMD (escape json):
```cmd
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d "{\"features\":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]}"
```

Admin `/admin/load-model` endpoint requires `ADMIN_TOKEN` in `.env` or environment and a simple token parameter.
