# Quick Start Guide - Music Feature Analysis

Get up and running in < 5 minutes.

---

## For Graders: View Deliverables

**Just want to grade the project?** Go directly to these files:

```bash
# Open graded notebooks
jupyter notebook notebooks/01_EDA.ipynb        # 15 points
jupyter notebook notebooks/02_Modeling.ipynb   # 5 points

# View presentation (submit day before)
open presentation/presentation.pdf             # 9 points

# Read project summary
open presentation/SUMMARY.md                   # 6 points
```

---

## For Students: Development Setup

### Prerequisites

- Python 3.9+
- Jupyter Notebook
- ~10 GB disk space (if downloading FMA dataset)
- (Optional) Docker for deployment
- (Optional) MATLAB for FFT validation

### 1. Clone and Navigate

```bash
cd /Users/P2956632/Documents/CS\ 3120/Music_Feature_Analysis/
```

### 2. Setup Data Directory

**Create data directory structure:**

**macOS/Linux:**
```bash
mkdir -p data/{raw,processed,metadata}
```

**Windows PowerShell:**
```powershell
New-Item -ItemType Directory -Path data\raw -Force
New-Item -ItemType Directory -Path data\processed -Force
New-Item -ItemType Directory -Path data\metadata -Force
```

**Windows Command Prompt:**
```batch
mkdir data\raw
mkdir data\processed
mkdir data\metadata
```

**Download dataset (choose one):**

**Option A: Full FMA Dataset (8 GB)**

*macOS/Linux:*
```bash
cd data/raw/
curl -L -o fma_small.zip https://os.unil.cloud.switch.ch/fma/fma_small.zip
unzip fma_small.zip
cd ../../
```

*Windows PowerShell:*
```powershell
cd data\raw\
Invoke-WebRequest -Uri "https://os.unil.cloud.switch.ch/fma/fma_small.zip" -OutFile "fma_small.zip"
Expand-Archive -Path fma_small.zip -DestinationPath . -Force
cd ..\..\
```

*Windows Command Prompt:*
```batch
cd data\raw\
curl -L -o fma_small.zip https://os.unil.cloud.switch.ch/fma/fma_small.zip
tar -xf fma_small.zip
cd ..\..
```

**Option B: Metadata Only (5 MB)**

*macOS/Linux:*
```bash
cd data/metadata/
curl -L -o fma_metadata.zip https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
unzip fma_metadata.zip
cd ../../
```

*Windows PowerShell:*
```powershell
cd data\metadata\
Invoke-WebRequest -Uri "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip" -OutFile "fma_metadata.zip"
Expand-Archive -Path fma_metadata.zip -DestinationPath . -Force
cd ..\..\
```

*Windows Command Prompt:*
```batch
cd data\metadata\
curl -L -o fma_metadata.zip https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
tar -xf fma_metadata.zip
cd ..\..
```

**Option C: Skip Download**  
Notebooks will use synthetic data automatically.

See **[data/README.md](data/README.md)** for details.

### 3. Create Virtual Environment

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows PowerShell:**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1

# If you get execution policy error:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Windows Command Prompt:**
```batch
python -m venv venv
venv\Scripts\activate.bat
```

### 4. Install Dependencies

**Core requirements (for notebooks):**
```bash
pip install -r requirements.txt
```

**Model inference server:**
```bash
# app/ has its own dependencies
cd app/
pip install fastapi uvicorn torch numpy pandas scikit-learn librosa
cd ..
```

**Backend API (if separate):**
```bash
cd backend/
pip install -r requirements.txt
cd ..
```

### 5. Run Jupyter Notebooks

```bash
# Start Jupyter
jupyter notebook

# Navigate to:
# - notebooks/01_EDA.ipynb (do this first)
# - notebooks/02_Modeling.ipynb (do this second)
```

---

## Verify Setup: Run Quick Tests

Prove that all components work (optional but recommended):

```bash
cd tests/
bash run_all_tests.sh
```

Expected output:
```
FFT validation test passed (10 sec)
Audio processing test passed (5 sec)
CNN architecture test passed (30 sec)
Bayesian optimization test passed (30 sec)

All tests completed in < 2 minutes
```

---

## Generate Presentation

### Option 1: Use RMarkdown (Recommended)

```bash
cd presentation/

# Install R dependencies (first time only)
Rscript -e "install.packages(c('rmarkdown', 'knitr'))"

# Compile slides to PDF
Rscript -e "rmarkdown::render('presentation.Rmd', output_format='pdf_document')"
```

### Option 2: Manual PDF Creation

1. Open `presentation/presentation.Rmd`
2. Copy content to Google Slides / PowerPoint
3. Export as PDF: `presentation/presentation.pdf`

---

## Run Model Inference Server

### Development Mode

```bash
cd app/
python main.py

# API runs at: http://localhost:8000
# Swagger docs: http://localhost:8000/docs
```

### Test API

```bash
# Health check
curl http://localhost:8000/health

# Predict genre (example)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.1, 0.2, 0.3, ...]}'
```

**Example API call (PowerShell):**
```powershell
Invoke-RestMethod -Uri http://localhost:8000/health -Method Get
```

**Example API call (curl):**
```bash
curl http://localhost:8000/api/v1/health
```

---

## Deploy with Docker (Optional)

### Build and Run

```bash
cd docker/
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

**Services:**
- Backend API: http://localhost:8000
- Model Server: http://localhost:8001
- Health checks: Built-in monitoring

---

## Development Workflow

### 1. Work on EDA
```bash
jupyter notebook notebooks/01_EDA.ipynb

# Complete these sections:
# - Dataset overview
# - Feature extraction
# - Statistical analysis
# - Visualizations (5+)
# - Preprocessing strategy
```

### 2. Work on Modeling
```bash
jupyter notebook notebooks/02_Modeling.ipynb

# Complete these sections:
# - Model architecture
# - Training procedure
# - Evaluation metrics
# - Results
# - Limitations
```

### 3. Create Presentation
```bash
cd presentation/

# Edit presentation.Rmd
# Generate PDF
# Verify 5 slides present
```

### 4. Document Findings
```bash
# Edit presentation/SUMMARY.md
# Include:
# - Key findings
# - Model insights
# - Limitations
# - Future work
```

### 5. Final Check
```bash
cd tests/
bash run_all_tests.sh  # Verify everything works
```

---

## Submission Checklist

Before submitting, verify:

- [ ] `notebooks/01_EDA.ipynb` runs without errors
- [ ] `notebooks/02_Modeling.ipynb` runs without errors
- [ ] `presentation/presentation.pdf` exists (< 10 MB)
- [ ] `presentation/SUMMARY.md` is complete
- [ ] All code is commented
- [ ] All visualizations have titles/labels
- [ ] Limitations are documented
- [ ] README.md is up-to-date

---

## Troubleshooting

### Jupyter Won't Start
```bash
pip install --upgrade jupyter notebook
jupyter notebook --version
```

### Missing Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Tests Fail
```bash
# Install test dependencies
pip install pytest numpy scipy librosa

# Run tests individually
cd tests/
python quick_fft_test.py
```

### Can't Generate PDF

```bash
# Option 1: Use online converter
# Upload .Rmd to https://pandoc.org/try/

# Option 2: Install Pandoc on macOS
brew install pandoc
```

### Docker Issues
```bash
# Check Docker is running
docker --version

# Rebuild containers
docker-compose down
docker-compose build --no-cache
docker-compose up
```

---

## Additional Resources

| Document | Purpose |
|----------|---------|
| [README.md](README.md) | Project overview and grading info |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System design and data flow |
| [DEPLOYMENT.md](DEPLOYMENT.md) | Production deployment guide |
| [PROJECT_ORGANIZATION.md](PROJECT_ORGANIZATION.md) | File structure guidelines |
| [tests/README.md](tests/README.md) | Testing documentation |

---

## Tips for Success

1. **Start with notebooks**: Complete `01_EDA.ipynb` before `02_Modeling.ipynb`
2. **Run cells sequentially**: Don't skip cells in notebooks
3. **Save often**: Jupyter autosaves, but manual save is safer
4. **Test early**: Run `tests/run_all_tests.sh` to catch issues
5. **Document as you go**: Add markdown cells explaining your work
6. **Check outputs**: Ensure all visualizations render correctly
7. **Review rubric**: See [PROJECT_ORGANIZATION.md](PROJECT_ORGANIZATION.md) for grading criteria

---

## Status Indicators

After setup, you should see:

[OK] Jupyter notebooks open  
[OK] All cells execute without errors  
[OK] Visualizations render correctly  
[OK] Tests pass (if run)  
[OK] Presentation PDF generated  
[OK] Summary document complete  

---

**Need help?** See [PROJECT_ORGANIZATION.md](PROJECT_ORGANIZATION.md) for detailed guidelines.

**Ready to submit?** Review checklist in [README.md](README.md#-submission-checklist).

---

**Last Updated**: 2024  
**Project Status**: Ready for Development

## macOS-Specific Notes

### Python Version

macOS typically has Python 3 pre-installed, but you may need to install it:

```bash
# Check Python version
python3 --version

# If not installed or old version, install via Homebrew
brew install python@3.11
```

### Installing Homebrew (if needed)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### Audio Libraries

librosa requires `ffmpeg` on macOS:

```bash
brew install ffmpeg
```

### Opening Files

```bash
# Open PDF on macOS
open presentation/presentation.pdf

# Open Markdown in default editor
open presentation/SUMMARY.md

# Open folder in Finder
open .
```

## Windows-Specific Notes

### Python Version

```powershell
# Check Python version
python --version

# If not installed, download from:
# https://www.python.org/downloads/
```

### Audio Libraries

*Option 1: Using Chocolatey (Recommended)*
```powershell
# Install Chocolatey (if not installed)
Set-ExecutionPolicy Bypass -Scope Process -Force
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Install ffmpeg
choco install ffmpeg
```

*Option 2: Manual Installation*
1. Download from https://www.gyan.dev/ffmpeg/builds/
2. Extract to `C:\ffmpeg`
3. Add to PATH:
   ```powershell
   [Environment]::SetEnvironmentVariable("Path", $env:Path + ";C:\ffmpeg\bin", "User")
   ```

### Opening Files

```powershell
# Open PDF
Start-Process presentation\presentation.pdf

# Open Markdown
Start-Process presentation\SUMMARY.md

# Open folder in Explorer
explorer .
```

## Linux (Ubuntu/Debian) Notes

**Python Version:**
```bash
# Check Python version
python3 --version

# Install if needed
sudo apt-get update
sudo apt-get install python3.11 python3-pip
```

**Audio Libraries:**
```bash
sudo apt-get install ffmpeg libsndfile1
```

**Opening Files:**
```bash
# Open PDF
xdg-open presentation/presentation.pdf

# Open Markdown
xdg-open presentation/SUMMARY.md
```

## Command Reference

### File Operations

| Action | macOS/Linux | Windows PowerShell | Windows CMD |
|--------|-------------|-------------------|-------------|
| **Create directory** | `mkdir -p dir/` | `New-Item -ItemType Directory -Path dir\ -Force` | `mkdir dir\` |
| **Navigate** | `cd dir/` | `cd dir\` | `cd dir\` |
| **Go back** | `cd ..` | `cd ..` | `cd ..` |
| **List files** | `ls` | `Get-ChildItem` or `ls` | `dir` |
| **Open file** | `open file` | `Start-Process file` | `start file` |
| **View file** | `cat file` | `Get-Content file` or `cat file` | `type file` |
| **Download** | `curl -L -o file url` | `Invoke-WebRequest -Uri url -OutFile file` | `curl -L -o file url` |
| **Extract ZIP** | `unzip file.zip` | `Expand-Archive file.zip` | `tar -xf file.zip` |

### Python Virtual Environment

| Action | macOS/Linux | Windows PowerShell | Windows CMD |
|--------|-------------|-------------------|-------------|
| **Create venv** | `python3 -m venv venv` | `python -m venv venv` | `python -m venv venv` |
| **Activate** | `source venv/bin/activate` | `.\venv\Scripts\Activate.ps1` | `venv\Scripts\activate.bat` |
| **Deactivate** | `deactivate` | `deactivate` | `deactivate` |
| **Install packages** | `pip install package` | `pip install package` | `pip install package` |
