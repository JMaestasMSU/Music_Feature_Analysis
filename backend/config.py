import os
from pathlib import Path
from typing import Optional

class Config:
    """Configuration management"""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent
    BACKEND_DIR = Path(__file__).parent
    MATLAB_DIR = PROJECT_ROOT / "matlab"
    MODELS_DIR = PROJECT_ROOT / "models" / "trained_models"
    UPLOAD_DIR = BACKEND_DIR / "uploads"
    LOGS_DIR = BACKEND_DIR / "logs"
    
    # Create directories if they don't exist
    UPLOAD_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)
    
    # Audio processing
    SAMPLE_RATE = 44100
    CHUNK_DURATION = 1.0  # seconds per chunk for analysis
    N_MELS = 128
    N_FFT = 2048
    HOP_LENGTH = 512
    
    # CNN Model
    CNN_MODEL_PATH = MODELS_DIR / "models" / "trained_models" / "multilabel_cnn_filtered_improved" / "best_model.pt"
    MODEL_INPUT_SIZE = (128, 216)  # (n_mels, time_steps)
    
    # API Configuration
    API_TITLE = "Music Feature Analysis API"
    API_VERSION = "1.0.0"
    API_DESCRIPTION = "End-to-end spectrogram analysis with CNN + FFT validation"
    
    # CORS
    ALLOWED_ORIGINS = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
    ]
    
    # File limits
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
    ALLOWED_AUDIO_FORMATS = ["audio/mpeg", "audio/wav", "audio/x-wav"]
    
    # MATLAB
    MATLAB_EXECUTABLE = os.getenv("MATLAB_PATH", "matlab")
    MATLAB_TIMEOUT = 60  # seconds
    RUN_MATLAB_ANALYSIS = True
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Cache
    CACHE_TTL = 3600  # seconds
    ENABLE_CACHING = True
    
    # Database (optional SQLite for metadata)
    DATABASE_URL = f"sqlite:///{BACKEND_DIR}/music_analysis.db"
    
    @classmethod
    def get_matlab_script_path(cls, script_name: str) -> Path:
        """Get path to MATLAB script"""
        return cls.MATLAB_DIR / f"{script_name}.m"
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate critical configuration paths"""
        required_files = [
            cls.MATLAB_DIR / "spectral_analysis.m",
            cls.MATLAB_DIR / "fft_validation.m",
        ]
        
        for file in required_files:
            if not file.exists():
                print(f"Warning: {file} not found")
                return False
        
        return True

# Environment-specific configs
class DevelopmentConfig(Config):
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    DEBUG = False
    TESTING = False
    ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "").split(",")

class TestingConfig(Config):
    DEBUG = True
    TESTING = True
    UPLOAD_DIR = Path("/tmp/music_analysis_test")

# Select config based on environment
ENV = os.getenv("ENVIRONMENT", "development").lower()
if ENV == "production":
    current_config = ProductionConfig()
elif ENV == "testing":
    current_config = TestingConfig()
else:
    current_config = DevelopmentConfig()
