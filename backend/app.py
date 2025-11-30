from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from pathlib import Path
from datetime import datetime

from config import current_config
from routes import analysis, health

# Configure logging
logging.basicConfig(
    level=getattr(logging, current_config.LOG_LEVEL),
    format=current_config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(current_config.LOGS_DIR / "app.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=current_config.API_TITLE,
    description=current_config.API_DESCRIPTION,
    version=current_config.API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=current_config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response logging middleware
@app.middleware("http")
async def log_requests(request, call_next):
    """Log all incoming requests"""
    logger.info(f"{request.method} {request.url.path}")
    
    response = await call_next(request)
    
    logger.info(f"Response: {response.status_code}")
    return response


# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            'error': str(exc),
            'timestamp': datetime.now().isoformat()
        }
    )


# Include routers
app.include_router(health.router)
app.include_router(analysis.router)


@app.get("/")
async def root() -> dict:
    """Root endpoint with API information"""
    return {
        'name': current_config.API_TITLE,
        'version': current_config.API_VERSION,
        'description': current_config.API_DESCRIPTION,
        'docs': '/docs',
        'redoc': '/redoc',
        'endpoints': {
            'health': '/api/v1/health',
            'upload_and_analyze': 'POST /api/v1/analysis/upload',
            'quick_predict': 'POST /api/v1/analysis/predict',
            'batch_analyze': 'POST /api/v1/analysis/batch-analyze',
            'feature_comparison': 'POST /api/v1/analysis/compare-features'
        }
    }


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("=" * 70)
    logger.info("Starting Music Feature Analysis API")
    logger.info(f"Environment: {current_config.__class__.__name__}")
    logger.info(f"Upload directory: {current_config.UPLOAD_DIR}")
    logger.info(f"Logs directory: {current_config.LOGS_DIR}")
    logger.info(f"MATLAB integration: {'Enabled' if current_config.RUN_MATLAB_ANALYSIS else 'Disabled'}")
    logger.info("=" * 70)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Music Feature Analysis API")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level=current_config.LOG_LEVEL.lower()
    )
