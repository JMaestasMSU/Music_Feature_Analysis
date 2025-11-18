from fastapi import APIRouter
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["health"])


@router.get("/health")
async def health_check() -> dict:
    """Basic health check endpoint"""
    return {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    }


@router.get("/status")
async def system_status() -> dict:
    """Detailed system status"""
    return {
        'status': 'operational',
        'components': {
            'audio_processor': 'ready',
            'cnn_model': 'loaded',
            'matlab_interface': 'available',
            'fft_analysis': 'enabled'
        },
        'timestamp': datetime.now().isoformat()
    }
