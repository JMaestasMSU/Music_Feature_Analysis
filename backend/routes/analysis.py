from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import logging
from pathlib import Path
import shutil
from typing import Optional
import json
from datetime import datetime

from ..config import current_config
from ..services.audio_processor import AudioProcessor
from ..services.matlab_interface import MATLABInterface
from ..services.model_loader import ModelLoader

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/analysis", tags=["analysis"])

# Initialize services
audio_processor = AudioProcessor(
    sample_rate=current_config.SAMPLE_RATE,
    n_mels=current_config.N_MELS,
    n_fft=current_config.N_FFT
)
matlab_interface = MATLABInterface(matlab_dir=current_config.MATLAB_DIR)
model_loader = ModelLoader(
    model_path=current_config.CNN_MODEL_PATH,
    device="cpu"
)


@router.post("/upload")
async def upload_and_analyze(file: UploadFile = File(...)) -> dict:
    """
    Upload audio file and perform complete analysis
    
    - CNN genre prediction
    - Spectral features (librosa)
    - FFT analysis (MATLAB/NumPy)
    - Feature correlation
    """
    try:
        # Validate file
        if not file.content_type in current_config.ALLOWED_AUDIO_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed: {current_config.ALLOWED_AUDIO_FORMATS}"
            )
        
        # Save uploaded file
        file_path = current_config.UPLOAD_DIR / f"{datetime.now().timestamp()}_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"File uploaded: {file_path}")
        
        # Process audio
        processing_result = audio_processor.process_audio_file(
            str(file_path),
            target_shape=current_config.MODEL_INPUT_SIZE
        )
        
        # CNN prediction
        cnn_prediction = model_loader.predict(processing_result['spectrogram_cnn'])
        
        # FFT spectral analysis
        fft_results = matlab_interface.run_fft_analysis(
            processing_result['audio'],
            processing_result['sample_rate']
        )
        
        # Compare FFT vs ML features
        feature_comparison = matlab_interface.validate_fft_vs_ml_features(
            fft_results,
            processing_result['spectral_features']
        )
        
        # Compile results
        analysis_result = {
            'file_name': file.filename,
            'upload_time': datetime.now().isoformat(),
            'file_path': str(file_path),
            'audio_info': {
                'sample_rate': processing_result['sample_rate'],
                'duration_seconds': len(processing_result['audio']) / processing_result['sample_rate'],
                'shape': processing_result['shape']
            },
            'cnn_prediction': cnn_prediction,
            'spectral_features': processing_result['spectral_features'],
            'fft_analysis': fft_results,
            'feature_correlation': feature_comparison,
            'validation_status': 'success'
        }
        
        logger.info(f"Analysis complete for {file.filename}")
        return analysis_result
    
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict")
async def predict_genre(file: UploadFile = File(...)) -> dict:
    """
    Quick prediction endpoint (CNN only, no FFT analysis)
    """
    try:
        # Save file temporarily
        file_path = current_config.UPLOAD_DIR / f"pred_{datetime.now().timestamp()}_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process and predict
        processing_result = audio_processor.process_audio_file(
            str(file_path),
            target_shape=current_config.MODEL_INPUT_SIZE
        )
        
        cnn_prediction = model_loader.predict(processing_result['spectrogram_cnn'])
        
        return {
            'file_name': file.filename,
            'prediction': cnn_prediction,
            'timestamp': datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch-analyze")
async def batch_analyze(files: list[UploadFile] = File(...)) -> dict:
    """
    Analyze multiple audio files in batch
    """
    try:
        results = []
        
        for file in files:
            if file.content_type not in current_config.ALLOWED_AUDIO_FORMATS:
                results.append({
                    'file_name': file.filename,
                    'success': False,
                    'error': f"Invalid file type: {file.content_type}"
                })
                continue
            
            try:
                # Save file
                file_path = current_config.UPLOAD_DIR / f"batch_{datetime.now().timestamp()}_{file.filename}"
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                
                # Process
                processing_result = audio_processor.process_audio_file(
                    str(file_path),
                    target_shape=current_config.MODEL_INPUT_SIZE
                )
                
                # Predict
                cnn_prediction = model_loader.predict(processing_result['spectrogram_cnn'])
                
                results.append({
                    'file_name': file.filename,
                    'success': True,
                    'prediction': cnn_prediction
                })
            
            except Exception as e:
                results.append({
                    'file_name': file.filename,
                    'success': False,
                    'error': str(e)
                })
        
        return {
            'batch_size': len(files),
            'successful': sum(1 for r in results if r['success']),
            'failed': sum(1 for r in results if not r['success']),
            'results': results,
            'timestamp': datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Batch analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/fft-validation")
async def fft_validation_info() -> dict:
    """
    Get information about FFT validation methodology
    """
    return {
        'title': 'FFT Spectral Analysis Validation',
        'description': 'Validates CNN features against mathematical FFT analysis',
        'methodology': {
            'fft_method': 'Fast Fourier Transform (NumPy/MATLAB)',
            'spectral_features': ['centroid', 'spread', 'rolloff'],
            'validation': 'Parseval\'s Theorem energy equivalence',
            'ml_features': ['spectral_centroid', 'zero_crossing_rate', 'mfcc', 'chroma']
        },
        'matlab_available': matlab_interface.validate_matlab_available(),
        'fallback_enabled': True
    }


@router.get("/features")
async def get_feature_list() -> dict:
    """
    Get list of all computed features
    """
    return {
        'spectral_features': [
            'spectral_centroid',
            'spectral_rolloff',
            'zero_crossing_rate',
            'mfcc_mean',
            'mfcc_std',
            'chroma_mean',
            'rms_energy'
        ],
        'fft_features': [
            'spectral_centroid',
            'spectral_spread',
            'spectral_rolloff',
            'time_domain_energy',
            'freq_domain_energy'
        ],
        'cnn_output': [
            'predicted_genre',
            'confidence',
            'top_3_predictions',
            'all_probabilities'
        ]
    }


@router.post("/compare-features")
async def compare_fft_vs_ml(file: UploadFile = File(...)) -> dict:
    """
    Detailed comparison: FFT analysis vs ML-extracted features
    Useful for numerical analysis validation
    """
    try:
        # Save file
        file_path = current_config.UPLOAD_DIR / f"compare_{datetime.now().timestamp()}_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process
        processing_result = audio_processor.process_audio_file(
            str(file_path),
            target_shape=current_config.MODEL_INPUT_SIZE
        )
        
        # FFT analysis
        fft_results = matlab_interface.run_fft_analysis(
            processing_result['audio'],
            processing_result['sample_rate']
        )
        
        # Detailed comparison
        comparison = {
            'file_name': file.filename,
            'audio_stats': {
                'duration': len(processing_result['audio']) / processing_result['sample_rate'],
                'sample_rate': processing_result['sample_rate'],
                'samples': len(processing_result['audio'])
            },
            'fft_analysis': fft_results,
            'ml_features': processing_result['spectral_features'],
            'correlation_analysis': {
                'spectral_centroid': {
                    'fft': fft_results['spectral_centroid'],
                    'ml': processing_result['spectral_features']['spectral_centroid'],
                    'difference_hz': abs(
                        fft_results['spectral_centroid'] - 
                        processing_result['spectral_features']['spectral_centroid']
                    ),
                    'pct_difference': abs(
                        (fft_results['spectral_centroid'] - 
                         processing_result['spectral_features']['spectral_centroid']) /
                        max(fft_results['spectral_centroid'], 1e-8) * 100
                    )
                },
                'energy_comparison': {
                    'time_domain': fft_results['time_domain_energy'],
                    'freq_domain': fft_results['freq_domain_energy'],
                    'parseval_error_pct': abs(
                        (fft_results['time_domain_energy'] - fft_results['freq_domain_energy']) /
                        max(fft_results['time_domain_energy'], 1e-8) * 100
                    )
                }
            },
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Feature comparison complete for {file.filename}")
        return comparison
    
    except Exception as e:
        logger.error(f"Feature comparison failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
