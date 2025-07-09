import os
import logging
from datetime import datetime
from fastapi import APIRouter
import tensorflow as tf
import numpy as np
import psycopg2
from PIL import Image

from config import IMAGE_DIR, UPLOAD_DIR
from models import HealthResponse
from database import test_database_connection
from ml_models import get_models_status
from utils import test_tensorflow, test_numpy, test_pil

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/health", response_model=HealthResponse)
def health_check():
    """Comprehensive health check"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        models_loaded=get_models_status(),
        imports_working={
            "tensorflow": test_tensorflow(),
            "numpy": test_numpy(),
            "pil": test_pil(),
            "psycopg2": True
        },
        directories_exist={
            "image_dir": os.path.exists(IMAGE_DIR),
            "upload_dir": os.path.exists(UPLOAD_DIR)
        },
        database_accessible=test_database_connection()
    )

@router.get("/test-imports")
def test_imports():
    """Test all imports individually"""
    results = {}
    
    # Test TensorFlow
    try:
        tf_version = tf.__version__
        results["tensorflow"] = {"status": "OK", "version": tf_version}
    except Exception as e:
        results["tensorflow"] = {"status": "ERROR", "error": str(e)}
    
    # Test NumPy
    try:
        np_version = np.__version__
        results["numpy"] = {"status": "OK", "version": np_version}
    except Exception as e:
        results["numpy"] = {"status": "ERROR", "error": str(e)}
    
    # Test PIL
    try:
        pil_version = Image.__version__ if hasattr(Image, '__version__') else "Unknown"
        results["pillow"] = {"status": "OK", "version": pil_version}
    except Exception as e:
        results["pillow"] = {"status": "ERROR", "error": str(e)}
    
    # Test psycopg2
    try:
        psycopg2_version = psycopg2.__version__
        results["psycopg2"] = {"status": "OK", "version": psycopg2_version}
    except Exception as e:
        results["psycopg2"] = {"status": "ERROR", "error": str(e)}
    
    return {"import_tests": results}
