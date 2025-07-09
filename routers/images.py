import os
import io
import uuid
import logging
from datetime import datetime
from fastapi import APIRouter, HTTPException, File, UploadFile
from fastapi.responses import FileResponse
from PIL import Image

from config import UPLOAD_DIR, IMAGE_DIR, CATEGORY_NAMES
from models import ImageUploadResponse, BasicUploadResponse, ClassificationResponse
from database import save_image_to_database, find_similar_fashion_items
from ml_models import generate_image_embedding, classify_image

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/upload-fashion-image", response_model=ImageUploadResponse)
async def upload_fashion_image(file: UploadFile = File(...)):
    """Upload a fashion image, classify it, and find similar items"""
    start_time = datetime.utcnow()
    logger.info(f"Fashion image upload started for file: {file.filename}")
    
    # Validate file
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Check file size (10MB limit)
    if file.size and file.size > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 10MB)")
    
    try:
        # Generate unique filename
        file_extension = file.filename.split('.')[-1].lower() if file.filename else "jpg"
        unique_filename = f"{uuid.uuid4()}.{file_extension}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)
        
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convert to RGB if necessary
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")
        
        # Save image
        image.save(file_path, "JPEG", quality=95)
        logger.info(f"Image saved to: {file_path}")
        
        # Generate embedding
        embedding = await generate_image_embedding(image)
        logger.info("Image embedding generated successfully")
        
        # Classify image
        predicted_category, confidence, all_predictions = await classify_image(image)
        logger.info(f"Image classified as: {predicted_category} (confidence: {confidence:.3f})")
        
        # Save to database
        image_id = await save_image_to_database(
            unique_filename, file_path, embedding, predicted_category
        )
        logger.info(f"Image saved to database with ID: {image_id}")
        
        # Find similar items
        similar_items = await find_similar_fashion_items(embedding, limit=10)
        logger.info(f"Found {len(similar_items)} similar items")
        
        # Calculate processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        return ImageUploadResponse(
            success=True,
            message="Image uploaded and processed successfully",
            uploaded_image_id=image_id,
            uploaded_image_url=f"/images/{unique_filename}",
            predicted_category=predicted_category,
            confidence=confidence,
            similar_items=similar_items,
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        # Clean up file if it exists
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@router.post("/classify-image", response_model=ClassificationResponse) 
async def classify_uploaded_image(file: UploadFile = File(...)):
    """Classify an uploaded image without saving or similarity search"""
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")
        
        # Classify image
        predicted_category, confidence, all_predictions = await classify_image(image)
        
        return ClassificationResponse(
            success=True,
            predicted_category=predicted_category,
            confidence=confidence,
            all_predictions=all_predictions[:5]  # Top 5 predictions
        )
        
    except Exception as e:
        logger.error(f"Classification error: {e}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

@router.post("/test-upload", response_model=BasicUploadResponse)
async def test_upload(file: UploadFile = File(...)):
    """Simple file upload test without ML processing"""
    
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Generate unique filename
        file_extension = file.filename.split('.')[-1].lower() if file.filename else "jpg"
        unique_filename = f"test_{uuid.uuid4()}.{file_extension}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)
        
        # Read file
        contents = await file.read()
        
        # Test PIL processing
        image = Image.open(io.BytesIO(contents))
        width, height = image.size
        
        # Convert to RGB if necessary
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")
        
        # Save file
        image.save(file_path, "JPEG", quality=95)
        
        logger.info(f"Test image saved to: {file_path}")
        
        return BasicUploadResponse(
            success=True,
            message="File uploaded and processed successfully",
            filename=unique_filename,
            file_size=len(contents),
            image_dimensions={"width": width, "height": height}
        )
        
    except Exception as e:
        logger.error(f"Upload test error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload test failed: {str(e)}")

@router.get("/images/{image_name}", response_class=FileResponse)
async def get_image(image_name: str):
    """Serve images from upload, validation, and training directories"""
    logger.info(f"Image requested: {image_name}")
    
    # Security check
    if ".." in image_name or "/" in image_name:
        raise HTTPException(status_code=403, detail="Invalid image name")
    
    # Define all possible image locations in order of priority
    image_locations = [
        # 1. Uploaded images (highest priority)
        os.path.join(UPLOAD_DIR, image_name),
        
        # 2. Validation images
        os.path.join(IMAGE_DIR, image_name),
        
        # 3. Training images
        os.path.join("/mnt/truenas/fashion-app-ml/data/train/image", image_name),
    ]
    
    # Check each location
    for image_path in image_locations:
        if os.path.exists(image_path):
            logger.info(f"Serving image from: {image_path}")
            return FileResponse(image_path)
    
    # If not found in any location, log the search paths for debugging
    logger.error(f"Image '{image_name}' not found in any of these locations:")
    for path in image_locations:
        logger.error(f"  - {path}")
    
    raise HTTPException(status_code=404, detail="Image not found")

@router.get("/categories")
def get_categories():
    """Get list of all fashion categories"""
    return {
        "categories": CATEGORY_NAMES,
        "total": len(CATEGORY_NAMES)
    }
