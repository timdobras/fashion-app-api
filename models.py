from typing import List, Optional, Dict
from pydantic import BaseModel

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    models_loaded: dict
    imports_working: dict
    directories_exist: dict
    database_accessible: bool

class FashionItem(BaseModel):
    id: str
    image_url: str
    category: str
    similarity_score: float
    metadata: Optional[dict] = None

class ImageUploadResponse(BaseModel):
    success: bool
    message: str
    uploaded_image_id: str
    uploaded_image_url: str
    predicted_category: str
    confidence: float
    similar_items: List[FashionItem]
    processing_time: float

class BasicUploadResponse(BaseModel):
    success: bool
    message: str
    filename: str
    file_size: int
    image_dimensions: dict

class ClassificationResponse(BaseModel):
    success: bool
    predicted_category: str
    confidence: float
    all_predictions: List[dict]

class ImageWithSimilarResponse(BaseModel):
    success: bool
    message: str
    image_info: dict
    similar_items: List[FashionItem]
    search_time: float

class DeleteSummary(BaseModel):
    success: bool
    message: str
    deleted_count: int
    failed_deletions: List[dict]
    files_deleted: List[str]
    database_records_deleted: List[str]
    safety_checks_passed: bool
    dry_run: bool
    processing_time: Optional[float] = None
