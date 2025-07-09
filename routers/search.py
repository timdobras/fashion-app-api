import logging
import numpy as np
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, HTTPException

from models import FashionItem
from database import get_db_connection, determine_image_type

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/search-by-id/{image_id}")
async def search_similar_by_id_enhanced(
    image_id: int, 
    limit: int = 10, 
    min_similarity: float = 0.0,
    category_filter: Optional[str] = None
):
    """Enhanced search by database ID with filtering options"""
    start_time = datetime.utcnow()
    
    # Validate parameters
    if limit > 50:
        limit = 50
    if not 0.0 <= min_similarity <= 1.0:
        raise HTTPException(status_code=400, detail="min_similarity must be between 0.0 and 1.0")
    
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Get target image information - retrieve embedding as text
        sql_get_image = """
        SELECT id, public_path, category, embedding::text as embedding_text
        FROM fashion_items 
        WHERE id = %s;
        """
        
        cur.execute(sql_get_image, (image_id,))
        image_row = cur.fetchone()
        
        if not image_row:
            cur.close()
            conn.close()
            raise HTTPException(status_code=404, detail=f"Image with ID {image_id} not found")
        
        target_id, target_path, target_category, embedding_text = image_row
        
        # Build target image info
        target_filename = target_path.split('/')[-1] if target_path else f"item_{target_id}.jpg"
        target_type = determine_image_type(target_path)
        
        target_image_info = {
            "id": str(target_id),
            "filename": target_filename,
            "type": target_type,
            "url": f"/images/{target_filename}",
            "category": target_category,
            "database_path": target_path,
            "has_embedding": embedding_text is not None
        }
        
        if not embedding_text:
            cur.close()
            conn.close()
            raise HTTPException(status_code=400, detail=f"Image {image_id} has no embedding for similarity search")
        
        # Build similarity search query with optional filters
        sql_similar = """
        SELECT id, public_path, category, embedding <-> %s::vector AS distance
        FROM fashion_items 
        WHERE id != %s
        """
        params = [embedding_text, target_id]
        
        # Add category filter if specified
        if category_filter:
            sql_similar += " AND category = %s"
            params.append(category_filter)
        
        # Add similarity threshold if specified
        max_distance = 1.0 - min_similarity
        sql_similar += " AND (embedding <-> %s::vector) <= %s"
        params.extend([embedding_text, max_distance])
        
        sql_similar += " ORDER BY distance LIMIT %s;"
        params.append(limit)
        
        cur.execute(sql_similar, params)
        similar_rows = cur.fetchall()
        cur.close()
        conn.close()
        
        # Process similar items
        similar_items = []
        for i, (item_id, path, category, distance) in enumerate(similar_rows):
            image_filename = path.split('/')[-1] if path else f"item_{item_id}.jpg"
            similarity_score = max(0.0, 1.0 - float(distance))
            
            # Apply similarity threshold check
            if similarity_score >= min_similarity:
                similar_items.append(FashionItem(
                    id=str(item_id),
                    image_url=f"/images/{image_filename}",
                    category=category or "unknown",
                    similarity_score=similarity_score,
                    metadata={
                        "distance": float(distance),
                        "rank": i + 1,
                        "original_path": path,
                        "category_match": category == target_category if target_category and category else False
                    }
                ))
        
        search_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Statistics
        category_matches = len([item for item in similar_items if item.metadata.get("category_match", False)])
        avg_similarity = np.mean([item.similarity_score for item in similar_items]) if similar_items else 0.0
        
        return {
            "success": True,
            "message": f"Found {len(similar_items)} similar items",
            "image_info": target_image_info,
            "similar_items": similar_items,
            "search_time": search_time,
            "statistics": {
                "total_found": len(similar_items),
                "category_matches": category_matches,
                "average_similarity": float(avg_similarity),
                "similarity_range": {
                    "min": float(min([item.similarity_score for item in similar_items])) if similar_items else 0.0,
                    "max": float(max([item.similarity_score for item in similar_items])) if similar_items else 0.0
                }
            },
            "filters_applied": {
                "category_filter": category_filter,
                "min_similarity": min_similarity,
                "limit": limit
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in enhanced search by ID {image_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@router.get("/random-image-id")
def get_random_image_id():
    """Get a random image ID from the database for testing purposes"""
    try:
        sql = """
        SELECT id, public_path, category
        FROM fashion_items 
        WHERE embedding IS NOT NULL
        ORDER BY RANDOM()
        LIMIT 1;
        """
        
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(sql)
        row = cur.fetchone()
        cur.close()
        conn.close()
        
        if row:
            image_id, public_path, category = row
            filename = public_path.split('/')[-1] if public_path else f"item_{image_id}.jpg"
            
            return {
                "id": image_id,
                "filename": filename,
                "category": category,
                "url": f"/images/{filename}",
                "message": "Random image selected for testing"
            }
        else:
            raise HTTPException(status_code=404, detail="No images found in database")
            
    except Exception as e:
        logger.error(f"Error getting random image ID: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get random image: {str(e)}")
