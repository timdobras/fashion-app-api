import logging
import psycopg2
import numpy as np
from typing import List
from datetime import datetime
from fastapi import HTTPException

from config import DB_PARAMS
from models import FashionItem

logger = logging.getLogger(__name__)

def get_db_connection():
    """Get database connection"""
    try:
        return psycopg2.connect(**DB_PARAMS)
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        raise HTTPException(status_code=503, detail="Database connection failed")

def test_database_connection():
    """Test database connectivity"""
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.close()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Database test failed: {e}")
        return False

async def save_image_to_database(filename: str, file_path: str, embedding: List[float], category: str):
    """Save image metadata and embedding to database with proper column mapping"""
    try:
        # Build vector literal for PostgreSQL
        emb_str = "[" + ",".join(f"{v:.6f}" for v in embedding) + "]"
        
        # Updated SQL to handle the correct column structure
        sql = """
        INSERT INTO fashion_items (public_path, image_path, embedding, category, created_at)
        VALUES (%s, %s, %s::vector, %s, %s)
        RETURNING id;
        """
        
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Use file_path for both public_path and image_path if they're separate columns
        cur.execute(sql, (file_path, file_path, emb_str, category, datetime.utcnow()))
        
        image_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()
        
        return str(image_id)
        
    except psycopg2.IntegrityError as e:
        logger.error(f"Database integrity error: {e}")
        raise HTTPException(status_code=400, detail="Database constraint violation")
        
    except psycopg2.OperationalError as e:
        logger.error(f"Database connection error: {e}")
        raise HTTPException(status_code=503, detail="Database connection failed")
        
    except Exception as e:
        logger.error(f"Database save error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save to database: {str(e)}")

async def find_similar_fashion_items(embedding: List[float], limit: int = 10):
    """Find similar items using vector similarity"""
    try:
        # Build vector literal for PostgreSQL
        emb_str = "[" + ",".join(f"{v:.6f}" for v in embedding) + "]"
        
        # Updated SQL query to match your existing table structure
        sql = """
        SELECT id, public_path, category, embedding <-> %s::vector AS distance
        FROM fashion_items 
        ORDER BY distance 
        LIMIT %s;
        """
        
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(sql, (emb_str, limit))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        
        similar_items = []
        for i, (item_id, path, category, distance) in enumerate(rows):
            # Extract filename from the public_path
            if path:
                # Get just the filename from the full path
                image_filename = path.split('/')[-1]
            else:
                image_filename = f"item_{item_id}.jpg"
            
            similar_items.append(FashionItem(
                id=str(item_id),
                image_url=f"/images/{image_filename}",
                category=category or "unknown",
                similarity_score=max(0.0, 1.0 - float(distance)),
                metadata={
                    "distance": float(distance),
                    "rank": i + 1,
                    "original_path": path
                }
            ))
        
        return similar_items
    except Exception as e:
        logger.error(f"Similarity search error: {e}")
        return []

def determine_image_type(path: str) -> str:
    """Determine image type based on the database path"""
    if not path:
        return "unknown"
    
    path_lower = path.lower()
    
    # Check for uploaded images (local paths)
    if "/home/gumeq/" in path_lower or "uploads" in path_lower:
        return "uploaded"
    
    # Check for training images
    elif "/train/" in path_lower or "train" in path_lower:
        return "training"
    
    # Check for validation images
    elif "/validation/" in path_lower or "validation" in path_lower:
        return "validation"
    
    # Default fallback
    else:
        return "unknown"
