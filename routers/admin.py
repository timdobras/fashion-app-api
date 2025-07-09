import os
import logging
from datetime import datetime
from typing import List
from fastapi import APIRouter, HTTPException

from models import DeleteSummary
from database import get_db_connection
from config import UPLOAD_DIR

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/discover-images")
def discover_images():
    """Discover random images from each category"""
    try:
        sql = """
        WITH categorized_images AS (
            SELECT
                id,
                public_path,
                category,
                CASE
                    WHEN public_path LIKE '%/validation/%' THEN 'validation'
                    WHEN public_path LIKE '%/train/%' THEN 'training'
                    WHEN public_path LIKE '%uploads/%' THEN 'uploaded'
                    ELSE 'unknown'
                END AS image_type,
                ROW_NUMBER() OVER (
                    PARTITION BY (
                        CASE
                            WHEN public_path LIKE '%/validation/%' THEN 'validation'
                            WHEN public_path LIKE '%/train/%' THEN 'training'
                            WHEN public_path LIKE '%uploads/%' THEN 'uploaded'
                            ELSE 'unknown'
                        END
                    )
                    ORDER BY RANDOM()
                ) AS rn
            FROM fashion_items
            WHERE public_path IS NOT NULL
        )
        SELECT id, public_path, category, image_type
        FROM categorized_images
        WHERE rn <= 10 AND image_type != 'unknown'
        ORDER BY image_type, category, id;
        """
        
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        cur.close()
        conn.close()
        
        all_images = []
        category_counts = {}
        
        for item_id, public_path, category, image_type in rows:
            if not public_path:
                continue
                
            filename = public_path.split('/')[-1] if public_path else f"item_{item_id}.jpg"
            
            if category:
                category_counts[category] = category_counts.get(category, 0) + 1
            
            image_info = {
                "id": str(item_id),
                "filename": filename,
                "type": image_type,
                "url": f"/images/{filename}",
                "category": category,
                "database_path": public_path
            }
            
            all_images.append(image_info)
        
        breakdown = {
            "validation": len([img for img in all_images if img['type'] == 'validation']),
            "training": len([img for img in all_images if img['type'] == 'training']),
            "uploaded": len([img for img in all_images if img['type'] == 'uploaded'])
        }
        
        stats = {
            "total_sampled": len(all_images),
            "max_per_category": 20,
            "categories_found": len(category_counts)
        }
        
        logger.info(f"📊 Sampled {len(all_images)} images: {breakdown}")
        
        return {
            "total_images": len(all_images),
            "images": all_images,
            "source": "database_sample",
            "breakdown": breakdown,
            "statistics": stats,
            "categories": category_counts,
            "last_updated": datetime.utcnow().isoformat(),
            "sample_size": 20
        }
        
    except Exception as e:
        logger.error(f"Database sampling error: {e}")
        return {
            "total_images": 0,
            "images": [],
            "source": "database_error",
            "breakdown": {"validation": 0, "training": 0, "uploaded": 0},
            "error": str(e),
            "last_updated": datetime.utcnow().isoformat()
        }

@router.get("/database-stats")
def get_database_stats():
    """Get comprehensive statistics from the entire database"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Get total counts by image type
        sql_stats = """
        SELECT 
            CASE
                WHEN public_path LIKE '%/validation/%' THEN 'validation'
                WHEN public_path LIKE '%/train/%' THEN 'training'
                WHEN public_path LIKE '%/uploads%' THEN 'uploaded'
                ELSE 'unknown'
            END AS image_type,
            COUNT(*) as count,
            COUNT(CASE WHEN embedding IS NOT NULL THEN 1 END) as with_embeddings,
            COUNT(CASE WHEN category IS NOT NULL THEN 1 END) as with_categories
        FROM fashion_items
        WHERE public_path IS NOT NULL
        GROUP BY image_type;
        """
        
        cur.execute(sql_stats)
        type_stats = cur.fetchall()
        
        # Get overall statistics
        sql_overall = """
        SELECT 
            COUNT(*) as total_items,
            COUNT(CASE WHEN embedding IS NOT NULL THEN 1 END) as total_with_embeddings,
            COUNT(CASE WHEN category IS NOT NULL THEN 1 END) as total_with_categories,
            COUNT(DISTINCT category) as unique_categories
        FROM fashion_items
        WHERE public_path IS NOT NULL;
        """
        
        cur.execute(sql_overall)
        overall_stats = cur.fetchone()
        
        # Get category breakdown
        sql_categories = """
        SELECT category, COUNT(*) as count
        FROM fashion_items
        WHERE category IS NOT NULL
        GROUP BY category
        ORDER BY count DESC;
        """
        
        cur.execute(sql_categories)
        category_breakdown = cur.fetchall()
        
        cur.close()
        conn.close()
        
        # Process type statistics
        type_counts = {"validation": 0, "training": 0, "uploaded": 0, "unknown": 0}
        type_embeddings = {"validation": 0, "training": 0, "uploaded": 0, "unknown": 0}
        type_categories = {"validation": 0, "training": 0, "uploaded": 0, "unknown": 0}
        
        for image_type, count, with_embeddings, with_categories in type_stats:
            type_counts[image_type] = count
            type_embeddings[image_type] = with_embeddings
            type_categories[image_type] = with_categories
        
        categories_dict = {category: count for category, count in category_breakdown}
        
        return {
            "total_statistics": {
                "total_items": overall_stats[0],
                "total_with_embeddings": overall_stats[1],
                "total_with_categories": overall_stats[2],
                "unique_categories": overall_stats[3]
            },
            "type_breakdown": {
                "validation": {
                    "total": type_counts["validation"],
                    "with_embeddings": type_embeddings["validation"],
                    "with_categories": type_categories["validation"]
                },
                "training": {
                    "total": type_counts["training"],
                    "with_embeddings": type_embeddings["training"],
                    "with_categories": type_categories["training"]
                },
                "uploaded": {
                    "total": type_counts["uploaded"],
                    "with_embeddings": type_embeddings["uploaded"],
                    "with_categories": type_categories["uploaded"]
                }
            },
            "category_breakdown": categories_dict,
            "last_updated": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Database statistics error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get database statistics: {str(e)}")

@router.get("/test-database")
def test_database():
    """Test database connection and table structure"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Test basic connectivity
        cur.execute("SELECT version();")
        version = cur.fetchone()[0]
        
        # Check if fashion_items table exists
        cur.execute("""
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_name = 'fashion_items'
        ORDER BY ordinal_position;
        """)
        columns = cur.fetchall()
        
        # Check if vector extension is available
        cur.execute("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector');")
        vector_available = cur.fetchone()[0]
        
        # Test a simple insert (then rollback)
        cur.execute("BEGIN;")
        test_embedding = "[" + ",".join(["0.1"] * 128) + "]"
        cur.execute("""
        INSERT INTO fashion_items (public_path, embedding, category, created_at)
        VALUES (%s, %s::vector, %s, %s)
        RETURNING id;
        """, ("/uploads/IMAGE.png", test_embedding, "test", datetime.utcnow()))
        
        test_id = cur.fetchone()[0]
        cur.execute("ROLLBACK;")
        
        cur.close()
        conn.close()
        
        return {
            "database_connected": True,
            "postgresql_version": version,
            "vector_extension": vector_available,
            "table_columns": [{"name": col[0], "type": col[1]} for col in columns],
            "test_insert": f"Success - would have created ID {test_id}",
            "message": "Database is working correctly"
        }
        
    except Exception as e:
        logger.error(f"Database test failed: {e}")
        return {
            "database_connected": False,
            "error": str(e),
            "message": "Database test failed"
        }

@router.delete("/uploaded-files")
async def delete_uploaded_files(
    confirm: bool = False,
    dry_run: bool = True,
    safety_keyword: str = ""
):
    """Safely delete ONLY uploaded files from database and filesystem"""
    start_time = datetime.utcnow()
    
    # Safety checks
    if not dry_run and safety_keyword != "DELETE_UPLOADS_ONLY":
        raise HTTPException(
            status_code=400, 
            detail="Safety keyword required. Use safety_keyword='DELETE_UPLOADS_ONLY'"
        )
    
    if not dry_run and not confirm:
        raise HTTPException(
            status_code=400,
            detail="Must set confirm=True to actually delete files"
        )
    
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Only select uploaded files with multiple path validations
        sql_find_uploads = """
        SELECT id, public_path, category, created_at
        FROM fashion_items 
        WHERE public_path IS NOT NULL
          AND (
              public_path LIKE '%uploads%' 
              OR public_path LIKE '%/home/gumeq/python_api_server/uploads%'
          )
          AND public_path NOT LIKE '%validation%'
          AND public_path NOT LIKE '%train%'
          AND public_path NOT LIKE '%/mnt/truenas%'
        ORDER BY id;
        """
        
        cur.execute(sql_find_uploads)
        upload_records = cur.fetchall()
        
        logger.info(f"Found {len(upload_records)} uploaded files for {'DRY RUN' if dry_run else 'DELETION'}")
        
        deleted_files = []
        failed_deletions = []
        deleted_db_records = []
        
        for record_id, file_path, category, created_at in upload_records:
            try:
                # Safety checks
                if not file_path or "uploads" not in file_path:
                    logger.warning(f"Skipping suspicious path: {file_path}")
                    continue
                
                if not file_path.startswith("/home/gumeq/python_api_server/uploads"):
                    logger.warning(f"Skipping non-uploads path: {file_path}")
                    continue
                
                filename = file_path.split('/')[-1]
                
                if dry_run:
                    logger.info(f"DRY RUN: Would delete {filename} (ID: {record_id})")
                    deleted_files.append(filename)
                    deleted_db_records.append(str(record_id))
                else:
                    # Delete file from filesystem
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        logger.info(f"Deleted file: {file_path}")
                    else:
                        logger.warning(f"File not found: {file_path}")
                    
                    deleted_files.append(filename)
                    deleted_db_records.append(str(record_id))
                    
            except Exception as e:
                logger.error(f"Failed to delete file {file_path}: {e}")
                failed_deletions.append({
                    "id": str(record_id),
                    "path": file_path,
                    "error": str(e)
                })
        
        # Delete database records
        if not dry_run and upload_records:
            upload_ids = [str(record[0]) for record in upload_records]
            placeholders = ','.join(['%s'] * len(upload_ids))
            
            sql_delete = f"""
            DELETE FROM fashion_items 
            WHERE id IN ({placeholders})
              AND public_path IS NOT NULL
              AND (
                  public_path LIKE '%uploads%' 
                  OR public_path LIKE '%/home/gumeq/python_api_server/uploads%'
              )
              AND public_path NOT LIKE '%validation%'
              AND public_path NOT LIKE '%train%'
              AND public_path NOT LIKE '%/mnt/truenas%'
            """
            
            cur.execute(sql_delete, upload_ids)
            deleted_count = cur.rowcount
            conn.commit()
            
            logger.info(f"Deleted {deleted_count} database records")
        else:
            deleted_count = len(upload_records) if dry_run else 0
        
        cur.close()
        conn.close()
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        safety_checks = all([
            safety_keyword == "DELETE_UPLOADS_ONLY" or dry_run,
            confirm or dry_run,
            all("uploads" in record[1] for record in upload_records if record[1])
        ])
        
        return DeleteSummary(
            success=True,
            message=f"{'DRY RUN: Would delete' if dry_run else 'Successfully deleted'} {deleted_count} uploaded files",
            deleted_count=deleted_count,
            failed_deletions=failed_deletions,
            files_deleted=deleted_files,
            database_records_deleted=deleted_db_records,
            safety_checks_passed=safety_checks,
            dry_run=dry_run,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error during upload deletion: {e}")
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")

@router.get("/uploaded-files/preview")
async def preview_uploaded_files():
    """Preview what files would be deleted"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        sql_preview = """
        SELECT id, public_path, category, created_at
        FROM fashion_items 
        WHERE public_path IS NOT NULL
          AND (
              public_path LIKE '%uploads%' 
              OR public_path LIKE '%/home/gumeq/python_api_server/uploads%'
          )
          AND public_path NOT LIKE '%validation%'
          AND public_path NOT LIKE '%train%'
          AND public_path NOT LIKE '%/mnt/truenas%'
        ORDER BY created_at DESC;
        """
        
        cur.execute(sql_preview)
        records = cur.fetchall()
        cur.close()
        conn.close()
        
        preview_items = []
        for record_id, file_path, category, created_at in records:
            filename = file_path.split('/')[-1] if file_path else f"item_{record_id}"
            file_exists = os.path.exists(file_path) if file_path else False
            
            preview_items.append({
                "id": str(record_id),
                "filename": filename,
                "path": file_path,
                "category": category,
                "created_at": created_at.isoformat() if created_at else None,
                "file_exists": file_exists
            })
        
        return {
            "total_uploaded_files": len(preview_items),
            "files": preview_items,
            "warning": "These are the uploaded files that would be deleted",
            "safety_note": "Validation and training data is protected and will NOT be affected"
        }
        
    except Exception as e:
        logger.error(f"Error previewing uploads: {e}")
        raise HTTPException(status_code=500, detail=f"Preview failed: {str(e)}")
