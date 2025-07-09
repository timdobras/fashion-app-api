import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import ORIGINS
from ml_models import load_models
from routers import health, images, search, admin

# Basic setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Fashion Image Search API",
    description="AI-powered fashion image similarity search and classification",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, tags=["health"])
app.include_router(images.router, tags=["images"])
app.include_router(search.router, tags=["search"])
app.include_router(admin.router, tags=["admin"])

# Model Loading on Startup
@app.on_event("startup")
async def startup_event():
    """Load ML models on startup"""
    await load_models()

@app.get("/")
def read_root():
    """Root endpoint"""
    logger.info("Root endpoint accessed")
    return {
        "message": "Fashion Image Search API",
        "version": "1.0.0",
        "status": "running"
    }
