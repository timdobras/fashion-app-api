import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database configuration
DB_PARAMS = {
    "dbname": os.getenv("DB_NAME", "fashion_db"),
    "user": os.getenv("DB_USER", "fashion_user"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432"))
}

# Validate required environment variables
required_env_vars = ["DB_PASSWORD"]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

# CORS configuration
CORS_ORIGINS_STR = os.getenv("CORS_ORIGINS", "http://localhost:3000")
ORIGINS = [origin.strip() for origin in CORS_ORIGINS_STR.split(",")]

# Directory configuration
IMAGE_DIR = os.getenv("IMAGE_DIR", "/mnt/truenas/fashion-app-ml/data/validation/image")
TRAIN_IMAGE_DIR = os.getenv("TRAIN_IMAGE_DIR", "/mnt/truenas/fashion-app-ml/data/train/image")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/home/gumeq/python_api_server/uploads")

# Create directories
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Model file paths
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", "robust_fashion_embedding_model.keras")
CLASSIFICATION_MODEL_PATH = os.getenv("CLASSIFICATION_MODEL_PATH", "fashion_category_classifier.keras")

# API configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# Category names
CATEGORY_NAMES = [
    "short sleeve top", "long sleeve top", "short sleeve outwear",
    "long sleeve outwear", "vest", "sling", "shorts", "trousers", 
    "skirt", "short sleeve dress", "long sleeve dress", 
    "vest dress", "sling dress"
]
