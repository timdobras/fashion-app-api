import logging
import numpy as np
import tensorflow as tf
from typing import List, Tuple
from fastapi import HTTPException
from PIL import Image

from config import EMBEDDING_MODEL_PATH, CLASSIFICATION_MODEL_PATH, CATEGORY_NAMES

logger = logging.getLogger(__name__)

# Global variables for models
embedding_model = None
classification_model = None

@tf.keras.utils.register_keras_serializable()
class TripletLoss(tf.keras.losses.Loss):
    def __init__(self, margin=0.3, name="triplet_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.margin = margin
    
    def call(self, y_true, y_pred):
        return tf.constant(0.0)
    
    def get_config(self):
        cfg = super().get_config()
        cfg.update({"margin": self.margin})
        return cfg

async def load_models():
    """Load ML models on startup"""
    global embedding_model, classification_model
    
    print("🔄 Loading ML models...")
    try:
        # Load embedding model
        print("📱 Loading embedding model...")
        embedding_model = tf.keras.models.load_model(
            EMBEDDING_MODEL_PATH,
            custom_objects={"TripletLoss": TripletLoss}
        )
        print("✅ Embedding model loaded successfully")
        
        # Load classification model
        print("📱 Loading classification model...")
        classification_model = tf.keras.models.load_model(
            CLASSIFICATION_MODEL_PATH
        )
        print("✅ Classification model loaded successfully")
        
        print("🎉 All models loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        print("⚠️ API will run with limited functionality")

async def generate_image_embedding(image: Image.Image) -> List[float]:
    """Generate embedding using your trained model"""
    if embedding_model is None:
        raise HTTPException(status_code=503, detail="Embedding model not loaded")
    
    try:
        # Preprocess image (resize to 224x224 as in your code)
        img_array = np.array(image.resize((224, 224)))
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
        img_batch = tf.expand_dims(img_tensor, 0)
        
        # Generate embedding
        embedding = embedding_model.predict(img_batch, verbose=0)
        embedding = tf.nn.l2_normalize(embedding, axis=1).numpy().flatten()
        return embedding.tolist()
    except Exception as e:
        logger.error(f"Embedding generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")

async def classify_image(image: Image.Image) -> Tuple[str, float, List[dict]]:
    """Classify image using your trained model"""
    if classification_model is None:
        raise HTTPException(status_code=503, detail="Classification model not loaded")
    
    try:
        # Preprocess image
        img_array = np.array(image.resize((224, 224)))
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
        img_batch = tf.expand_dims(img_tensor, 0)
        
        # Get predictions
        predictions = classification_model.predict(img_batch, verbose=0)
        probabilities = tf.nn.softmax(predictions[0]).numpy()
        
        # Get top prediction
        predicted_class = np.argmax(probabilities)
        confidence = float(probabilities[predicted_class])
        predicted_category = CATEGORY_NAMES[predicted_class]
        
        # Get all predictions for detailed response
        all_predictions = []
        for i, prob in enumerate(probabilities):
            all_predictions.append({
                "category": CATEGORY_NAMES[i],
                "confidence": float(prob)
            })
        
        # Sort by confidence
        all_predictions.sort(key=lambda x: x["confidence"], reverse=True)
        
        return predicted_category, confidence, all_predictions
    except Exception as e:
        logger.error(f"Classification error: {e}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

def get_models_status():
    """Get status of loaded models"""
    return {
        "embedding_model": embedding_model is not None,
        "classification_model": classification_model is not None
    }
