import tensorflow as tf
import numpy as np
from PIL import Image
import json
from create_dataset import load_and_preprocess_image

# Category mapping
CATEGORY_NAMES = [
    "short sleeve top",
    "long sleeve top", 
    "short sleeve outwear",
    "long sleeve outwear",
    "vest",
    "sling",
    "shorts",
    "trousers",
    "skirt",
    "short sleeve dress",
    "long sleeve dress",
    "vest dress",
    "sling dress"
]

class FashionClassifier:
    """Fashion category classifier for single images."""
    
    def __init__(self, model_path="fashion_category_classifier.keras"):
        """Initialize the classifier."""
        print("Loading fashion classification model...")
        try:
            self.model = tf.keras.models.load_model(model_path)
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model = None
    
    def predict_image(self, image_path, bbox=None, show_top_k=3):
        """
        Predict category for a single image.
        
        Args:
            image_path: Path to the image file
            bbox: Bounding box [x1, y1, x2, y2] or None for full image
            show_top_k: Number of top predictions to return
        
        Returns:
            List of prediction dictionaries
        """
        if self.model is None:
            print("❌ Model not loaded properly")
            return None
        
        try:
            # Handle full image if no bbox provided
            if bbox is None:
                # Load image to get dimensions
                image = tf.io.read_file(image_path)
                image = tf.image.decode_jpeg(image, channels=3)
                shape = tf.shape(image)
                # Use full image bbox [x1, y1, x2, y2]
                bbox = [0, 0, shape[1].numpy(), shape[0].numpy()]
                print(f"Using full image bbox: {bbox}")
            
            # Preprocess image using your existing function
            processed_image = load_and_preprocess_image(image_path, bbox, is_training=False)
            
            # Add batch dimension
            image_batch = tf.expand_dims(processed_image, 0)
            
            # Make prediction
            predictions = self.model(image_batch, training=False)
            probabilities = tf.nn.softmax(predictions[0]).numpy()
            
            # Get top-k predictions
            top_indices = np.argsort(probabilities)[::-1][:show_top_k]
            top_probs = probabilities[top_indices]
            
            results = []
            for i, (idx, prob) in enumerate(zip(top_indices, top_probs)):
                results.append({
                    'rank': i + 1,
                    'category': CATEGORY_NAMES[idx],
                    'confidence': float(prob),
                    'category_id': int(idx)
                })
            
            return results
            
        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            return None
    
    def predict_and_display(self, image_path, bbox=None):
        """Predict and display results in a nice format."""
        print(f"\n🔍 Analyzing image: {image_path}")
        if bbox:
            print(f"📦 Using bounding box: {bbox}")
        print("-" * 50)
        
        results = self.predict_image(image_path, bbox)
        
        if results:
            print("🎯 PREDICTION RESULTS:")
            for result in results:
                confidence_bar = "█" * int(result['confidence'] * 20)
                print(f"{result['rank']}. {result['category']:20} | "
                      f"{result['confidence']:.3f} | {confidence_bar}")
            
            # Interpretation
            top_prediction = results[0]
            if top_prediction['confidence'] > 0.8:
                confidence_level = "Very High"
                emoji = "🎯"
            elif top_prediction['confidence'] > 0.6:
                confidence_level = "High"
                emoji = "✅"
            elif top_prediction['confidence'] > 0.4:
                confidence_level = "Moderate"
                emoji = "⚠️"
            else:
                confidence_level = "Low"
                emoji = "❓"
            
            print(f"\n{emoji} Primary Prediction: {top_prediction['category']}")
            print(f"🔥 Confidence Level: {confidence_level} ({top_prediction['confidence']:.1%})")
            
            return results
        else:
            print("❌ Failed to generate predictions")
            return None

def test_multiple_images(image_paths, classifier):
    """Test multiple images and show summary."""
    print("\n🚀 BATCH IMAGE TESTING")
    print("=" * 60)
    
    all_results = []
    for i, image_path in enumerate(image_paths, 1):
        print(f"\n[{i}/{len(image_paths)}] Testing: {image_path}")
        results = classifier.predict_and_display(image_path)
        if results:
            all_results.append({
                'image': image_path,
                'predictions': results
            })
    
    # Summary
    if all_results:
        print(f"\n📊 BATCH TESTING SUMMARY:")
        print("-" * 40)
        
        categories_predicted = {}
        confidence_levels = []
        
        for result in all_results:
            top_pred = result['predictions'][0]
            category = top_pred['category']
            confidence = top_pred['confidence']
            
            categories_predicted[category] = categories_predicted.get(category, 0) + 1
            confidence_levels.append(confidence)
        
        print(f"Images processed: {len(all_results)}")
        print(f"Average confidence: {np.mean(confidence_levels):.3f}")
        print(f"Categories detected:")
        for category, count in categories_predicted.items():
            print(f"  - {category}: {count} image(s)")

def main():
    """Main testing function."""
    print("👗 Fashion Category Classifier - Image Testing")
    print("=" * 60)
    
    # Initialize classifier
    classifier = FashionClassifier()
    
    if classifier.model is None:
        return
    
    # Interactive testing mode
    while True:
        print(f"\n🎨 IMAGE TESTING OPTIONS:")
        print("1. Test single image")
        print("2. Test multiple images")
        print("3. Exit")
        
        choice = input("\nSelect option (1-3): ").strip()
        
        if choice == "1":
            image_path = input("Enter image path: ").strip()
            
            # Ask for bounding box (optional)
            use_bbox = input("Use bounding box? (y/n): ").strip().lower()
            bbox = None
            
            if use_bbox == 'y':
                try:
                    bbox_input = input("Enter bbox as x1,y1,x2,y2: ")
                    bbox = [int(x.strip()) for x in bbox_input.split(',')]
                    if len(bbox) != 4:
                        print("❌ Invalid bbox format, using full image")
                        bbox = None
                except:
                    print("❌ Invalid bbox format, using full image")
                    bbox = None
            
            classifier.predict_and_display(image_path, bbox)
        
        elif choice == "2":
            image_paths = []
            print("Enter image paths (one per line, empty line to finish):")
            while True:
                path = input().strip()
                if not path:
                    break
                image_paths.append(path)
            
            if image_paths:
                test_multiple_images(image_paths, classifier)
            else:
                print("No images provided")
        
        elif choice == "3":
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice")

if __name__ == "__main__":
    main()
