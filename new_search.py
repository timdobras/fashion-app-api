import tensorflow as tf
import faiss
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import os
import argparse

# --- CONFIGURATION ---
class SearchConfig:
    MODEL_PATH = "robust_fashion_embedding_model.keras"
    INDEX_PATH = "faiss_index.index"
    IMAGE_PATHS_FILE = "image_paths.json"
    METADATA_FILE = "index_metadata.json"
    TARGET_SIZE = (224, 224)
    DEFAULT_NUM_RESULTS = 5

config = SearchConfig()

# --- CUSTOM TRIPLET LOSS (Required for model loading) ---

@tf.keras.utils.register_keras_serializable()
class TripletLoss(tf.keras.losses.Loss):
    """Custom triplet loss class for model loading."""
    
    def __init__(self, margin=0.3, name="triplet_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.margin = margin
    
    def call(self, y_true, y_pred):
        """Loss computation."""
        labels = tf.cast(y_true, tf.int32)
        embeddings = y_pred
        embeddings = tf.nn.l2_normalize(embeddings, axis=1)
        return self._compute_triplet_loss_vectorized(labels, embeddings)
    
    def _compute_triplet_loss_vectorized(self, labels, embeddings):
        """Compute triplet loss using vectorized operations."""
        pairwise_dist = self._pairwise_distances(embeddings)
        labels = tf.reshape(labels, [-1, 1])
        adjacency = tf.equal(labels, tf.transpose(labels))
        adjacency_not = tf.logical_not(adjacency)
        batch_size = tf.shape(embeddings)[0]
        
        mask_anchor_positive = tf.cast(adjacency, tf.float32) - tf.eye(tf.cast(batch_size, tf.float32))
        mask_anchor_negative = tf.cast(adjacency_not, tf.float32)
        
        anchor_positive_dist = pairwise_dist * mask_anchor_positive
        hardest_positive_dist = tf.reduce_max(anchor_positive_dist, axis=1, keepdims=True)
        
        max_anchor_negative_dist = tf.reduce_max(pairwise_dist, axis=1, keepdims=True)
        anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)
        hardest_negative_dist = tf.reduce_min(anchor_negative_dist, axis=1, keepdims=True)
        
        triplet_loss = tf.maximum(hardest_positive_dist - hardest_negative_dist + self.margin, 0.0)
        mask_positive_triplets = tf.cast(tf.reduce_sum(mask_anchor_positive, axis=1, keepdims=True) > 0, tf.float32)
        triplet_loss = triplet_loss * mask_positive_triplets
        
        valid_triplets = tf.reduce_sum(mask_positive_triplets)
        triplet_loss = tf.reduce_sum(triplet_loss) / tf.maximum(valid_triplets, 1.0)
        return triplet_loss
    
    def _pairwise_distances(self, embeddings):
        """Compute pairwise Euclidean distances."""
        dot_product = tf.matmul(embeddings, tf.transpose(embeddings))
        square_norm = tf.linalg.diag_part(dot_product)
        distances = (tf.expand_dims(square_norm, 1) - 2.0 * dot_product + tf.expand_dims(square_norm, 0))
        distances = tf.maximum(distances, 0.0)
        distances = tf.sqrt(distances + 1e-12)
        return distances
    
    def get_config(self):
        """Return the config for serialization."""
        config = super().get_config()
        config.update({"margin": self.margin})
        return config

# --- SEARCH SYSTEM CLASSES ---

class FashionSearchEngine:
    """Complete fashion image search engine for new images."""
    
    def __init__(self):
        self.model = None
        self.index = None
        self.image_paths = None
        self.metadata = None
        self.is_loaded = False
    
    def load_system(self):
        """Load the trained model, index, and metadata."""
        print("🔄 Loading fashion search system...")
        
        # Check required files
        required_files = [config.MODEL_PATH, config.INDEX_PATH, config.IMAGE_PATHS_FILE]
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            print(f"❌ Missing required files: {missing_files}")
            print("Please ensure you have run train_model.py and index_images.py first.")
            return False
        
        try:
            # Load model
            print("📱 Loading trained model...")
            custom_objects = {"TripletLoss": TripletLoss}
            self.model = tf.keras.models.load_model(config.MODEL_PATH, custom_objects=custom_objects)
            print("✅ Model loaded successfully")
            
            # Load Faiss index
            print("🔍 Loading Faiss index...")
            self.index = faiss.read_index(config.INDEX_PATH)
            print(f"✅ Index loaded: {self.index.ntotal} embeddings")
            
            # Load image paths
            print("📁 Loading image paths...")
            with open(config.IMAGE_PATHS_FILE, 'r') as f:
                self.image_paths = json.load(f)
            print(f"✅ Loaded {len(self.image_paths)} image paths")
            
            # Load metadata if available
            if os.path.exists(config.METADATA_FILE):
                with open(config.METADATA_FILE, 'r') as f:
                    self.metadata = json.load(f)
                print(f"✅ Loaded metadata: {self.metadata.get('index_type', 'Unknown')} index")
            
            self.is_loaded = True
            return True
            
        except Exception as e:
            print(f"❌ Error loading system: {e}")
            return False
    
    def preprocess_new_image(self, image_path, bbox=None):
        """Preprocess a new image for search."""
        try:
            # Load image
            image = tf.io.read_file(image_path)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.cast(image, tf.float32)
            
            if bbox is not None:
                # Crop using bounding box [x1, y1, x2, y2]
                y1, x1, y2, x2 = bbox[1], bbox[0], bbox[3], bbox[2]
                height, width = y2 - y1, x2 - x1
                
                # Ensure valid crop dimensions
                height = tf.maximum(height, 1)
                width = tf.maximum(width, 1)
                
                # Handle edge cases
                image_shape = tf.shape(image)
                y1 = tf.clip_by_value(y1, 0, image_shape[0] - 1)
                x1 = tf.clip_by_value(x1, 0, image_shape[1] - 1)
                height = tf.minimum(height, image_shape[0] - y1)
                width = tf.minimum(width, image_shape[1] - x1)
                
                image = tf.image.crop_to_bounding_box(image, y1, x1, height, width)
            
            # Resize to target size
            image = tf.image.resize(image, config.TARGET_SIZE, method='bilinear')
            
            # Ensure values are in valid range
            image = tf.clip_by_value(image, 0.0, 255.0)
            
            return image
            
        except Exception as e:
            print(f"⚠️  Error preprocessing image {image_path}: {e}")
            return None
    
    def generate_embedding(self, image_path, bbox=None):
        """Generate embedding for a new image."""
        if not self.is_loaded:
            print("❌ System not loaded. Call load_system() first.")
            return None
        
        # Preprocess image
        image = self.preprocess_new_image(image_path, bbox)
        if image is None:
            return None
        
        # Add batch dimension
        image_batch = tf.expand_dims(image, 0)
        
        # Generate embedding
        try:
            embedding = self.model.predict(image_batch, verbose=0)
            
            # Ensure L2 normalization
            embedding = tf.nn.l2_normalize(embedding, axis=1).numpy()
            
            return embedding
            
        except Exception as e:
            print(f"❌ Error generating embedding: {e}")
            return None
    
    def search_similar(self, image_path, bbox=None, num_results=5):
        """Search for similar fashion items using a new image."""
        print(f"🔍 Searching for similar items to: {Path(image_path).name}")
        
        # Generate embedding for query image
        query_embedding = self.generate_embedding(image_path, bbox)
        if query_embedding is None:
            return None, None
        
        # Perform search
        try:
            distances, indices = self.index.search(query_embedding, num_results)
            
            print(f"✅ Found {len(indices[0])} similar items")
            print(f"📏 Distance range: {distances[0].min():.4f} - {distances[0].max():.4f}")
            
            return distances[0], indices[0]
            
        except Exception as e:
            print(f"❌ Error during search: {e}")
            return None, None
    
    def get_result_paths(self, indices):
        """Get image paths for search result indices."""
        return [self.image_paths[idx] for idx in indices if idx < len(self.image_paths)]

class SearchVisualizer:
    """Handles visualization of search results."""
    
    @staticmethod
    def load_and_resize_image(image_path, target_size=(224, 224)):
        """Load and resize an image for display."""
        try:
            if not os.path.exists(image_path):
                print(f"⚠️  Image not found: {image_path}")
                return np.zeros((*target_size, 3), dtype=np.uint8)
            
            image = cv2.imread(image_path)
            if image is None:
                print(f"⚠️  Could not load image: {image_path}")
                return np.zeros((*target_size, 3), dtype=np.uint8)
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, target_size)
            return image
            
        except Exception as e:
            print(f"⚠️  Error loading {image_path}: {e}")
            return np.zeros((*target_size, 3), dtype=np.uint8)
    
    @staticmethod
    def display_search_results(query_image_path, distances, indices, result_paths, num_display=5):
        """Display query image and search results."""
        print(f"\n🖼️  Displaying search results for: {Path(query_image_path).name}")
        
        # Create figure
        fig_width = 4 * (num_display + 1)
        fig, axes = plt.subplots(1, num_display + 1, figsize=(fig_width, 4))
        
        # Load and display query image
        query_img = SearchVisualizer.load_and_resize_image(query_image_path)
        axes[0].imshow(query_img)
        axes[0].set_title("Query Image", fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Display result images
        for i in range(min(num_display, len(result_paths))):
            result_img_path = result_paths[i]
            result_img = SearchVisualizer.load_and_resize_image(result_img_path)
            
            axes[i + 1].imshow(result_img)
            axes[i + 1].set_title(f"Result {i + 1}\nDistance: {distances[i]:.3f}", fontsize=10)
            axes[i + 1].axis('off')
            
            print(f"   Result {i + 1}: {Path(result_img_path).name} (distance: {distances[i]:.4f})")
        
        # Hide unused subplots
        for i in range(len(result_paths) + 1, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        # Save and show results
        output_file = f"search_results_{Path(query_image_path).stem}.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"💾 Results saved to {output_file}")

# --- OBJECT DETECTION UTILITIES ---

class FashionItemDetector:
    """Simple fashion item detection utilities."""
    
    @staticmethod
    def detect_with_opencv(image_path):
        """Use OpenCV to detect the main object (simple center crop)."""
        try:
            image = cv2.imread(image_path)
            h, w = image.shape[:2]
            
            # Simple center crop (assumes fashion item is centered)
            margin = 0.1  # 10% margin
            x1 = int(w * margin)
            y1 = int(h * margin)
            x2 = int(w * (1 - margin))
            y2 = int(h * (1 - margin))
            
            return [x1, y1, x2, y2]
            
        except Exception as e:
            print(f"⚠️  Detection failed: {e}")
            return None
    
    @staticmethod
    def get_full_image_bbox(image_path):
        """Get bounding box for the entire image."""
        try:
            image = cv2.imread(image_path)
            h, w = image.shape[:2]
            return [0, 0, w, h]
        except Exception as e:
            print(f"⚠️  Could not get image dimensions: {e}")
            return None

# --- MAIN SEARCH INTERFACE ---

def search_fashion_item(query_image_path, bbox=None, num_results=5, show_visualization=True):
    """Main function to search for similar fashion items."""
    print("🚀 Starting Fashion Image Search")
    print("=" * 50)
    
    # Initialize search engine
    search_engine = FashionSearchEngine()
    
    # Load system
    if not search_engine.load_system():
        return None
    
    # Perform search
    distances, indices = search_engine.search_similar(
        query_image_path, bbox=bbox, num_results=num_results
    )
    
    if distances is None or indices is None:
        print("❌ Search failed")
        return None
    
    # Get result paths
    result_paths = search_engine.get_result_paths(indices)
    
    # Display results
    if show_visualization:
        SearchVisualizer.display_search_results(
            query_image_path, distances, indices, result_paths, num_results
        )
    
    # Return results
    results = {
        'query_image': query_image_path,
        'distances': distances.tolist(),
        'indices': indices.tolist(),
        'result_paths': result_paths
    }
    
    print("\n" + "=" * 50)
    print("🎉 Search Complete!")
    return results

def batch_search(query_images_folder, num_results=5):
    """Search multiple images in a folder."""
    print(f"🔍 Batch searching images in: {query_images_folder}")
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(query_images_folder).glob(f"*{ext}"))
        image_files.extend(Path(query_images_folder).glob(f"*{ext.upper()}"))
    
    if not image_files:
        print("❌ No image files found in the specified folder")
        return []
    
    print(f"📁 Found {len(image_files)} images to search")
    
    # Initialize search engine once
    search_engine = FashionSearchEngine()
    if not search_engine.load_system():
        return []
    
    all_results = []
    
    for image_file in image_files:
        print(f"\n🔍 Searching: {image_file.name}")
        
        distances, indices = search_engine.search_similar(
            str(image_file), num_results=num_results
        )
        
        if distances is not None and indices is not None:
            result_paths = search_engine.get_result_paths(indices)
            
            result = {
                'query_image': str(image_file),
                'distances': distances.tolist(),
                'indices': indices.tolist(),
                'result_paths': result_paths
            }
            all_results.append(result)
        else:
            print(f"❌ Failed to search {image_file.name}")
    
    print(f"\n✅ Completed batch search: {len(all_results)} successful searches")
    return all_results

# --- COMMAND LINE INTERFACE ---

def main():
    """Command line interface for the search engine."""
    parser = argparse.ArgumentParser(description="Fashion Image Search Engine")
    parser.add_argument("query_image", help="Path to the query image")
    parser.add_argument("--num_results", "-n", type=int, default=5, 
                       help="Number of similar items to return (default: 5)")
    parser.add_argument("--bbox", nargs=4, type=int, metavar=('X1', 'Y1', 'X2', 'Y2'),
                       help="Bounding box coordinates [x1 y1 x2 y2]")
    parser.add_argument("--no_visualization", action="store_true",
                       help="Skip visualization of results")
    parser.add_argument("--detect_center", action="store_true",
                       help="Use center crop detection")
    parser.add_argument("--batch", action="store_true",
                       help="Treat query_image as a folder for batch processing")
    
    args = parser.parse_args()
    
    # Determine bounding box
    bbox = None
    if args.bbox:
        bbox = args.bbox
    elif args.detect_center:
        bbox = FashionItemDetector.detect_with_opencv(args.query_image)
        print(f"🎯 Detected center crop: {bbox}")
    
    # Perform search
    if args.batch:
        results = batch_search(args.query_image, args.num_results)
        
        # Save batch results
        output_file = "batch_search_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Batch results saved to {output_file}")
        
    else:
        results = search_fashion_item(
            args.query_image, 
            bbox=bbox, 
            num_results=args.num_results,
            show_visualization=not args.no_visualization
        )
        
        if results:
            # Save individual results
            output_file = f"search_results_{Path(args.query_image).stem}.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"💾 Results saved to {output_file}")

if __name__ == "__main__":
    # For testing, you can also call the functions directly
    if len(os.sys.argv) == 1:
        # Interactive mode - example usage
        print("🎯 Fashion Search Engine - Interactive Mode")
        print("=" * 50)
        
        # Example search
        example_image = input("Enter path to your fashion image (or 'quit' to exit): ")
        
        if example_image.lower() != 'quit' and os.path.exists(example_image):
            results = search_fashion_item(example_image, num_results=5)
            
            if results:
                print("\n✅ Search completed successfully!")
