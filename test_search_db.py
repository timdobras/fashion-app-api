import tensorflow as tf
import numpy as np
import psycopg2
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import os
import argparse

# ─── CONFIG ──────────────────────────────────────────────────────────────
class SearchConfig:
    MODEL_PATH = "robust_fashion_embedding_model.keras"
    TARGET_SIZE = (224, 224)
    DEFAULT_NUM_RESULTS = 5

config = SearchConfig()

# ─── DATABASE ────────────────────────────────────────────────────────────
DB_PARAMS = {
    "dbname":   "fashion_db",
    "user":     "fashion_user",
    "password": "FASHION",
    "host":     "192.168.1.17",   # your Postgres VM IP
    "port":     5432
}

# ─── PATH MAPPING ─────────────────────────────────────────────────────────
# When we stored to DB, we replaced LOCAL_BASE → DB_BASE.
# Now we reverse that so the script can load the actual files.
LOCAL_BASE = "/home/gumeq/TrueNAS"
DB_BASE    = "/mnt/truenas"

def map_db_path_to_local(db_path: str) -> str:
    """
    Convert a DB-returned path (/mnt/truenas/...) 
    back to your local TrueNAS mount (/home/gumeq/TrueNAS/...).
    """
    if db_path.startswith(DB_BASE):
        return db_path.replace(DB_BASE, LOCAL_BASE, 1)
    return db_path

# ─── CUSTOM LOSS (needed to load your model) ─────────────────────────────
@tf.keras.utils.register_keras_serializable()
class TripletLoss(tf.keras.losses.Loss):
    def __init__(self, margin=0.3, name="triplet_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.margin = margin

    def call(self, y_true, y_pred):
        # We only need the architecture for loading; the loss value is unused here.
        return tf.constant(0.0)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"margin": self.margin})
        return cfg

# ─── SEARCH ENGINE ────────────────────────────────────────────────────────
class FashionSearchEngine:
    def __init__(self):
        self.model = None
        self.is_loaded = False

    def load_system(self):
        print("🔄 Loading fashion search system…")
        if not os.path.exists(config.MODEL_PATH):
            print(f"❌ Missing model file: {config.MODEL_PATH}")
            return False
        try:
            print("📱 Loading trained model…")
            self.model = tf.keras.models.load_model(
                config.MODEL_PATH,
                custom_objects={"TripletLoss": TripletLoss}
            )
            print("✅ Model loaded successfully")
            self.is_loaded = True
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

    def preprocess_new_image(self, image_path, bbox=None):
        try:
            img = tf.io.read_file(image_path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.cast(img, tf.float32)
            if bbox:
                y1, x1, y2, x2 = bbox[1], bbox[0], bbox[3], bbox[2]
                h = y2 - y1; w = x2 - x1
                img = tf.image.crop_to_bounding_box(img, y1, x1, h, w)
            img = tf.image.resize(img, config.TARGET_SIZE)
            return img
        except Exception as e:
            print(f"⚠️ Error preprocessing {image_path}: {e}")
            return None

    def generate_embedding(self, image_path, bbox=None):
        if not self.is_loaded:
            print("❌ System not loaded. Call load_system() first.")
            return None
        img = self.preprocess_new_image(image_path, bbox)
        if img is None:
            return None
        batch = tf.expand_dims(img, 0)
        emb = self.model.predict(batch, verbose=0)
        emb = tf.nn.l2_normalize(emb, axis=1).numpy().flatten()
        return emb

    def search_similar(self, image_path, bbox=None, num_results=None):
        num_results = num_results or config.DEFAULT_NUM_RESULTS
        print(f"🔍 Searching for similar items to: {Path(image_path).name}")
        qemb = self.generate_embedding(image_path, bbox)
        if qemb is None:
            return [], []
        # Build a Postgres vector literal
        emb_str = "[" + ",".join(f"{v:.6f}" for v in qemb.tolist()) + "]"
        sql = """
            SELECT public_path,
                   embedding <-> %s::vector AS distance
            FROM fashion_items
            ORDER BY distance
            LIMIT %s;
        """
        try:
            conn = psycopg2.connect(**DB_PARAMS)
            cur  = conn.cursor()
            cur.execute(sql, (emb_str, num_results))
            rows = cur.fetchall()
            cur.close()
            conn.close()
            if not rows:
                print("⚠️ No results found.")
                return [], []
            paths, distances = zip(*rows)
            print(f"✅ Found {len(paths)} similar items")
            print(f"📏 Distance range: {min(distances):.4f} - {max(distances):.4f}")
            return list(distances), list(paths)
        except Exception as e:
            print(f"❌ DB search error: {e}")
            return [], []

# ─── VISUALIZATION ─────────────────────────────────────────────────────────
class SearchVisualizer:
    @staticmethod
    def load_and_resize_image(path, target_size=(224,224)):
        # Remap the DB path back to your local filesystem
        local_path = map_db_path_to_local(path)
        if not os.path.exists(local_path):
            print(f"⚠️ Image not found: {local_path}")
            return np.zeros((*target_size, 3), dtype=np.uint8)
        img = cv2.imread(local_path)
        if img is None:
            print(f"⚠️ Could not load image: {local_path}")
            return np.zeros((*target_size, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return cv2.resize(img, target_size)

    @staticmethod
    def display_search_results(query_path, distances, paths, num_display=5):
            """
            Display and save the search results.
            Saves a PNG named `search_results_<queryname>.png`.
            """
            # Remap the query path if needed
            query_local = map_db_path_to_local(query_path)
            
            # Build the figure
            fig, axes = plt.subplots(1, num_display+1, figsize=(4*(num_display+1), 4))
            
            # Query image
            qimg = SearchVisualizer.load_and_resize_image(query_local)
            axes[0].imshow(qimg)
            axes[0].set_title("Query")
            axes[0].axis('off')
            
            # Result images
            for i, (p, d) in enumerate(zip(paths, distances)):
                img = SearchVisualizer.load_and_resize_image(p)
                axes[i+1].imshow(img)
                axes[i+1].set_title(f"{i+1}. {d:.3f}")
                axes[i+1].axis('off')
            
            plt.tight_layout()
            
            # Save to file instead of plt.show()
            output_file = f"search_results_{Path(query_local).stem}.png"
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"💾 Results saved to {output_file}")

# ─── ENTRYPOINT ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Fashion Search Engine")
    parser.add_argument("query_image", help="Local path to the query image")
    parser.add_argument("-n", "--num_results", type=int, default=5,
                        help="How many similar items to return")
    parser.add_argument("--bbox", nargs=4, type=int, metavar=('X1','Y1','X2','Y2'),
                        help="Optional bounding box [x1 y1 x2 y2]")
    args = parser.parse_args()

    bbox = args.bbox if args.bbox else None

    engine = FashionSearchEngine()
    if not engine.load_system():
        exit(1)

    distances, paths = engine.search_similar(
        args.query_image, bbox=bbox, num_results=args.num_results
    )
    if distances and paths:
        SearchVisualizer.display_search_results(
            args.query_image, distances, paths, args.num_results
        )