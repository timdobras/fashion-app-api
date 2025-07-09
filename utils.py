import logging
import numpy as np
import tensorflow as tf
from PIL import Image

logger = logging.getLogger(__name__)

def test_tensorflow():
    """Test TensorFlow import and basic functionality"""
    try:
        x = tf.constant([1, 2, 3, 4])
        y = tf.reduce_sum(x)
        return True
    except Exception as e:
        logger.error(f"TensorFlow test failed: {e}")
        return False

def test_pil():
    """Test PIL/Pillow functionality"""
    try:
        test_image = Image.new('RGB', (100, 100), color='red')
        test_array = np.array(test_image)
        return True
    except Exception as e:
        logger.error(f"PIL test failed: {e}")
        return False

def test_numpy():
    """Test NumPy functionality"""
    try:
        arr = np.array([1, 2, 3, 4, 5])
        result = np.mean(arr)
        return True
    except Exception as e:
        logger.error(f"NumPy test failed: {e}")
        return False
