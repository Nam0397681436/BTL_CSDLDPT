import cv2
import numpy as np
from .feature.ColorFeature import extract_color_vector
from .feature.TextureFeature import extract_texture_vector
from .feature.VenationFeature import extract_venation_vector


def extract_feature_color(img_np: np.ndarray) -> np.ndarray:
    """Extract color histogram features (132D)."""
    if img_np is None or img_np.size == 0:
        return np.array([])
    
    try:
        if img_np.ndim == 2:
            img_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        elif img_np.shape[2] == 4:
            img_rgb = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
        else:
            img_rgb = img_np.copy()
        
        color_vector = extract_color_vector(img_rgb)
        return np.array(color_vector, dtype=np.float32)
    except Exception as e:
        print(f"Error extracting color features: {e}")
        return np.array([])


def extract_feature_texture(img_np: np.ndarray) -> np.ndarray:
    """Extract texture features - LBP and GLCM (14D)."""
    if img_np is None or img_np.size == 0:
        return np.array([])
    
    try:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY) if img_np.ndim == 3 else img_np.copy()
        texture_vector = extract_texture_vector(gray)
        return np.array(texture_vector, dtype=np.float32)
    except Exception as e:
        print(f"Error extracting texture features: {e}")
        return np.array([])


def extract_feature_venation(img_np: np.ndarray) -> np.ndarray:
    """Extract venation features - vein length, branch points, density (3D)."""
    if img_np is None or img_np.size == 0:
        return np.array([])
    
    try:
        if img_np.ndim == 2:
            img_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        elif img_np.shape[2] == 4:
            img_rgb = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
        else:
            img_rgb = img_np.copy()
        
        vein_vector = extract_venation_vector(img_rgb)
        return np.array(vein_vector, dtype=np.float32)
    except Exception as e:
        print(f"Error extracting venation features: {e}")
        return np.array([])

def extract_feature_HOG(img_np: np.ndarray) -> np.ndarray:
    pass


def extract_feature_SIFT(img_np: np.ndarray) -> np.ndarray:

    pass


def extract_all_features(img_np: np.ndarray) -> dict:
    """Extract all features: color (132D) + texture (14D) + venation (3D) = 149D total."""
    return {
        "color": extract_feature_color(img_np),
        "texture": extract_feature_texture(img_np),
        "venation": extract_feature_venation(img_np),
    }
