import cv2
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from services.ExtractFeatureImage import (
    extract_feature_color,
    extract_feature_texture,
    extract_feature_venation,
    extract_all_features,
)


def main():
    # Load an image
    img_path = PROJECT_ROOT / "0003_0002_512.JPG"
    if not img_path.exists():
        candidates = list((PROJECT_ROOT / "data/Basil/healthy").glob("0008_0001.*"))
        if candidates:
            img_path = candidates[0]
        else:
            raise FileNotFoundError(f"No image found in {PROJECT_ROOT / 'data/Basil/healthy'}")

    img_np = cv2.imread(str(img_path))
    if img_np is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    print(f"Image shape: {img_np.shape}")
    print("-" * 60)

    # Extract individual features
    color_vec = extract_feature_color(img_np)
    texture_vec = extract_feature_texture(img_np)
    venation_vec = extract_feature_venation(img_np)

    print(f"Color Features (132D): shape={color_vec.shape}, dtype={color_vec.dtype}")
    print(f"  Values: {color_vec}")
    print()
    
    print(f"Texture Features (14D): shape={texture_vec.shape}, dtype={texture_vec.dtype}")
    print(f"  Values: {texture_vec}")
    print()
    
    print(f"Venation Features (3D): shape={venation_vec.shape}, dtype={venation_vec.dtype}")
    print(f"  Values: {venation_vec}")
    print()

    # Extract all features at once
    all_features = extract_all_features(img_np)
    total_dim = len(all_features["color"]) + len(all_features["texture"]) + len(all_features["venation"])
    print(f"Total feature dimension: {total_dim}D")
    print("-" * 60)


if __name__ == "__main__":
    main()