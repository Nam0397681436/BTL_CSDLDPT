import cv2
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from services.ExtractFeatureImage import (
    extract_feature_HOG,
    extract_feature_SIFT,
    extract_feature_color_histogram,
    extract_feature_texture,
)


def main():
    # Load an image
    img_path = PROJECT_ROOT / "data/Basil/healthy/0008_0001.JPG"
    if not img_path.exists():
        candidates = list((PROJECT_ROOT / "data/Basil/healthy").glob("0008_0001.*"))
        if candidates:
            img_path = candidates[0]

    img_np = cv2.imread(str(img_path))
    img_np = cv2.resize(img_np, (128, 128))  # Resize for consistent feature extraction
    if img_np is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    hog_features = extract_feature_HOG(img_np)
    print("Color Histogram Features:", extract_feature_color_histogram(img_np, output_dir="./output"))
    

    # Print the HOG features and their shape
    # print("HOG Features:", hog_features)
    # print("HOG Features Shape:", hog_features.shape)
    # print("length of HOG Features:", len(hog_features))

if __name__ == "__main__":
    main()