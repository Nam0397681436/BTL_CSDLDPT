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
    extract_feature_HOG,
    extract_feature_texture,
    extract_feature_venation,
    extract_feature_shape,
    extract_feature_HOG_ROTATE
)

def main():
    img_path="./test/img_test/img2.jpeg"
    img_input=cv2.imread(img_path)
    feature_hog=extract_feature_HOG(img_input)
    feature_hog_rotate= extract_feature_HOG_ROTATE(img_input)


    