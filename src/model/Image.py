import numpy as np
import cv2

from services.ExtractFeatureImage import (
    extract_feature_color_histogram,
    extract_feature_texture,
)


class Image:

    url_minio=None 

    def __init__(self, path):
        self.source_path = path
        self.image = cv2.imread(path)
        self.url_minio = None

        if self.image is None:
            raise FileNotFoundError(f"Cannot read image: {path}")

    def preprocess(self):
        self.image = cv2.resize(self.image, (256, 256), interpolation=cv2.INTER_AREA)
        
    def ExtractFeatures(self):
        features = {
            'color_histogram': self._compute_color_histogram(),
            'texture': self._compute_texture_features(),
            'edges': self._detect_edges(),
        }
        return features

    def _compute_color_histogram(self):
        return extract_feature_color_histogram(self.image)

    def _compute_texture_features(self):
        texture_features = extract_feature_texture(self.image)
        return texture_features.tolist()

    def _detect_edges(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        edge_pixels = int(np.count_nonzero(edges))
        total_pixels = int(edges.size)

        return {
            'edge_pixels': edge_pixels,
            'edge_density': float(edge_pixels / max(1, total_pixels)),
        }
