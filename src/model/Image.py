import numpy as np
import cv2
import hashlib
import os
from datetime import datetime
from pathlib import Path

from services.ExtractFeatureImage import (
    extract_feature_texture,
    extract_feature_shape,
    extract_feature_HOG,
    extract_feature_venation,
    extract_feature_color,
)


class Image:
    def __init__(self, path):
        self.source_path = path
        self.original_image = cv2.imread(path) # Giữ ảnh gốc độ phân giải cao
        if self.original_image is None:
            raise FileNotFoundError(f"Cannot read image: {path}")

        # Bản sao để trích xuất đặc trưng (sẽ được resize ở preprocess)
        self.image = self.original_image.copy()

        path_obj = Path(self.source_path)
        parts = path_obj.parent.parent.name.split()
        self.category = " ".join(parts[:-1]) if len(parts) > 1 else parts[0]
        self.image_id = hashlib.md5(path.encode()).hexdigest()
        self.object_name = f"{self.category}/{self.image_id}.jpg"
        self.url_minio = f"http://localhost:9001/browser/plantsimage/{self.object_name}"

    def preprocess(self):
        # Resize bản để trích xuất đặc trưng về 256x256
        self.image = cv2.resize(self.image, (256, 256), interpolation=cv2.INTER_AREA)
        
    def get_storage_image_bytes(self):
        """Trả về dữ liệu ảnh 1200x800 định dạng bytes để upload lên MinIO."""
        resized = cv2.resize(self.original_image, (1200, 800), interpolation=cv2.INTER_CUBIC)
        is_success, buffer = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if is_success:
            import io
            return io.BytesIO(buffer)
        return None
        

    def ExtractFeatures(self):
        features = {
            "image_id": self.image_id,
            "texture": self._compute_texture_features(),
            'shape': extract_feature_shape(self.image),
            'hog': extract_feature_HOG(self.image),
            'venation': extract_feature_venation(self.image), # xem bo feature nay 
            'color': extract_feature_color(self.image)
        }
        return features

    def _compute_texture_features(self):
        texture_features = extract_feature_texture(self.image)
        return texture_features.tolist()
