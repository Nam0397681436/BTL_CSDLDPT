import hashlib
import cv2
import hashlib
from pathlib import Path

from src.services.ExtractFeatureImage import (
    extract_feature_texture,
    extract_feature_shape,
    extract_feature_HOG,
    extract_feature_venation,
    extract_feature_color,
)


class Image:
    def __init__(self, path=None, img_input=None):
        if img_input is not None: # sử dụng cho trường hợp tìm kiếm ảnh đầu vào
            self.image=img_input
            self.image_id = hashlib.md5(img_input.tobytes()).hexdigest()
        else: # sử dụng cho phần trích xuất đặc trưng lưu vào db
            self.source_path = path
            self.original_image = cv2.imread(path) # Giữ ảnh gốc độ phân giải cao
            if self.original_image is None:
                raise FileNotFoundError(f"Cannot read image: {path}")

            # Bản sao để trích xuất đặc trưng (sẽ được resize ở preprocess)
            self.image = self.original_image.copy()

            path_obj = Path(self.source_path)
            self.category = self._extract_category_from_path(path_obj)
            self.image_id = hashlib.md5(path.encode()).hexdigest()
            self.object_name = f"{self.category}/{self.image_id}.jpg"
            self.url_minio = f"http://localhost:9001/browser/plantsimage/{self.object_name}"

    @staticmethod
    def _extract_category_from_path(path_obj: Path) -> str:
        parts = [p for p in path_obj.parts if p not in (".", "")]
        for idx, part in enumerate(parts):
            if part.lower() == "data" and idx + 1 < len(parts):
                candidate = parts[idx + 1]
                if candidate.lower() != "healthy":
                    return candidate
        return "unknown"

    def preprocess(self):
        # Resize bản dùng cho trích xuất đặc trưng
        self.image = cv2.resize(self.image, (256, 256), interpolation=cv2.INTER_AREA)
        
    def get_storage_image_bytes(self):
        """Trả về dữ liệu ảnh 1200x800 định dạng bytes để upload lên MinIO."""
        resized = cv2.resize(self.original_image, (1200, 800), interpolation=cv2.INTER_CUBIC)
        is_success, buffer = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if is_success:
            import io
            return io.BytesIO(buffer)
        return None

    @staticmethod
    def get_storage_image_bytes_from_path(path: str):
        """Đọc ảnh theo path và trả về bytes JPEG 1200x800 để upload MinIO."""
        image = cv2.imread(path)
        if image is None:
            return None
        resized = cv2.resize(image, (1200, 800), interpolation=cv2.INTER_CUBIC)
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
