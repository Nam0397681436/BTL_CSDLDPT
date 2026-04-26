"""
ExtractImageDBWeighted.py
--------------------------
Cách tìm kiếm thay thế: thay vì cascade filter (HOG→Shape→Texture→Color),
tính combined weighted distance cho TOÀN BỘ ảnh trong DB rồi lấy top-K.

Ưu điểm:
- Không bị mất ứng viên tốt do lọc quá chặt ở bước đầu
- Kết quả phản ánh đúng sự tương đồng tổng hợp từ tất cả feature
"""

import numpy as np
from src.dao.DAOPostgresql import DAOPostgresql
from src.dao.DAOMinio import DAOMinio
from src.services.computeDistance import (
    compute_distance_color_histogram,
    compute_distance_texture,
    compute_distance_shape,
    compute_distance_hog,
    compute_distance_venation,
)

# Trọng số cho từng loại feature (tổng = 1.0)
FEATURE_WEIGHTS = {
    "hog":      0.40,
    "shape":    0.25,
    "color":    0.20,
    "texture":  0.15
}


class ExtractImageDBWeighted:

    def __init__(self, dao_minio: DAOMinio, dao_postgresql: DAOPostgresql, custom_weights: dict = None):
        self.dao_minio = dao_minio
        self.dao_postgresql = dao_postgresql
        self.top_images = []  # list of (combined_distance, image_id)
        self.weights = custom_weights if custom_weights is not None else FEATURE_WEIGHTS

    def extract_image_postgresql(self, feature_query: dict, top_k: int = 20) -> list:
        """
        Tính combined weighted distance cho toàn bộ ảnh trong DB,
        trả về top_k ảnh có khoảng cách nhỏ nhất (giống nhất).
        """
        scored = []

        for batch in self.dao_postgresql.get_features_in_batches():
            for item in batch:
                score = self._compute_combined_distance(feature_query, item)
                if score is not None:
                    scored.append((score, item["image_id"]))

        scored.sort(key=lambda x: x[0])
        self.top_images = scored[:top_k]
        return self.top_images

    def _compute_combined_distance(self, query: dict, db_item: dict) -> float | None:
        """Tính khoảng cách tổng hợp có trọng số giữa query và 1 ảnh trong DB."""
        total = 0.0
        total_weight = 0.0

        pairs = [
            ("hog",      compute_distance_hog,             query.get("hog"),      db_item.get("hog")),
            ("shape",    compute_distance_shape,           query.get("shape"),    db_item.get("shape")),
            ("color",    compute_distance_color_histogram, query.get("color"),    db_item.get("color")),
            ("texture",  compute_distance_texture,         query.get("texture"),  db_item.get("texture"))
        ]

        for feature_name, dist_fn, q_vec, db_vec in pairs:
            if q_vec is None or db_vec is None:
                continue
            q_arr = np.asarray(q_vec, dtype=np.float32).flatten()
            d_arr = np.asarray(db_vec, dtype=np.float32).flatten()
            if q_arr.size == 0 or d_arr.size == 0:
                continue
            min_dim = min(q_arr.size, d_arr.size)
            try:
                dist = float(dist_fn(q_arr[:min_dim], d_arr[:min_dim]))
            except Exception:
                continue

            w = self.weights.get(feature_name, 0.0)
            total += w * dist
            total_weight += w

        if total_weight == 0:
            return None
        return total / total_weight

    def extract_image_minio(self, _feature_query=None) -> list[str]:
        """Lấy URL MinIO của top_images, giữ đúng thứ tự xếp hạng."""
        list_image_id = [x[1] for x in self.top_images]
        metadata_list = self.dao_postgresql.get_metadata_by_ids(list_image_id)
        url_map = {m["image_id"]: m["minio_url"] for m in metadata_list}
        return [url_map[img_id] for img_id in list_image_id if img_id in url_map]
