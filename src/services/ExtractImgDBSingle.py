import cv2
from src.model.Image import Image
from src.dao.DAOPostgresql import DAOPostgresql
from src.dao.DAOMinio import DAOMinio
import heapq
from src.services.computeDistance import (
    compute_distance_color_histogram,
    compute_distance_texture,
    compute_distance_shape,
    compute_distance_hog,
)


# Trọng số cho từng loại feature (tổng = 1.0)
FEATURE_WEIGHTS = {
    "hog":      0.25,
    "shape":    0.40,
    "color":    0.15,
    "texture":  0.20
}

class ExtractImageDBSingle:
    
    def __init__(self, k_langgieng: int, dao_minio: DAOMinio, dao_postgresql: DAOPostgresql):
        self.k_langgieng = k_langgieng
        self.dao_minio = dao_minio
        self.dao_postgresql = dao_postgresql
        self.k5_images = []

    def extract_image_postgresql(self, feauture_img_query: dict, debug_mode: bool = False) -> list[tuple[float, str, dict]]:
        self.k5_images = []  # Reset for each search
        K = self.k_langgieng

        for batch in self.dao_postgresql.get_features_in_batches():
            for item in batch:
                # Kiểm tra đầy đủ các đặc trưng cần thiết
                if (item["shape"] is not None and feauture_img_query.get("shape") is not None and 
                    item["texture"] is not None and feauture_img_query.get("texture") is not None and 
                    item["hog"] is not None and feauture_img_query.get("hog") is not None and 
                    item["color"] is not None and feauture_img_query.get("color") is not None):
                    
                    distance_shape = compute_distance_shape(feauture_img_query["shape"], item["shape"])
                    distance_texture = compute_distance_texture(feauture_img_query["texture"], item["texture"])
                    distance_hog = compute_distance_hog(feauture_img_query["hog"], item["hog"])
                    distance_color = compute_distance_color_histogram(feauture_img_query["color"], item["color"])

                    total_distance = (distance_shape * FEATURE_WEIGHTS["shape"] + 
                                      distance_texture * FEATURE_WEIGHTS["texture"] + 
                                      distance_hog * FEATURE_WEIGHTS["hog"] + 
                                      distance_color * FEATURE_WEIGHTS["color"])

                    debug_info = {
                        "shape": distance_shape,
                        "texture": distance_texture,
                        "hog": distance_hog,
                        "color": distance_color,
                        "total": total_distance
                    } if debug_mode else {}

                    if len(self.k5_images) < K:
                        heapq.heappush(self.k5_images, (-total_distance, item["image_id"], debug_info))                  
                    else:
                        if total_distance < -self.k5_images[0][0]:
                            heapq.heapreplace(self.k5_images, (-total_distance, item["image_id"], debug_info))
                
        # Chuyển đổi heap thành list đã sắp xếp: [(distance, image_id, debug_dict), ...]
        final_results = []
        while self.k5_images:
            neg_dist, img_id, debug_info = heapq.heappop(self.k5_images)
            final_results.append((-neg_dist, img_id, debug_info))
        
        self.k5_images = sorted(final_results, key=lambda x: x[0])
        return self.k5_images
    
    def extract_image_minio(self, feauture_img_query :dict) -> list[str]:
        # self.k5_images lưu (similarity, image_id)
        list_image_id = [x[1] for x in self.k5_images]
        
        # Gọi hàm get_metadata_by_ids để lấy thông tin ảnh bao gồm URL
        metadata_list = self.dao_postgresql.get_metadata_by_ids(list_image_id)
        
        # Trích xuất danh sách URL từ metadata
        list_image_url = [m["minio_url"] for m in metadata_list]

        return list_image_url
