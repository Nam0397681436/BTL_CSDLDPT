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
    compute_distance_venation
)

class ExtractImageDB:
    
    k5_images=[]
    
    def __init__(self,dao_minio:DAOMinio,dao_postgresql:DAOPostgresql):
        self.dao_minio=dao_minio
        self.dao_postgresql=dao_postgresql

    def extract_image_postgresql(self, feauture_img_query :dict) -> list[tuple[float,str]]:
        
        candidate1_shape=[]
        candidate2_hog=[]
        candidate3_texture=[]
        candidate4_color=[]

        K1 = 100
        K2 = 10
        K3 = 10
        K4 = 10

        for batch in self.dao_postgresql.get_features_in_batches():
            for item in batch:
                if item["color"] is not None and feauture_img_query.get("color") is not None:
                    distance_shape = compute_distance_shape(
                        feauture_img_query["shape"], item["shape"]
                    )

                    if len(candidate1_shape) < K1:
                        heapq.heappush(candidate1_shape, (-distance_shape, item))                  
                    else:
                        # so với phần tử lớn nhất hiện tại max-heapq giả
                        if distance_shape < -candidate1_shape[0][0]:
                            heapq.heapreplace(candidate1_shape, (-distance_shape, item))
                
        for item1 in candidate1_shape:
            distance_hog=compute_distance_shape(feauture_img_query["shape"], item1[1]["shape"])

            if len(candidate2_hog) < K2:
                heapq.heappush(candidate2_hog, (-distance_hog, item1[1]))
            else:
                # so với phần tử lớn nhất hiện tại max-heapq giả
                if distance_hog < -candidate2_hog[0][0]:
                    heapq.heapreplace(candidate2_hog, (-distance_hog, item1[1]))
        
        for item2 in candidate2_hog:
            distance_texture=compute_distance_texture(feauture_img_query["texture"], item2[1]["texture"])

            if len(candidate3_texture) < K3:
                heapq.heappush(candidate3_texture, (-distance_texture, item2[1]))
            else:
                # so với phần tử lớn nhất hiện tại max-heapq giả
                if distance_texture < -candidate3_texture[0][0]:
                    heapq.heapreplace(candidate3_texture, (-distance_texture, item2[1]))
        
        
        for item4 in candidate3_texture:
            distance=compute_distance_color_histogram(feauture_img_query["color"], item4[1]["color"])

            if len(self.k5_images) < K4:
                heapq.heappush(self.k5_images, (-distance, item4[1]["image_id"]))
            else:
                # so với phần tử lớn nhất hiện tại max-heapq giả
                if distance < -self.k5_images[0][0]:
                    heapq.heapreplace(self.k5_images, (-distance, item4[1]["image_id"]))
        
        return self.k5_images
    
    def extract_image_minio(self, feauture_img_query :dict) -> list[str]:
        # self.k5_images lưu (similarity, image_id)
        list_image_id = [x[1] for x in self.k5_images]
        
        # Gọi hàm get_metadata_by_ids để lấy thông tin ảnh bao gồm URL
        metadata_list = self.dao_postgresql.get_metadata_by_ids(list_image_id)
        
        # Trích xuất danh sách URL từ metadata
        list_image_url = [m["minio_url"] for m in metadata_list]

        return list_image_url

