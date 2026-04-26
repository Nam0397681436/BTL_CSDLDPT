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
        
        candidate_shape=[]
        candidate_hog=[]
        candidate_texture=[]
        candidate_color=[]

        K1 = 30
        K2 = 30
        K3 = 20
        K4 = 20

        for batch in self.dao_postgresql.get_features_in_batches():
            for item in batch:
                if item["hog"] is not None and feauture_img_query.get("hog") is not None:
                    distance_hog = compute_distance_hog(
                        feauture_img_query["hog"], item["hog"]
                    )

                    if len(candidate_hog) < K1:
                        heapq.heappush(candidate_hog, (-distance_hog, item))                  
                    else:
                        # so với phần tử lớn nhất hiện tại max-heapq giả
                        if distance_hog < -candidate_hog[0][0]:
                            heapq.heapreplace(candidate_hog, (-distance_hog, item))
                
        for item1 in candidate_hog:
            distance_shape=compute_distance_shape(feauture_img_query["shape"], item1[1]["shape"])

            if len(candidate_shape) < K2:
                heapq.heappush(candidate_shape, (-distance_shape, item1[1]))
            else:
                # so với phần tử lớn nhất hiện tại max-heapq giả
                if distance_shape < -candidate_shape[0][0]:
                    heapq.heapreplace(candidate_shape, (-distance_shape, item1[1]))
        
        for item2 in candidate_shape:
            distance_texture=compute_distance_texture(feauture_img_query["texture"], item2[1]["texture"])

            if len(candidate_texture) < K3:
                heapq.heappush(candidate_texture, (-distance_texture, item2[1]))
            else:
                # so với phần tử lớn nhất hiện tại max-heapq giả
                if distance_texture < -candidate_texture[0][0]:
                    heapq.heapreplace(candidate_texture, (-distance_texture, item2[1]))
        
        
        for item4 in candidate_texture:
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

