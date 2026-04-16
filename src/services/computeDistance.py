import cv2
import numpy as np
from src.model.Image import Image

def compute_distance(image1: Image, image2: Image) -> float:
    feature_image1= image1.ExtractFeatures()
    feature_image2= image2.ExtractFeatures()
    
    distance_hog= compute_distance_hog(feature_image1['hog'], feature_image2['hog'])
    distance_shape= compute_distance_shape(feature_image1['shape'], feature_image2['shape'])
    distance_texture= compute_distance_texture(feature_image1['texture'], feature_image2['texture'])
    distance_color= compute_distance_color_histogram(feature_image1['color'], feature_image2['color'])
    
    distance= 0.05*distance_hog + 0.8*distance_shape + 0.8*distance_texture + 0.8*distance_color
    return distance


def compute_distance_hog(feature1: np.ndarray, feature2: np.ndarray) -> float:
    # Tính khoảng cách L3 (căn bậc 3 của tổng các lập phương hiệu) giữa hai vector HOG
    distance = np.linalg.norm(feature1 - feature2, ord=3)
    return distance

def compute_distance_venation(feature1: np.ndarray, feature2: np.ndarray) -> float:
    # Tính khoảng cách tương quan (Correlation) giữa hai vector đặc trưng gân lá
    feat1 = np.asarray(feature1, dtype=np.float32)
    feat2 = np.asarray(feature2, dtype=np.float32)
    # Sử dụng HISTCMP_CORREL: Giá trị càng cao (gần 1) thì càng giống nhau
    distance = np.linalg.norm(feat1-feat2, ord=1)
    return distance

def compute_distance_color_histogram(feature1: np.ndarray, feature2: np.ndarray) -> float:
    # Tính khoảng cách tương quan giữa hai histogram màu
    feat1 = np.asarray(feature1, dtype=np.float32)
    feat2 = np.asarray(feature2, dtype=np.float32)
    distance = np.linalg.norm(feat1-feat2, ord=1)
    return distance

def compute_distance_texture(feature1: np.ndarray, feature2: np.ndarray) -> float:
    # Tính khoảng cách tương quan giữa hai histogram texture (kết cấu)
    feat1 = np.asarray(feature1, dtype=np.float32)
    feat2 = np.asarray(feature2, dtype=np.float32)
    distance = np.linalg.norm(feat1-feat2, ord=1)
    return distance

def compute_distance_shape(feature1: np.ndarray, feature2: np.ndarray) -> float:
    # Tính khoảng cách tương quan giữa hai vector đặc trưng hình dạng
    feat1 = np.asarray(feature1, dtype=np.float32)
    feat2 = np.asarray(feature2, dtype=np.float32)
    distance = np.linalg.norm(feat1-feat2, ord=1)
    return distance

def compute_distance_venation(feature1: np.ndarray, feature2: np.ndarray) -> float:
    # Tính khoảng cách tương quan giữa hai vector đặc trưng venotion
    try: 
        feat1 = np.asarray(feature1, dtype=np.float32)
        feat2 = np.asarray(feature2, dtype=np.float32)
        distance = np.linalg.norm(feat1-feat2, ord=1)
        return distance
    except Exception as e:
        print(e)
        return 0.0