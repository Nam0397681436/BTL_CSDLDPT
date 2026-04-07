import cv2
import numpy as np
from model.Image import Image

def compute_distance(image1: Image, image2: Image) -> float:
    feature_image1= image1.ExtractFeatures()
    feature_image2= image2.ExtractFeatures()
    
    distance_hog= compute_distance_hog(feature_image1['hog'], feature_image2['hog'])
    distance_shape= compute_distance_shape(feature_image1['shape'], feature_image2['shape'])
    
    distance= 0.6*distance_hog + 0.4*distance_shape
    return distance


def compute_distance_hog(feature1: np.ndarray, feature2: np.ndarray) -> float:
    pass

def compute_distance_color_histogram(feature1: np.ndarray, feature2: np.ndarray) -> float:
    pass

def compute_distance_texture(feature1: np.ndarray, feature2: np.ndarray) -> float:
    pass

def compute_distance_shape(feature1: np.ndarray, feature2: np.ndarray) -> float:
    pass

