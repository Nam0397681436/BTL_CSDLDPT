import numpy as np
from opencv import cv2
from services.FeatureExtractor import featureExtractor


class Image:

    url_minio=None 

    def __init__(self, path):
        self.source_path = path
        self.image = cv2.imread(path)
        self.url_minio = None

    def preprocess(self):
        self.image = cv2.resize(self.image, (256, 256))
        
    def ExtractFeatures(self):

        features = {
            'color_histogram': self._compute_color_histogram(),
            'edges': self._detect_edges()
        }
        return features
