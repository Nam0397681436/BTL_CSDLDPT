import cv2
import numpy as np



def preprocess_image(img_input: np.ndarray) -> np.ndarray:
    img_new=cv2.resize(img_input, (256, 256), interpolation=cv2.INTER_AREA)
    img_new=lam_min_anh(img_new)
    img_new=can_bang_clahe(img_new)
    img_new=filtering_image(img_new)
    return img_new

def lam_min_anh(img_input: np.ndarray) -> np.ndarray:
    """ lamf min ảnh bằng bộ lọc Gaussian """
    img_new=cv2.GaussianBlur(img_input, (5, 5), 0)
    return img_new

def can_bang_clahe(img_input: np.ndarray) -> np.ndarray:
    """ Cân bằng histogram thích nghi (CLAHE) giúp làm nổi bật gân lá và đặc trưng HOG """
    # Nếu là ảnh màu, chuyển sang ảnh xám trước khi dùng CLAHE
    if len(img_input.shape) == 3:
        img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
        
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_new = clahe.apply(img_input)
    return img_new


def filtering_image(img_input: np.ndarray) -> np.ndarray: # lọc với thông cao để nổi bật các chi tiết có tần số cao như biên của đối tượng
    """ lọc với thông cao để nổi bật các chi tiết có tần số cao như biên của đối tượng """
    img_new=cv2.Laplacian(img_input, cv2.CV_8U)
    return img_new




