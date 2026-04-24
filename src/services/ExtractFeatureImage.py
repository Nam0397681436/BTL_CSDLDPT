
import cv2
import numpy as np
from services.feature.ColorFeature import extract_color_vector
from services.feature.TextureFeature import extract_texture_vector


def extract_feature_color(img_np: np.ndarray) -> np.ndarray:
    """Extract color histogram features (132D)."""
    if img_np is None or img_np.size == 0:
        return np.array([])
    
    try:
        if img_np.ndim == 2:
            img_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        elif img_np.shape[2] == 4:
            img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGRA2RGB)
        else:
            img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        
        color_vector = extract_color_vector(img_rgb)
        return np.array(color_vector, dtype=np.float32)
    except Exception as e:
        print(f"Error extracting color features: {e}")
        return np.array([])

def extract_feature_texture(img_np: np.ndarray) -> np.ndarray:
    """Extract texture features - LBP and GLCM (14D)."""
    if img_np is None or img_np.size == 0:
        return np.array([])
    
    try:
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY) if img_np.ndim == 3 else img_np.copy()
        texture_vector = extract_texture_vector(gray)
        return np.array(texture_vector, dtype=np.float32)
    except Exception as e:
        print(f"Error extracting texture features: {e}")
        return np.array([])


def extract_feature_HOG(img_np: np.ndarray) -> np.ndarray:
    if img_np is None or img_np.size == 0:
        return np.array([])

    try:
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (144, 256), interpolation=cv2.INTER_AREA)

        hog = cv2.HOGDescriptor(
            _winSize=(144, 256),
            _blockSize=(32, 32),
            _blockStride=(16, 16),
            _cellSize=(16, 16),
            _nbins=9
        )

        hog_features = hog.compute(gray)
        return hog_features.flatten()

    except Exception as e:
        print(f"Error in extract_feature_HOG: {e}")
        return np.array([])

def _otsu_leaf_mask(img_np: np.ndarray) -> np.ndarray:
    """Otsu thuần: grayscale → GaussianBlur → threshold."""
    gray    = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY) if img_np.ndim == 3 else img_np.copy()
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def detect_leaf_contour(img_np: np.ndarray) -> np.ndarray:
    """Tìm đường bao lớn nhất của lá bằng Otsu thuần."""
    thresh = _otsu_leaf_mask(img_np)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.array([])
    return max(contours, key=cv2.contourArea)
    
def extract_feature_shape(img_np: np.ndarray) -> np.ndarray:
    """
     Sử dungj thuật toán otsu để tìm đường bao bên ngoài của lá. cho ngưỡng cao để tránh lấy gân lá
    """
    leaf_contours = detect_leaf_contour(img_np)
    
    if leaf_contours is None or (isinstance(leaf_contours, np.ndarray) and leaf_contours.size == 0):
        return np.array([])

    area = cv2.contourArea(leaf_contours)
    chuVi = cv2.arcLength(leaf_contours, True)
    
    x, y, w, h = cv2.boundingRect(leaf_contours)
    aspect_ratio = float(w)/h if h > 0 else 0

    rect_area = w * h
    extent = float(area) / rect_area if rect_area > 0 else 0  #Cho biết lá này có chiếm đầy khung hình chữ nhật hay không
    
    hull= cv2.convexHull(leaf_contours)
    hull_area=cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 0
    
    mask= np.zeros(img_np.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask,[leaf_contours],-1,255,-1)

    moments=cv2.moments(mask)
    hu_moments=cv2.HuMoments(moments)

    # Log-transform Hu Moments để đưa về cùng thang đo
    # Hu Moments có giá trị dao động từ 1e-2 đến 1e-22 → log nén lại
    hu_moments_log = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)

    feature_vector= np.hstack([
        aspect_ratio,
        extent,
        solidity,
        hu_moments_log.flatten()
    ])
    
    return feature_vector

