import cv2
import numpy as np
from .feature.ColorFeature import extract_color_vector
from .feature.TextureFeature import extract_texture_vector
from .feature.VenationFeature import extract_venation_vector


def extract_feature_color(img_np: np.ndarray) -> np.ndarray:
    """Extract color histogram features (132D)."""
    if img_np is None or img_np.size == 0:
        return np.array([])
    
    try:
        if img_np.ndim == 2:
            img_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        elif img_np.shape[2] == 4:
            img_rgb = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
        else:
            img_rgb = img_np.copy()
        
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
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY) if img_np.ndim == 3 else img_np.copy()
        texture_vector = extract_texture_vector(gray)
        return np.array(texture_vector, dtype=np.float32)
    except Exception as e:
        print(f"Error extracting texture features: {e}")
        return np.array([])


def extract_feature_venation(img_np: np.ndarray) -> np.ndarray:
    """Extract venation features - vein length, branch points, density (3D)."""
    if img_np is None or img_np.size == 0:
        return np.array([])
    
    try:
        if img_np.ndim == 2:
            img_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        elif img_np.shape[2] == 4:
            img_rgb = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
        else:
            img_rgb = img_np.copy()
        
        vein_vector = extract_venation_vector(img_rgb)
        return np.array(vein_vector, dtype=np.float32)
    except Exception as e:
        print(f"Error extracting venation features: {e}")
        return np.array([])

def extract_feature_HOG(img_np: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)

    img_resized = cv2.resize(gray, (160, 128))
    
    hog = cv2.HOGDescriptor(_winSize=(160, 128), # kích thước ảnh
                            _blockSize=(16, 16), # kích thước block
                            _blockStride=(8, 8), # bước nhảy của block
                            _cellSize=(8, 8), # kích thước cell
                            _nbins=9) # số lượng histogram
    
    hog_features = hog.compute(img_resized)
    
    return hog_features.flatten()

def detect_leaf_contour(img_np: np.ndarray) -> np.ndarray:
    gray= cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    # dùng bộ lọc kernel 5x5
    gray_new= cv2.GaussianBlur(gray, (5,5), 0)
    edges= cv2.Canny(gray_new, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    leaf_contours= max(contours, key=cv2.contourArea)   
    return leaf_contours
    

def extract_feature_shape(img_np: np.ndarray) -> np.ndarray:
    """
     Sử dungj thuật toán tìm biên Canny để tìm đường bao bên ngoài của lá. cho ngưỡng cao để tránh lấy gân lá
    """
    leaf_contours= detect_leaf_contour(img_np)

    area= cv2.contourArea(leaf_contours)
    chuVi= cv2.arcLength(leaf_contours, True)
    
    x, y, w, h=cv2.boundingRect(leaf_contours)
    aspect_ratio= float(w)/h

    rect_area = w * h
    extent = float(area) / rect_area  #Cho biết lá này có chiếm đầy khung hình chữ nhật hay không
    
    hull= cv2.convexHull(leaf_contours)
    hull_area=cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 0
    
    mask= np.zeros(img_np.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask,[leaf_contours],-1,255,-1)

    moments=cv2.moments(mask)
    hu_moments=cv2.HuMoments(moments)

    feature_vector= np.hstack([
        aspect_ratio,
        extent,
        solidity,
        hu_moments.flatten()
    ])
    
    return feature_vector


def extract_feature_color_histogram(img_np: np.ndarray, output_dir: str = None):

    pass

def extract_feature_texture(img_np: np.ndarray) -> np.ndarray:
    pass



