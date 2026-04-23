import cv2
import numpy as np
from .feature.ColorFeature import extract_color_vector
from .feature.TextureFeature import extract_texture_vector
from .feature.VenationFeature import extract_venation_vector
from .PreprocessImage import can_bang_clahe


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
    """Trích xuất đặc trưng HOG tập trung vào vùng lá bằng cách sử dụng mask lọc nền mạnh mẽ."""
    if img_np is None or img_np.size == 0:
        return np.array([])

    try:
        # 1. Chuyển sang ảnh xám và làm mờ để giảm nhiễu
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        img_clahe=can_bang_clahe(gray)
        
        # 2. Tạo mask dùng Otsu Thresholding (tự động tìm ngưỡng tối ưu)
        # Thử cả 2 trường hợp: Lá sáng trên nền tối và ngược lại
        _, thresh = cv2.threshold(img_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Đảm bảo phần diện tích lớn hơn là nền (thường là vậy), nếu không thì invert
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_cnt = max(contours, key=cv2.contourArea)
            # Nếu contour lớn nhất chiếm quá ít diện tích, có thể cần invert mask
            if cv2.contourArea(largest_cnt) < (img_np.shape[0] * img_np.shape[1] * 0.1):
                _, thresh = cv2.threshold(img_clahe, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest_cnt = max(contours, key=cv2.contourArea)

            # Tạo mask từ contour lớn nhất
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [largest_cnt], -1, 255, -1)
            
            # Làm mượt mask bằng phép đóng (Closing)
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        else:
            mask = np.ones_like(gray) * 255 # Fallback nếu không tìm thấy contour

        # 3. Loại bỏ nền
        img_masked = cv2.bitwise_and(img_np, img_np, mask=mask)
        
        # 4. Chuyển sang ảnh xám và resize (160x128)
        gray_masked = cv2.cvtColor(img_masked, cv2.COLOR_BGR2GRAY)
        img_resized = cv2.resize(gray_masked, (160, 128))
    
        # 5. Khởi tạo HOG Descriptor
        hog = cv2.HOGDescriptor(_winSize=(160, 128),
                                _blockSize=(16, 16),
                                _blockStride=(8, 8),
                                _cellSize=(8, 8),
                                _nbins=9)
        
        # 6. Tính toán các đặc trưng
        hog_features = hog.compute(img_resized)
        return hog_features.flatten()
        
    except Exception as e:
        print(f"Error in extract_feature_HOG: {e}")
        return np.array([])

def extract_feature_HOG_ROTATE(img_np:np.ndarray):
    """Trích xuất đặc trưng HOG tập trung vào vùng lá bằng cách sử dụng mask lọc nền mạnh mẽ."""
    if img_np is None or img_np.size == 0:
        return np.array([])

    try:
        # 1. Chuyển sang ảnh xám và làm mờ để giảm nhiễu
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        img_clahe=can_bang_clahe(gray)
        
        # 2. Tạo mask dùng Otsu Thresholding (tự động tìm ngưỡng tối ưu)
        # Thử cả 2 trường hợp: Lá sáng trên nền tối và ngược lại
        _, thresh = cv2.threshold(img_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Đảm bảo phần diện tích lớn hơn là nền (thường là vậy), nếu không thì invert
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_cnt = max(contours, key=cv2.contourArea)
            # Nếu contour lớn nhất chiếm quá ít diện tích, có thể cần invert mask
            if cv2.contourArea(largest_cnt) < (img_np.shape[0] * img_np.shape[1] * 0.1):
                _, thresh = cv2.threshold(img_clahe, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest_cnt = max(contours, key=cv2.contourArea)

            # Tạo mask từ contour lớn nhất
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [largest_cnt], -1, 255, -1)
            
            # Làm mượt mask bằng phép đóng (Closing)
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        else:
            mask = np.ones_like(gray) * 255 # Fallback nếu không tìm thấy contour

        # 3. Loại bỏ nền
        img_masked = cv2.bitwise_and(img_np, img_np, mask=mask)
        
        # 4. Chuyển sang ảnh xám và resize (160x128)
        gray_masked = cv2.cvtColor(img_masked, cv2.COLOR_BGR2GRAY)
        img_resized = cv2.resize(gray_masked, (160, 128))
    
        # 5. Khởi tạo HOG Descriptor
        hog = cv2.HOGDescriptor(_winSize=(160, 128),
                                _blockSize=(16, 16),
                                _blockStride=(8, 8),
                                _cellSize=(8, 8),
                                _nbins=9)
        
        # 6. Tính toán các đặc trưng
        hog_features = hog.compute(img_resized)
        return hog_features.flatten()
        
    except Exception as e:
        print(f"Error in extract_feature_HOG: {e}")
        return np.array([])


def detect_leaf_contour(img_np: np.ndarray) -> np.ndarray:
    """Tìm đường bao lớn nhất của lá (sử dụng Otsu để tăng độ chính xác)."""
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.array([])
        
    return max(contours, key=cv2.contourArea)
    
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





