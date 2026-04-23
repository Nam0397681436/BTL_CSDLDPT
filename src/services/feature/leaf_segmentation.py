import cv2
import numpy as np


def enlarge_if_small(image: np.ndarray, min_dim: int = 256) -> np.ndarray:
    """Phóng to ảnh nếu kích thước quá nhỏ để tránh lỗi khi xử lý hình thái học (morphology)."""
    h, w = image.shape[:2]
    if min(h, w) < min_dim:
        scale = min_dim / min(h, w)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return image

def _ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image.copy()


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8) * 255
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def _extract_largest_contour(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask
    largest = max(contours, key=cv2.contourArea)
    result = np.zeros_like(mask)
    cv2.drawContours(result, [largest], -1, 255, thickness=cv2.FILLED)
    return result


def _is_texture_only_image(gray: np.ndarray) -> bool:
    """
    Kiểm tra xem ảnh có phải là texture-only (không có nền rõ ràng) không.
    """
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-7)
    
    dark_pixels = hist[:30].sum()
    bright_pixels = hist[225:].sum()
    
    if dark_pixels < 0.05 and bright_pixels < 0.05:
        return True
        
    h, w = gray.shape
    border = 5
    if h > 20 and w > 20:
        border_region = np.concatenate([
            gray[:border, :].flatten(),
            gray[-border:, :].flatten(),
            gray[:, :border].flatten(),
            gray[:, -border:].flatten()
        ])
        center_region = gray[h//3:2*h//3, w//3:2*w//3].flatten()
        
        border_mean, border_std = np.mean(border_region), np.std(border_region)
        center_mean, center_std = np.mean(center_region), np.std(center_region)
        
        mean_diff = abs(border_mean - center_mean) / 255.0
        std_diff = abs(border_std - center_std) / (max(border_std, center_std) + 1e-7)
        
        if mean_diff < 0.15 and std_diff < 0.3:
            return True
    
    return False


def _segment_color_leaf(image_rgb: np.ndarray) -> np.ndarray | None:
    """Segment lá xanh dựa trên màu HSV."""
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    rgb = image_rgb.astype(np.int16)

    green_hue = ((h >= 20) & (h <= 90)).astype(np.uint8) * 255
    green_saturation = (s >= 25).astype(np.uint8) * 255
    green_value = (v >= 20).astype(np.uint8) * 255
    green_dominance = (
        (rgb[:, :, 1] > rgb[:, :, 0] + 5) & 
        (rgb[:, :, 1] > rgb[:, :, 2] + 5)
    ).astype(np.uint8) * 255

    mask = cv2.bitwise_and(green_hue, green_saturation)
    mask = cv2.bitwise_and(mask, green_value)
    mask = cv2.bitwise_and(mask, green_dominance)

    cleaned = _clean_mask(mask)
    result = _extract_largest_contour(cleaned)
    
    coverage = np.count_nonzero(result) / result.size
    if coverage == 0 or coverage > 0.98:
        return None
    return result


def _segment_grayscale_leaf(image_bgr: np.ndarray) -> np.ndarray:
    """Segment ảnh grayscale/non-green bằng nhiều phương pháp."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    if _is_texture_only_image(gray):
        return np.full((h, w), 255, dtype=np.uint8)
    
    candidates = []
    
    # Strategy 1: OTSU trên ảnh blur mạnh
    strong_blur = cv2.GaussianBlur(gray, (51, 51), 0)
    _, otsu_mask = cv2.threshold(strong_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg_mean = np.mean(gray[otsu_mask > 0])
    bg_mean = np.mean(gray[otsu_mask == 0])
    bright_mask = otsu_mask if fg_mean > bg_mean else (255 - otsu_mask)
    bright_mask = _clean_mask(bright_mask)
    bright_mask = _extract_largest_contour(bright_mask)
    candidates.append(("otsu", bright_mask))
    
    # Strategy 2: Adaptive thresholding
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 51, 5)
    adaptive = _clean_mask(adaptive)
    adaptive = _extract_largest_contour(adaptive)
    candidates.append(("adaptive", adaptive))
    
    # Strategy 3: Edge-based
    edges = cv2.Canny(gray, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)
    edges_closed = cv2.morphologyEx(edges_closed, cv2.MORPH_DILATE, kernel, iterations=2)
    edge_mask = _extract_largest_contour(edges_closed)
    candidates.append(("edge", edge_mask))
    
    # Strategy 4: GrabCut
    mask_gc = np.zeros((h, w), np.uint8)
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)
    rect = (w//6, h//6, 2*w//3, 2*h//3)
    try:
        cv2.grabCut(image_bgr, mask_gc, rect, bg_model, fg_model, 5, cv2.GC_INIT_WITH_RECT)
        grabcut_mask = np.where((mask_gc == 2) | (mask_gc == 0), 0, 1).astype('uint8') * 255
        grabcut_mask = _extract_largest_contour(grabcut_mask)
        candidates.append(("grabcut", grabcut_mask))
    except:
        pass
    
    best_mask = None
    best_score = -1
    
    for name, candidate in candidates:
        if np.count_nonzero(candidate) == 0:
            continue
        coverage = np.count_nonzero(candidate) / candidate.size
        if coverage == 0 or coverage > 0.98:
            continue
        moments = cv2.moments(candidate)
        if moments["m00"] == 0:
            continue
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        center_dist = np.sqrt((cx - w/2)**2 + (cy - h/2)**2) / np.sqrt((w/2)**2 + (h/2)**2)
        
        contours, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        hull = cv2.convexHull(largest)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        score = (1.0 - center_dist) * 0.4 + solidity * 0.3 + (1.0 - abs(coverage - 0.5) * 2) * 0.3
        
        if score > best_score:
            best_score = score
            best_mask = candidate
    
    if best_mask is not None:
        return best_mask
    
    return np.full((h, w), 255, dtype=np.uint8)


def extract_leaf_mask(image: np.ndarray) -> np.ndarray:
    """
    Trích xuất mask lá từ ảnh. Hỗ trợ cả ảnh màu (lá xanh) và ảnh grayscale.
    """
    image_bgr = _ensure_bgr(image)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    color_mask = _segment_color_leaf(image_rgb)
    if color_mask is not None:
        return color_mask
    
    return _segment_grayscale_leaf(image_bgr)


def crop_leaf_region(image: np.ndarray, mask: np.ndarray, target_size: tuple[int, int] = (256, 256)) -> tuple[np.ndarray, np.ndarray]:
    image_bgr = _ensure_bgr(image)
    mask = (mask > 0).astype(np.uint8) * 255

    coords = cv2.findNonZero(mask)
    if coords is None:
        crop_img = image_bgr
        crop_mask = mask
    else:
        # Cắt ảnh theo Bounding Box của lá
        x, y, width, height = cv2.boundingRect(coords)
        crop_img = image_bgr[y:y + height, x:x + width]
        crop_mask = mask[y:y + height, x:x + width]

    # Kích thước gốc của phần ảnh vừa cắt
    h, w = crop_img.shape[:2]
    target_w, target_h = target_size

    # Tính toán tỷ lệ thu phóng để giữ nguyên tỷ lệ khung hình (Aspect Ratio)
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    # Chọn interpolation method phù hợp
    interpolation_method = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    resized_image = cv2.resize(crop_img, (new_w, new_h), interpolation=interpolation_method)
    resized_mask = cv2.resize(crop_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # Tạo Canvas (khung hình) nền đen theo target_size
    canvas_img = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    canvas_mask = np.zeros((target_h, target_w), dtype=np.uint8)

    # Tính toán tọa độ để đặt ảnh vào chính giữa Canvas
    offset_x = (target_w - new_w) // 2
    offset_y = (target_h - new_h) // 2

    canvas_img[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized_image
    canvas_mask[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized_mask

    # Làm sạch mask lần cuối
    canvas_mask = _clean_mask(canvas_mask)

    return canvas_img, canvas_mask