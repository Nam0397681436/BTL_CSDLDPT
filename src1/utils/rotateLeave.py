import cv2
import numpy as np

def _get_leaf_mask(img_np: np.ndarray) -> np.ndarray:
    """
    Tạo mask nhị phân tách lá khỏi nền.
    Ưu tiên: HSV green mask → GrabCut → Otsu auto-invert fallback
    """
    h, w = img_np.shape[:2]

    # --- Bước 1: HSV green mask (phát hiện vùng xanh lá) ---
    hsv = cv2.cvtColor(img_np, cv2.COLOR_BGR2HSV)
    # Xanh lá cây (hue 35–85), bao phủ cả lá xanh đậm/nhạt
    lower_green = np.array([25, 20, 20])
    upper_green = np.array([95, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Morphological closing để lấp lỗ bên trong lá
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)

    # Tính tỉ lệ vùng xanh phát hiện được
    green_ratio = np.sum(green_mask == 255) / green_mask.size

    if green_ratio > 0.05:
        # HSV đủ tốt → dùng luôn
        return green_mask

    # --- Bước 2: GrabCut với ROI trung tâm (fallback khi lá không xanh rõ) ---
    mask_gc = np.zeros((h, w), np.uint8)
    margin_x, margin_y = int(w * 0.1), int(h * 0.1)
    rect_gc = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(img_np, mask_gc, rect_gc, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        grabcut_mask = np.where((mask_gc == cv2.GC_FGD) | (mask_gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        if np.sum(grabcut_mask == 255) / grabcut_mask.size > 0.05:
            kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            grabcut_mask = cv2.morphologyEx(grabcut_mask, cv2.MORPH_CLOSE, kernel2)
            return grabcut_mask
    except Exception:
        pass

    # --- Bước 3: Otsu fallback với auto-invert ---
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white_ratio = np.sum(thresh == 255) / thresh.size
    if white_ratio > 0.5:
        thresh = cv2.bitwise_not(thresh)
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel3)
    return thresh


def rotate_leave(img_np: np.ndarray) -> np.ndarray:
    """Tách nền để xoay vật thể (lá) vào ảnh mới theo trục dọc lá."""
    thresh = _get_leaf_mask(img_np)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_np  # fallback: không tìm được contour

    # Tìm contour lớn nhất (= lá)
    largest_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest_contour)

    box_point_rect = cv2.boxPoints(rect)
    box_point_rect = np.intp(box_point_rect)

    # Vẽ bounding box lên ảnh (để debug)
    cv2.drawContours(img_np, [box_point_rect], 0, (0, 255, 0), 2)

    width = int(rect[1][0])
    height = int(rect[1][1])

    if width == 0 or height == 0:
        return img_np

    src_pts = box_point_rect.astype("float32")
    dst_pts = np.array([[0, height - 1],
                        [0, 0],
                        [width - 1, 0],
                        [width - 1, height - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img_np, M, (width, height))

    # Đảm bảo ảnh luôn nằm ngang (landscape: width >= height)
    if warped.shape[0] > warped.shape[1]:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

    return warped

def ensure_rgb(image: np.ndarray) -> np.ndarray:
    if image is None:
        raise ValueError("Input image is empty.")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    return image.copy()

def largest_leaf_mask(image_rgb: np.ndarray) -> np.ndarray:
    """Sử dụng _get_leaf_mask thay vì logic cũ để đồng nhất với các phần khác."""
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    mask = _get_leaf_mask(image_bgr)
    return mask

def clean_binary_mask(mask: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    return cleaned

def crop_leaf_region(img: np.ndarray, mask: np.ndarray, target_size=(512, 512)):
    coords = cv2.findNonZero(mask)
    if coords is None:
        resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        resized_mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
        return resized_img, resized_mask

    x, y, w, h = cv2.boundingRect(coords)
    crop_img = img[y:y+h, x:x+w]
    crop_mask = mask[y:y+h, x:x+w]

    scale = min(target_size[0] / w, target_size[1] / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized_img = cv2.resize(crop_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(crop_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    canvas_img = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    canvas_mask = np.zeros((target_size[1], target_size[0]), dtype=np.uint8)

    offset_x = (target_size[0] - new_w) // 2
    offset_y = (target_size[1] - new_h) // 2

    canvas_img[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = resized_img
    canvas_mask[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = resized_mask

    return canvas_img, canvas_mask