import cv2
import numpy as np


def _ensure_rgb(image):
    if image is None:
        raise ValueError("Input image is empty.")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    return image.copy()


def _largest_leaf_mask(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, threshold = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((5, 5), np.uint8)
    candidates = []

    for candidate in (threshold, cv2.bitwise_not(threshold)):
        cleaned = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area <= 0:
            continue

        mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)
        x, y, w, h = cv2.boundingRect(largest)
        area_ratio = area / float(max(1, w * h))
        score = area * (1.5 if 0.2 <= area_ratio <= 0.98 else 0.5)
        candidates.append((score, mask))

    if not candidates:
        return np.full_like(gray, 255, dtype=np.uint8)

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _square_canvas(image_rgb, mask, target=512):
    coords = cv2.findNonZero(mask)
    if coords is None:
        resized_image = cv2.resize(image_rgb, (target, target), interpolation=cv2.INTER_AREA)
        resized_mask = cv2.resize(mask, (target, target), interpolation=cv2.INTER_NEAREST)
        return resized_image, resized_mask

    x, y, w, h = cv2.boundingRect(coords)
    crop = image_rgb[y:y + h, x:x + w]
    crop_mask = mask[y:y + h, x:x + w]

    scale = target / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized_image = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(crop_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    canvas = np.zeros((target, target, 3), dtype=np.uint8)
    canvas_mask = np.zeros((target, target), dtype=np.uint8)

    offset_x = (target - new_w) // 2
    offset_y = (target - new_h) // 2
    canvas[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized_image
    canvas_mask[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized_mask

    return canvas, canvas_mask


def compute_color_features(image_rgb, mask, bins=42):
    image_rgb = _ensure_rgb(image_rgb)
    mask = mask.astype(np.uint8)

    if mask.ndim != 2:
        raise ValueError("Mask must be a single-channel image.")

    if cv2.countNonZero(mask) == 0:
        mask = np.full(mask.shape, 255, dtype=np.uint8)

    channels = cv2.split(image_rgb)
    color_vec = []

    for channel in channels:
        hist = cv2.calcHist([channel], [0], mask, [bins], [0, 256]).flatten()
        hist = hist / (hist.sum() + 1e-7)
        color_vec.extend(hist.tolist())

    mean, std = cv2.meanStdDev(image_rgb, mask=mask)
    mean = mean.flatten()
    std = std.flatten()

    return {
        "RGB_Histogram_126D": color_vec,
        "R_mean": float(mean[0]),
        "G_mean": float(mean[1]),
        "B_mean": float(mean[2]),
        "R_std": float(std[0]),
        "G_std": float(std[1]),
        "B_std": float(std[2]),
    }


def extract_color_vector(image):
    """Extract color histogram feature vector (132D) from image."""
    image_rgb = _ensure_rgb(image)
    mask = _largest_leaf_mask(image_rgb)
    final_image, final_mask = _square_canvas(image_rgb, mask)
    color_feats = compute_color_features(final_image, final_mask)

    hist = np.array(color_feats["RGB_Histogram_126D"], dtype=np.float32).reshape(3, -1).T
    hist_rows = [[int(i), float(hist[i, 0]), float(hist[i, 1]), float(hist[i, 2])] for i in range(hist.shape[0])]
    
    color_data = {
        "R_mean": color_feats["R_mean"],
        "G_mean": color_feats["G_mean"],
        "B_mean": color_feats["B_mean"],
        "R_std": color_feats["R_std"],
        "G_std": color_feats["G_std"],
        "B_std": color_feats["B_std"],
        "RGB_histogram_42x3": hist_rows,
    }
    
    stats = [
        color_data["R_mean"], color_data["G_mean"], color_data["B_mean"],
        color_data["R_std"], color_data["G_std"], color_data["B_std"]
    ]
    hist_array = np.array(color_data["RGB_histogram_42x3"])
    hist_values = hist_array[:, 1:].flatten().tolist()
    color_vector = stats + hist_values
    
    return color_vector