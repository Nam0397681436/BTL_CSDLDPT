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


def _clean_binary_mask(mask):
    mask = (mask > 0).astype(np.uint8) * 255
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask

    largest = max(contours, key=cv2.contourArea)
    cleaned = np.zeros_like(mask)
    cv2.drawContours(cleaned, [largest], -1, 255, thickness=cv2.FILLED)
    return cleaned


def _seed_from_color(image_rgb):
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    rgb = image_rgb.astype(np.int16)

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    green_dominance = ((rgb[:, :, 1] > rgb[:, :, 0] + 8) & (rgb[:, :, 1] > rgb[:, :, 2] + 8)).astype(np.uint8) * 255
    green_hue = ((h >= 20) & (h <= 90)).astype(np.uint8) * 255
    green_saturation = (s >= 30).astype(np.uint8) * 255
    green_value = (v >= 20).astype(np.uint8) * 255

    seed = cv2.bitwise_and(green_hue, green_saturation)
    seed = cv2.bitwise_and(seed, green_value)
    seed = cv2.bitwise_and(seed, green_dominance)

    return _clean_binary_mask(seed)


def _largest_leaf_mask(image_rgb):
    image_rgb = _ensure_rgb(image_rgb)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    candidate_masks = []

    strict_seed = _seed_from_color(image_rgb)
    if cv2.countNonZero(strict_seed) > 0:
        candidate_masks.append(strict_seed)

    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    rgb = image_rgb.astype(np.int16)
    relaxed_seed = (
        (
            ((hsv[:, :, 0] >= 15) & (hsv[:, :, 0] <= 100))
            & (hsv[:, :, 1] >= 18)
            & (hsv[:, :, 2] >= 15)
            & (rgb[:, :, 1] >= rgb[:, :, 0] - 2)
            & (rgb[:, :, 1] >= rgb[:, :, 2] - 2)
        ).astype(np.uint8)
        * 255
    )
    relaxed_seed = _clean_binary_mask(relaxed_seed)
    if cv2.countNonZero(relaxed_seed) > 0:
        candidate_masks.append(relaxed_seed)

    if not candidate_masks:
        return np.zeros_like(gray, dtype=np.uint8)

    scored_masks = []
    for seed in candidate_masks:
        contours, _ = cv2.findContours(seed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area <= 0:
            continue

        mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)
        x, y, w, h = cv2.boundingRect(largest)
        coverage = area / float(mask.shape[0] * mask.shape[1])
        bbox_fill = area / float(max(1, w * h))
        perimeter = cv2.arcLength(largest, True)
        compactness = (4.0 * np.pi * area / (perimeter * perimeter)) if perimeter > 0 else 0.0

        score = float(area)
        score *= 1.4 if 0.01 <= coverage <= 0.5 else 0.5
        score *= 1.3 if 0.15 <= bbox_fill <= 0.98 else 0.7
        score *= 1.15 if compactness >= 0.04 else 0.85
        scored_masks.append((score, mask))

    if not scored_masks:
        return np.zeros_like(gray, dtype=np.uint8)

    scored_masks.sort(key=lambda item: item[0], reverse=True)
    return _clean_binary_mask(scored_masks[0][1])


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
        raise ValueError("Mask does not contain any leaf pixels.")

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
    """Extract a 132D color feature vector from an image."""
    image_rgb = _ensure_rgb(image)
    mask = _largest_leaf_mask(image_rgb)
    final_image, final_mask = _square_canvas(image_rgb, mask)
    color_feats = compute_color_features(final_image, final_mask)
    hist = np.array(color_feats["RGB_Histogram_126D"], dtype=np.float32).reshape(3, -1).T
    hist_rows = [[int(i), float(hist[i, 0]), float(hist[i, 1]), float(hist[i, 2])] for i in range(hist.shape[0])]
    color_data_object = {
        "R_mean": color_feats["R_mean"],
        "G_mean": color_feats["G_mean"],
        "B_mean": color_feats["B_mean"],
        "R_std": color_feats["R_std"],
        "G_std": color_feats["G_std"],
        "B_std": color_feats["B_std"],
        "RGB_histogram_42x3": hist_rows,
    }
    stats = [
        color_data_object["R_mean"],
        color_data_object["G_mean"],
        color_data_object["B_mean"],
        color_data_object["R_std"],
        color_data_object["G_std"],
        color_data_object["B_std"],
    ]
    hist_values = np.array(color_data_object["RGB_histogram_42x3"], dtype=np.float32)[:, 1:].flatten().tolist()
    return stats + hist_values