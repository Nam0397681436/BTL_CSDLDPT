import cv2
import numpy as np
from utils.rotateLeave import ensure_rgb, largest_leaf_mask, crop_leaf_region, clean_binary_mask

# ---------- Các hàm giữ nguyên (đặc thù màu) ----------
def _seed_from_color(image_rgb):
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    rgb = image_rgb.astype(np.int16)

    green_hue = ((hsv[:, :, 0] >= 20) & (hsv[:, :, 0] <= 90)).astype(np.uint8) * 255
    green_saturation = (hsv[:, :, 1] >= 30).astype(np.uint8) * 255
    green_value = (hsv[:, :, 2] >= 20).astype(np.uint8) * 255
    green_dominance = ((rgb[:, :, 1] > rgb[:, :, 0] + 8) & (rgb[:, :, 1] > rgb[:, :, 2] + 8)).astype(np.uint8) * 255

    seed = cv2.bitwise_and(green_hue, green_saturation)
    seed = cv2.bitwise_and(seed, green_value)
    seed = cv2.bitwise_and(seed, green_dominance)
    return clean_binary_mask(seed)

def compute_color_features(image_rgb, mask, bins=42):
    image_rgb = ensure_rgb(image_rgb)
    mask = mask.astype(np.uint8)

    if cv2.countNonZero(mask) == 0:
        raise ValueError("Mask does not contain any leaf pixels.")

    channels = cv2.split(image_rgb)
    color_vec = []
    for ch in channels:
        hist = cv2.calcHist([ch], [0], mask, [bins], [0, 256]).flatten()
        hist = hist / (hist.sum() + 1e-7)
        color_vec.extend(hist.tolist())

    mean, std = cv2.meanStdDev(image_rgb, mask=mask)
    mean = mean.flatten()
    std = std.flatten()
    return {
        "RGB_Histogram_126D": color_vec,
        "R_mean": float(mean[0]), "G_mean": float(mean[1]), "B_mean": float(mean[2]),
        "R_std": float(std[0]), "G_std": float(std[1]), "B_std": float(std[2]),
    }

def extract_color_vector(image):
    """Vector 132 chiều: 6 thống kê + 126 histogram (42 bins * 3)."""
    image_rgb = ensure_rgb(image)
    raw_mask = largest_leaf_mask(image_rgb)
    # Dùng crop_leaf_region với target_size=512 để căn giữa (thay thế square_canvas)
    cropped_bgr, cropped_mask = crop_leaf_region(image_rgb, raw_mask, target_size=(512, 512))
    final_image = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
    final_mask = cropped_mask

    feats = compute_color_features(final_image, final_mask)
    stats = [feats["R_mean"], feats["G_mean"], feats["B_mean"],
             feats["R_std"], feats["G_std"], feats["B_std"]]
    # Histogram: list of 3*42 = 126 giá trị
    hist_values = feats["RGB_Histogram_126D"]
    return stats + hist_values