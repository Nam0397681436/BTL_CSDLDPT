import cv2
import numpy as np
from utils.rotateLeave import ensure_rgb, largest_leaf_mask, crop_leaf_region, clean_binary_mask

# ---------- Các hàm đặc thù texture ----------
def _uniform_lbp(gray, radius=1):
    if radius != 1:
        raise ValueError("This implementation currently supports radius=1 only.")
    padded = cv2.copyMakeBorder(gray, radius, radius, radius, radius, cv2.BORDER_REFLECT)
    center = gray.astype(np.uint8)

    offsets = [(-1,-1), (-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1)]
    neighbor_bits = []
    for dy, dx in offsets:
        neighbor = padded[radius+dy:radius+dy+gray.shape[0],
                         radius+dx:radius+dx+gray.shape[1]]
        neighbor_bits.append((neighbor >= center).astype(np.uint8))

    binary_code = np.zeros_like(gray, dtype=np.uint8)
    for bit_index, bit_plane in enumerate(neighbor_bits):
        binary_code |= (bit_plane << bit_index)

    transitions = np.zeros_like(gray, dtype=np.uint8)
    for idx in range(len(neighbor_bits)):
        transitions += (neighbor_bits[idx] != neighbor_bits[(idx+1)%8]).astype(np.uint8)

    bitcount_lookup = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    uniform_patterns = transitions <= 2
    lbp = np.where(uniform_patterns, bitcount_lookup[binary_code], 9).astype(np.uint8)
    return lbp, 10

def _glcm_features(gray, mask, levels=32):
    if levels < 2:
        raise ValueError("levels must be at least 2")
    masked_gray = np.where(mask > 0, gray, 0).astype(np.uint16)
    quantized = (masked_gray * levels) // 256
    quantized = np.clip(quantized, 0, levels-1).astype(np.uint8)

    valid_left = (mask[:, :-1] > 0) & (mask[:, 1:] > 0)
    if not np.any(valid_left):
        return 0.0, 0.0, 0.0, 0.0

    row = quantized[:, :-1][valid_left]
    col = quantized[:, 1:][valid_left]
    glcm = np.zeros((levels, levels), dtype=np.float64)
    np.add.at(glcm, (row, col), 1)
    np.add.at(glcm, (col, row), 1)
    glcm /= glcm.sum()

    i = np.arange(levels, dtype=np.float64)
    j = i[:, None]
    contrast = float(np.sum(((j - i)**2) * glcm))
    energy = float(np.sum(glcm**2))
    mean_i = float(np.sum(i * glcm.sum(axis=1)))
    mean_j = float(np.sum(i * glcm.sum(axis=0)))
    std_i = float(np.sqrt(np.sum(((i - mean_i)**2) * glcm.sum(axis=1))))
    std_j = float(np.sqrt(np.sum(((i - mean_j)**2) * glcm.sum(axis=0))))
    correlation = 0.0
    if std_i > 0 and std_j > 0:
        correlation = float(np.sum(((j - mean_i) * (i - mean_j)) * glcm) / (std_i * std_j))
    entropy = float(-np.sum(glcm * np.log2(glcm + 1e-12)))
    return contrast, energy, correlation, entropy

def compute_texture_features(gray, mask):
    gray = gray.astype(np.uint8)
    if mask is None:
        mask = np.full(gray.shape, 255, dtype=np.uint8)
    else:
        mask = (mask > 0).astype(np.uint8) * 255
    if cv2.countNonZero(mask) == 0:
        mask = np.full(mask.shape, 255, dtype=np.uint8)

    masked_gray = np.where(mask > 0, gray, 128).astype(np.uint8)
    lbp, n_bins = _uniform_lbp(masked_gray, radius=1)
    lbp_vals = lbp[mask > 0] if np.any(mask > 0) else lbp.reshape(-1)
    hist, _ = np.histogram(lbp_vals, bins=n_bins, range=(0, n_bins), density=True)

    contrast, energy, correlation, entropy = _glcm_features(gray, mask)
    return {
        "LBP_histogram": hist.tolist(),
        "GLCM_Contrast": contrast,
        "GLCM_Energy": energy,
        "GLCM_Correlation": correlation,
        "GLCM_Entropy": entropy,
    }

def extract_texture_vector(image):
    """Vector 14 chiều: 4 GLCM + 10 histogram LBP."""
    image_rgb = ensure_rgb(image)
    raw_mask = largest_leaf_mask(image_rgb)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cropped_bgr, cropped_mask = crop_leaf_region(image_bgr, raw_mask, target_size=(256, 256))

    # Làm sạch mask thêm lần nữa (giữ nguyên logic cũ)
    cropped_mask = (cropped_mask > 0).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cropped_mask = cv2.morphologyEx(cropped_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2GRAY)

    tex = compute_texture_features(gray, cropped_mask)
    glcm_part = [tex["GLCM_Contrast"], tex["GLCM_Energy"], tex["GLCM_Correlation"], tex["GLCM_Entropy"]]
    return glcm_part + tex["LBP_histogram"]