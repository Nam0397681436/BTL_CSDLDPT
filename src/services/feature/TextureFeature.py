import cv2
import numpy as np


def _uniform_lbp(gray, radius=1):
    if radius != 1:
        raise ValueError("This implementation currently supports radius=1 only.")

    padded = cv2.copyMakeBorder(gray, radius, radius, radius, radius, cv2.BORDER_REFLECT)
    center = gray.astype(np.uint8)

    offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, 1), (1, 1), (1, 0),
        (1, -1), (0, -1),
    ]

    neighbor_bits = []
    for dy, dx in offsets:
        neighbor = padded[radius + dy:radius + dy + gray.shape[0], radius + dx:radius + dx + gray.shape[1]]
        neighbor_bits.append((neighbor >= center).astype(np.uint8))

    binary_code = np.zeros_like(gray, dtype=np.uint8)
    for bit_index, bit_plane in enumerate(neighbor_bits):
        binary_code |= (bit_plane << bit_index)

    transitions = np.zeros_like(gray, dtype=np.uint8)
    for index in range(len(neighbor_bits)):
        current_bit = neighbor_bits[index]
        next_bit = neighbor_bits[(index + 1) % len(neighbor_bits)]
        transitions += (current_bit != next_bit).astype(np.uint8)

    bitcount_lookup = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    uniform_patterns = transitions <= 2
    lbp = np.where(uniform_patterns, bitcount_lookup[binary_code], 9).astype(np.uint8)
    return lbp, 10


def _glcm_features(gray, mask, levels=32):
    if levels < 2:
        raise ValueError("levels must be at least 2")

    quantized = (gray.astype(np.uint16) * levels) // 256
    quantized = np.clip(quantized, 0, levels - 1).astype(np.uint8)

    valid_left = mask[:, :-1] > 0
    valid_right = mask[:, 1:] > 0
    valid_pairs = valid_left & valid_right

    if not np.any(valid_pairs):
        return 0.0, 0.0, 0.0, 0.0

    row_values = quantized[:, :-1][valid_pairs]
    col_values = quantized[:, 1:][valid_pairs]

    glcm = np.zeros((levels, levels), dtype=np.float64)
    np.add.at(glcm, (row_values, col_values), 1)
    np.add.at(glcm, (col_values, row_values), 1)

    total = glcm.sum()
    if total == 0:
        return 0.0, 0.0, 0.0, 0.0

    glcm /= total

    i = np.arange(levels, dtype=np.float64)
    j = i[:, None]
    contrast = float(np.sum(((j - i) ** 2) * glcm))
    energy = float(np.sum(glcm ** 2))

    mean_i = float(np.sum(i * glcm.sum(axis=1)))
    mean_j = float(np.sum(i * glcm.sum(axis=0)))
    std_i = float(np.sqrt(np.sum(((i - mean_i) ** 2) * glcm.sum(axis=1))))
    std_j = float(np.sqrt(np.sum(((i - mean_j) ** 2) * glcm.sum(axis=0))))

    if std_i == 0.0 or std_j == 0.0:
        correlation = 0.0
    else:
        correlation = float(
            np.sum(((j - mean_i) * (i - mean_j)) * glcm) / (std_i * std_j)
        )

    entropy = float(-np.sum(glcm * np.log2(glcm + 1e-12)))
    return contrast, energy, correlation, entropy


def compute_texture_features(gray, mask=None, lbp_P=8, lbp_R=1):
    gray = gray.astype(np.uint8)

    if mask is None:
        mask = np.full(gray.shape, 255, dtype=np.uint8)
    else:
        mask = mask.astype(np.uint8)

    if mask.ndim != 2:
        raise ValueError("Mask must be a single-channel image.")

    if cv2.countNonZero(mask) == 0:
        mask = np.full(mask.shape, 255, dtype=np.uint8)

    lbp, n_bins = _uniform_lbp(gray, radius=lbp_R)
    lbp_values = lbp[mask > 0]
    if lbp_values.size == 0:
        lbp_values = lbp.reshape(-1)

    lbp_hist, _ = np.histogram(
        lbp_values,
        bins=n_bins,
        range=(0, n_bins),
        density=True,
    )

    contrast, energy, correlation, entropy = _glcm_features(gray, mask)

    return {
        "LBP_histogram": lbp_hist.tolist(),
        "GLCM_Contrast": float(contrast),
        "GLCM_Energy": float(energy),
        "GLCM_Correlation": float(correlation),
        "GLCM_Entropy": float(entropy),
    }


def extract_texture_vector(image):
    """Extract texture feature vector (14D) from image - GLCM (4D) + LBP (10D)."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    texture_data = compute_texture_features(gray)
    
    glcm_part = [
        texture_data["GLCM_Contrast"],
        texture_data["GLCM_Energy"],
        texture_data["GLCM_Correlation"],
        texture_data["GLCM_Entropy"]
    ]
    lbp_part = texture_data["LBP_histogram"]
    texture_vector = glcm_part + lbp_part
    
    return texture_vector