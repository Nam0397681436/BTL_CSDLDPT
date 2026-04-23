import cv2
import numpy as np

from .leaf_segmentation import extract_leaf_mask, crop_leaf_region


def _ensure_rgb(image):
    if image is None:
        raise ValueError("Input image is empty.")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    return image.copy()


def _clean_mask(mask):
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


def _largest_leaf_mask(image_rgb):
    image_rgb = _ensure_rgb(image_rgb)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    return _clean_mask(extract_leaf_mask(image_bgr))


def _zhang_suen_thinning(binary_image):
    image = np.pad((binary_image > 0).astype(np.uint8), 1, mode="constant")

    def neighbors(pixel_image):
        p2 = np.roll(pixel_image, -1, axis=0)
        p3 = np.roll(np.roll(pixel_image, -1, axis=0), -1, axis=1)
        p4 = np.roll(pixel_image, -1, axis=1)
        p5 = np.roll(np.roll(pixel_image, 1, axis=0), -1, axis=1)
        p6 = np.roll(pixel_image, 1, axis=0)
        p7 = np.roll(np.roll(pixel_image, 1, axis=0), 1, axis=1)
        p8 = np.roll(pixel_image, 1, axis=1)
        p9 = np.roll(np.roll(pixel_image, -1, axis=0), 1, axis=1)
        return p2, p3, p4, p5, p6, p7, p8, p9

    changing = True
    while changing:
        changing = False
        p2, p3, p4, p5, p6, p7, p8, p9 = neighbors(image)
        neighbor_count = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
        transitions = (
            (p2 == 0) & (p3 == 1)
        ).astype(np.uint8) + (
            (p3 == 0) & (p4 == 1)
        ).astype(np.uint8) + (
            (p4 == 0) & (p5 == 1)
        ).astype(np.uint8) + (
            (p5 == 0) & (p6 == 1)
        ).astype(np.uint8) + (
            (p6 == 0) & (p7 == 1)
        ).astype(np.uint8) + (
            (p7 == 0) & (p8 == 1)
        ).astype(np.uint8) + (
            (p8 == 0) & (p9 == 1)
        ).astype(np.uint8) + (
            (p9 == 0) & (p2 == 1)
        ).astype(np.uint8)

        marker = (
            (image == 1)
            & (neighbor_count >= 2)
            & (neighbor_count <= 6)
            & (transitions == 1)
            & ((p2 * p4 * p6) == 0)
            & ((p4 * p6 * p8) == 0)
        )
        if np.any(marker):
            image[marker] = 0
            changing = True

        p2, p3, p4, p5, p6, p7, p8, p9 = neighbors(image)
        neighbor_count = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
        transitions = (
            (p2 == 0) & (p3 == 1)
        ).astype(np.uint8) + (
            (p3 == 0) & (p4 == 1)
        ).astype(np.uint8) + (
            (p4 == 0) & (p5 == 1)
        ).astype(np.uint8) + (
            (p5 == 0) & (p6 == 1)
        ).astype(np.uint8) + (
            (p6 == 0) & (p7 == 1)
        ).astype(np.uint8) + (
            (p7 == 0) & (p8 == 1)
        ).astype(np.uint8) + (
            (p8 == 0) & (p9 == 1)
        ).astype(np.uint8) + (
            (p9 == 0) & (p2 == 1)
        ).astype(np.uint8)

        marker = (
            (image == 1)
            & (neighbor_count >= 2)
            & (neighbor_count <= 6)
            & (transitions == 1)
            & ((p2 * p4 * p8) == 0)
            & ((p2 * p6 * p8) == 0)
        )
        if np.any(marker):
            image[marker] = 0
            changing = True

    return (image[1:-1, 1:-1].astype(np.uint8) * 255)


def _count_branch_points(skeleton):
    skeleton_mask = skeleton > 0
    branch_points = 0

    for y in range(1, skeleton.shape[0] - 1):
        for x in range(1, skeleton.shape[1] - 1):
            if not skeleton_mask[y, x]:
                continue

            neighborhood = skeleton_mask[y - 1:y + 2, x - 1:x + 2]
            neighbors = int(np.sum(neighborhood)) - 1
            if neighbors > 2:
                branch_points += 1

    return branch_points


def _count_end_points(skeleton):
    skeleton_mask = skeleton > 0
    end_points = 0

    for y in range(1, skeleton.shape[0] - 1):
        for x in range(1, skeleton.shape[1] - 1):
            if not skeleton_mask[y, x]:
                continue

            neighborhood = skeleton_mask[y - 1:y + 2, x - 1:x + 2]
            neighbors = int(np.sum(neighborhood)) - 1
            if neighbors == 1:
                end_points += 1

    return end_points


def _count_connected_components(binary_image):
    binary = (binary_image > 0).astype(np.uint8)
    if np.count_nonzero(binary) == 0:
        return 0

    component_count, _ = cv2.connectedComponents(binary)
    return max(0, int(component_count - 1))


def _fractal_dimension(binary_image):
    binary = (binary_image > 0)
    if not np.any(binary):
        return 0.0

    h, w = binary.shape
    max_power = int(np.floor(np.log2(min(h, w))))
    if max_power < 1:
        return 0.0

    sizes = [2 ** p for p in range(1, max_power + 1)]
    counts = []
    valid_sizes = []

    for size in sizes:
        trimmed_h = (h // size) * size
        trimmed_w = (w // size) * size
        if trimmed_h == 0 or trimmed_w == 0:
            continue

        cropped = binary[:trimmed_h, :trimmed_w]
        blocks = cropped.reshape(trimmed_h // size, size, trimmed_w // size, size)
        occupied = np.any(blocks, axis=(1, 3))
        count = int(np.count_nonzero(occupied))

        if count > 0:
            counts.append(count)
            valid_sizes.append(size)

    if len(counts) < 2:
        return 0.0

    log_inv_size = np.log(1.0 / np.array(valid_sizes, dtype=np.float64))
    log_count = np.log(np.array(counts, dtype=np.float64))
    slope, _ = np.polyfit(log_inv_size, log_count, 1)
    return float(max(0.0, slope))


def extract_venation_vector(image):
    """Extract venation feature vector (10D) for detailed vein structure analysis."""
    image_rgb = _ensure_rgb(image)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    raw_mask = _largest_leaf_mask(image_rgb)
    
    cropped_bgr, leaf_mask = crop_leaf_region(image_bgr, raw_mask, target_size=(256, 256))
    cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2GRAY)

    # 1. Thay Gaussian Blur bằng Bilateral Filter để khử nhiễu nhưng GIỮ LẠI biên (gân lá)
    blurred = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)

    # 2. Tăng cường độ tương phản cục bộ (clipLimit từ 2.0 -> 3.0)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)

    # 3. Thu nhỏ kích thước kernel để bắt các gân mảnh hơn
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    blackhat = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, kernel_small)
    tophat = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, kernel_small)
    
    # 4. Hạ ngưỡng Canny để bắt được các đường gân mờ (từ 25, 90 xuống 15, 50)
    edges = cv2.Canny(enhanced, 15, 50)

    # 5. Tăng trọng số làm nổi bật gân từ Blackhat và Tophat
    vein_enhanced = cv2.addWeighted(blackhat, 0.7, tophat, 0.3, 0)
    vein_enhanced = cv2.addWeighted(vein_enhanced, 1.0, edges, 0.5, 0)
    vein_enhanced = cv2.bitwise_and(vein_enhanced, vein_enhanced, mask=leaf_mask)

    # 6. Thay thế Otsu bằng Adaptive Threshold để không sót gân ở vùng sáng/tối không đều
    vein_binary = cv2.adaptiveThreshold(
        vein_enhanced, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        15, -3  # Block size 15, hằng số C=-3 giúp giữ lại các đường mờ
    )

    # Lọc bớt nhiễu hạt tiêu sau khi dùng Adaptive Threshold
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    vein_binary = cv2.morphologyEx(vein_binary, cv2.MORPH_OPEN, kernel_clean, iterations=1)

    vein_binary = cv2.bitwise_and(vein_binary, vein_binary, mask=leaf_mask)

    # Rút trích khung xương (skeleton)
    skeleton = _zhang_suen_thinning(vein_binary)
    skeleton = cv2.bitwise_and(skeleton, skeleton, mask=leaf_mask)

    # Tính toán đặc trưng
    vein_length = int(np.count_nonzero(skeleton))
    vein_area = int(np.count_nonzero(vein_binary))
    leaf_area = max(1, int(np.count_nonzero(leaf_mask)))
    vein_density = float(vein_length / leaf_area)
    branch_points = int(_count_branch_points(skeleton))
    end_points = int(_count_end_points(skeleton))
    component_count = int(_count_connected_components(skeleton))
    fractal_dimension = float(_fractal_dimension(skeleton))

    branch_density = float(branch_points / leaf_area)
    end_point_density = float(end_points / leaf_area)
    avg_vein_thickness = float(vein_area / max(1, vein_length))
    branch_end_ratio = float(branch_points / max(1, end_points))
    complexity_index = float((branch_points + end_points) / max(1, vein_length))

    vein_vector = [
        float(vein_length),
        float(branch_points),
        float(vein_density),
        float(end_points),
        float(branch_density),
        float(end_point_density),
        float(avg_vein_thickness),
        float(component_count),
        float(fractal_dimension),
        float(branch_end_ratio),
        float(complexity_index)
    ]
    
    return vein_vector