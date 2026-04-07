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
        candidates.append((area, mask))

    if not candidates:
        return np.full_like(gray, 255, dtype=np.uint8)

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


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


def extract_venation_vector(image):
    """Extract venation feature vector (3D) from image - vein length, branch points, density."""
    image_rgb = _ensure_rgb(image)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    leaf_mask = _largest_leaf_mask(image_rgb)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)

    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    blackhat = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, kernel_small)
    tophat = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, kernel_small)
    edges = cv2.Canny(enhanced, 25, 90)

    vein_enhanced = cv2.addWeighted(blackhat, 0.55, tophat, 0.25, 0)
    vein_enhanced = cv2.addWeighted(vein_enhanced, 1.0, edges, 0.35, 0)
    vein_enhanced = cv2.bitwise_and(vein_enhanced, vein_enhanced, mask=leaf_mask)

    _, vein_binary = cv2.threshold(vein_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    vein_binary = cv2.bitwise_and(vein_binary, vein_binary, mask=leaf_mask)

    skeleton = _zhang_suen_thinning(vein_binary)
    skeleton = cv2.bitwise_and(skeleton, skeleton, mask=leaf_mask)

    vein_length = int(np.count_nonzero(skeleton))
    leaf_area = max(1, int(np.count_nonzero(leaf_mask)))
    vein_density = float(vein_length / leaf_area)
    branch_points = int(_count_branch_points(skeleton))

    vein_vector = [
        float(vein_length),
        float(branch_points),
        float(vein_density)
    ]
    
    return vein_vector