import cv2
import numpy as np
from pathlib import Path
import sys

# Thêm thư mục src vào sys.path để import các service
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from services.ExtractFeatureImage import extract_feature_HOG, detect_leaf_contour

def visualize_hog_with_mask(image_path):
    # 1. Đọc ảnh
    img_np = cv2.imread(str(image_path))
    if img_np is None:
        print(f"Error: Could not read image at {image_path}")
        return

    # 2. Tạo Mask và Masked Image (giống logic trong extract_feature_HOG)
    leaf_contour = detect_leaf_contour(img_np)
    mask = np.zeros(img_np.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [leaf_contour], -1, 255, thickness=-1)
    img_masked = cv2.bitwise_and(img_np, img_np, mask=mask)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
    # Preprocessing cho HOG
    gray_masked = cv2.cvtColor(img_masked, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(gray_masked, (160, 128))

    # 3. Tính toán HOG hướng (Manual Visualization)
    gx = cv2.Sobel(img_resized, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(img_resized, cv2.CV_32F, 0, 1, ksize=1)
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

    # 4. Vẽ trực quan
    scale = 4
    h_small, w_small = img_resized.shape
    vis_h, vis_w = h_small * scale, w_small * scale
    
    # Ảnh nền đen để vẽ gradients
    hog_vis = np.zeros((vis_h, vis_w, 3), dtype=np.uint8)
    # Ảnh gốc đã resize và xám để so sánh
    img_large = cv2.resize(cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR), (vis_w, vis_h))
    
    for y in range(0, h_small, 8):
        for x in range(0, w_small, 8):
            cell_mag = mag[y:y+8, x:x+8]
            cell_angle = angle[y:y+8, x:x+8]
            
            # Chỉ vẽ nếu có gradient (vùng có chi tiết/gân)
            if np.max(cell_mag) > 10:
                # Lấy hướng tại điểm có gradient mạnh nhất trong cell
                idx = np.unravel_index(np.argmax(cell_mag), cell_mag.shape)
                best_angle = cell_angle[idx]
                
                # Vẽ đoạn thẳng vuông góc với gradient (song song với cạnh/gân)
                rad = np.deg2rad(best_angle + 90)
                cx, cy = (x + 4) * scale, (y + 4) * scale
                dx, dy = int(12 * np.cos(rad)), int(12 * np.sin(rad))
                
                cv2.line(hog_vis, (cx - dx, cy - dy), (cx + dx, cy + dy), (0, 255, 255), 1)

    # 5. Kết hợp các bước để giải thích
    # Hàng 1: Gốc | Mask | Masked
    img_orig_small = cv2.resize(img_np, (vis_w, vis_h))
    mask_vis = cv2.resize(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), (vis_w, vis_h))
    img_masked_vis = cv2.resize(img_masked, (vis_w, vis_h))
    top_row = np.hstack([img_orig_small, mask_vis, img_masked_vis])
    
    # Hàng 2: Trống | Gray Resized | HOG Gradients
    empty = np.zeros_like(img_large)
    bottom_row = np.hstack([empty, img_large, hog_vis])
    
    combined = np.vstack([top_row, bottom_row])
    
    output_path = PROJECT_ROOT / "test" / "output" / "hog_masked_vis.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), combined)
    
    print("-" * 60)
    print(f"HOG Masking Test Result:")
    print(f"1. Top-Left: Original Image")
    print(f"2. Top-Center: Leaf Mask (detect_leaf_contour)")
    print(f"3. Top-Right: Masked Image (Background items removed)")
    print(f"4. Bottom-Center: Gray Resized (160x128)")
    print(f"5. Bottom-Right: HOG Orientations (Only on leaf)")
    print(f"Output saved to: {output_path}")
    print("-" * 60)

if __name__ == "__main__":
    sample_image = PROJECT_ROOT / "data/Basil/healthy/0008_0001.JPG"
    if not sample_image.exists():
        import glob
        files = glob.glob(str(PROJECT_ROOT / "data/**/*.JPG"), recursive=True)
        if files: sample_image = Path(files[0])
    
    if sample_image.exists():
        visualize_hog_with_mask(sample_image)
    else:
        print("Test image not found.")
