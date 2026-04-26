"""
EvaluateFeatures.py
-------------------
Script đánh giá độ hiệu quả của từng loại đặc trưng (Feature Importance / Ablation Study)
dựa trên độ đo Precision@K.

Ý tưởng:
1. Lấy ngẫu nhiên N ảnh từ CSDL làm các ảnh truy vấn (query images).
2. Thiết lập các cấu hình trọng số (weights) khác nhau: 
   - Chỉ dùng 1 loại đặc trưng (ví dụ: chỉ HOG, chỉ Shape...).
   - Bỏ đi 1 loại đặc trưng (Leave-one-out) để xem độ chính xác giảm bao nhiêu.
3. Chạy tìm kiếm top-K ảnh gần nhất với từng cấu hình.
4. Tính Precision@K: Tỷ lệ ảnh trong top-K có cùng 'category' với ảnh truy vấn.
"""

import os
import sys
from pathlib import Path
import random

# Đảm bảo import được package `src`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
for p in (str(PROJECT_ROOT), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.dao.DAOPostgresql import DAOPostgresql
from src.services.ExtractImageDBWeighted import ExtractImageDBWeighted

# Cấu hình đánh giá
NUM_QUERIES = 20  # Số lượng ảnh lấy ngẫu nhiên làm truy vấn
TOP_K = 10        # Tìm top 10 ảnh gần nhất (Precision@10)

def main():
    print(f"=== ĐÁNH GIÁ ĐỘ HIỆU QUẢ CỦA CÁC ĐẶC TRƯNG (Ablation Study) ===")
    print(f"Sử dụng {NUM_QUERIES} ảnh ngẫu nhiên làm truy vấn. Đo Precision@{TOP_K}\n")

    pg_conn_str = os.getenv("POSTGRES_CONNECTION_STRING", "postgresql://admin:admin123@localhost:5432/mydb")
    dao_postgresql = DAOPostgresql(pg_conn_str)
    
    try:
        dao_postgresql.connect()
    except Exception as e:
        print(f"Lỗi kết nối CSDL: {e}")
        return

    # 1. Lấy tất cả ảnh và đặc trưng từ CSDL để làm tập chọn ngẫu nhiên
    all_images = []
    with dao_postgresql.connection.cursor() as cursor:
        cursor.execute('''
            SELECT m.image_id, m.category, f.color_vector, f.texture_vector, 
                   f.hog_vector, f.shape_vector
            FROM "Basic_Metadata" m
            JOIN "Images_Features" f ON m.image_id = f.image_id
            WHERE m.category IS NOT NULL AND m.category != ''
        ''')
        rows = cursor.fetchall()
        for r in rows:
            all_images.append({
                "image_id": r[0],
                "category": r[1],
                "color": r[2],
                "texture": r[3],
                "hog": r[4],
                "shape": r[5],
                "venation": r[6]
            })

    if len(all_images) < NUM_QUERIES:
        print("Không đủ ảnh trong CSDL để test.")
        return

    # Lấy ngẫu nhiên N ảnh làm queries
    queries = random.sample(all_images, NUM_QUERIES)

    # 2. Định nghĩa các cấu hình trọng số để test
    test_configs = {
        "ALL_FEATURES (Base)": {"hog": 0.40, "shape": 0.25, "color": 0.20, "texture": 0.10, "venation": 0.05},
        
        # Chỉ dùng 1 feature (Single Feature Performance)
        "ONLY_HOG":      {"hog": 1.0, "shape": 0.0, "color": 0.0, "texture": 0.0, "venation": 0.0},
        "ONLY_SHAPE":    {"hog": 0.0, "shape": 1.0, "color": 0.0, "texture": 0.0, "venation": 0.0},
        "ONLY_COLOR":    {"hog": 0.0, "shape": 0.0, "color": 1.0, "texture": 0.0, "venation": 0.0},
        "ONLY_TEXTURE":  {"hog": 0.0, "shape": 0.0, "color": 0.0, "texture": 1.0, "venation": 0.0},
        "ONLY_VENATION": {"hog": 0.0, "shape": 0.0, "color": 0.0, "texture": 0.0, "venation": 1.0},

        # Bỏ đi 1 feature (Leave-one-out)
        "NO_HOG":      {"hog": 0.0, "shape": 0.35, "color": 0.35, "texture": 0.15, "venation": 0.15},
        "NO_SHAPE":    {"hog": 0.50, "shape": 0.0, "color": 0.25, "texture": 0.15, "venation": 0.10},
        "NO_COLOR":    {"hog": 0.50, "shape": 0.30, "color": 0.0, "texture": 0.10, "venation": 0.10},
        "NO_TEXTURE":  {"hog": 0.45, "shape": 0.25, "color": 0.25, "texture": 0.0, "venation": 0.05},
        "NO_VENATION": {"hog": 0.45, "shape": 0.25, "color": 0.20, "texture": 0.10, "venation": 0.0},

        # Combo tốt nhất gợi ý
        "HOG+TEXTURE": {"hog": 0.60, "shape": 0.0, "color": 0.0, "texture": 0.40, "venation": 0.0},
        "HOG+COLOR":   {"hog": 0.60, "shape": 0.0, "color": 0.40, "texture": 0.0, "venation": 0.0},
    }

    # Helper function để lấy category của 1 image_id nhanh
    def get_category_by_id(img_id):
        return next((img["category"] for img in all_images if img["image_id"] == img_id), None)

    # 3. Chạy từng cấu hình và đánh giá
    print(f"{'Cấu hình đánh giá':<25} | {'Mean Precision@' + str(TOP_K):<20} | {'Đánh giá'}")
    print("-" * 75)

    base_precision = 0.0

    for config_name, weights in test_configs.items():
        search_engine = ExtractImageDBWeighted(dao_minio=None, dao_postgresql=dao_postgresql, custom_weights=weights)
        
        total_precision = 0.0

        for q in queries:
            # Tham số query phải đủ dạng dictionary mà extract_image_postgresql mong muốn
            feature_query = {
                "color": q["color"], "texture": q["texture"], 
                "hog": q["hog"], "shape": q["shape"], "venation": q["venation"]
            }

            # Lấy top_k + 1 kết quả (do kết quả đầu tiên thường chính là ảnh truy vấn đó)
            top_results = search_engine.extract_image_postgresql(feature_query, top_k=TOP_K + 1)
            
            # Loại bỏ ảnh tự tìm thấy chính nó
            retrieved_ids = [res[1] for res in top_results if res[1] != q["image_id"]]
            retrieved_ids = retrieved_ids[:TOP_K] # Lấy đúng top K

            # Đếm số lượng ảnh cùng category
            true_category = q["category"]
            correct_count = sum(1 for ret_id in retrieved_ids if get_category_by_id(ret_id) == true_category)
            
            # Tính precision = đúng / tổng số trả về (hoặc TOP_K)
            precision = correct_count / TOP_K if TOP_K > 0 else 0
            total_precision += precision

        mean_precision = total_precision / NUM_QUERIES

        # Lưu lại precision của ALL_FEATURES để so sánh
        if config_name == "ALL_FEATURES (Base)":
            base_precision = mean_precision
            eval_text = "Baseline (Mức cơ sở)"
        else:
            diff = mean_precision - base_precision
            eval_text = f"{diff:+.3f} so với Base"
            if diff > 0.02:
                eval_text += " (👍 Tốt hơn)"
            elif diff < -0.05:
                eval_text += " (📉 Quan trọng, vì bỏ đi làm giảm độ chính xác)"

        print(f"{config_name:<25} | {mean_precision:20.4f} | {eval_text}")

if __name__ == "__main__":
    main()
