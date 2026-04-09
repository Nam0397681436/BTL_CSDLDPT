import cv2 as cv
import numpy as np
from model.Image import Image
import os
import hashlib

def main():
    # quét lấy hết các đường dẫn file ảnh
    image_folder = "./data"
    image_paths = []

    for root, dirs, files in os.walk(image_folder):
        for file in files:
            if file.endswith((".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")):
                image_paths.append(os.path.join(root, file))

    # 1. Khởi tạo các kết nối DAO trước
    from dao.DAOPostgresql import DAOPostgresql
    from dao.DAOMinio import DAOMinio
    import gc

    try:
        pg_conn_str = os.getenv("POSTGRES_CONNECTION_STRING", "postgresql://admin:admin123@localhost:5432/mydb")
        dao_postgresql = DAOPostgresql(pg_conn_str)
        dao_postgresql.connect()

        minio_endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9000")
        dao_minio = DAOMinio(minio_endpoint, bucket_name="plantsimage")
        dao_minio.connect()
    except Exception as e:
        print(f"Lỗi khởi tạo kết nối: {e}")
        return

    BATCH_SIZE = 50
    total_images = len(image_paths)
    print(f"Tổng số ảnh tìm thấy: {total_images}")
    print(f"Kích thước mỗi batch xử lý: {BATCH_SIZE}")

    # 2. Xử lý theo từng batch
    for i in range(0, total_images, BATCH_SIZE):
        batch_paths = image_paths[i:i + BATCH_SIZE]
        batch_data = []
        current_batch_num = i // BATCH_SIZE + 1
        total_batches = (total_images + BATCH_SIZE - 1) // BATCH_SIZE
        
        print(f"\n--- Đang xử lý Batch {current_batch_num}/{total_batches} (Ảnh {i+1} đến {min(i + BATCH_SIZE, total_images)}) ---")

        # Bước A: Trích xuất đặc trưng cho Batch
        for path in batch_paths:
            try:
                img_instance = Image(path)
                img_instance.preprocess()
                metadata_features = img_instance.ExtractFeatures()
                
                metadata_basic = {
                    "image_id": img_instance.image_id,
                    "original_filename": img_instance.source_path,
                    "minio_url": img_instance.url_minio,
                    "category": img_instance.category,
                    "description": f"Ảnh lá {img_instance.category}"       
                }           
                
                batch_data.append({
                    "instance": img_instance,
                    "basic": metadata_basic,
                    "features": metadata_features
                })
            except Exception as e:
                print(f"Lỗi khi xử lý ảnh {path}: {e}")

        # Bước B: Insert Metadata vào PostgreSQL cho Batch
        print(f"Đang lưu metadata cho Batch {current_batch_num} vào PostgreSQL...")
        for item in batch_data:
            try:
                dao_postgresql.insert_image_metadata(item["basic"], item["features"])
            except Exception as e:
                print(f"Lỗi khi lưu DB cho {item['basic']['original_filename']}: {e}")

        # Bước C: Upload ảnh vào MinIO cho Batch
        print(f"Đang upload ảnh cho Batch {current_batch_num} lên MinIO...")
        for item in batch_data:
            try:
                img = item["instance"]
                image_data = img.get_storage_image_bytes()
                if image_data:
                    dao_minio.upload_image_bytes(image_data, img.object_name)
                else:
                    print(f"Lỗi resize ảnh để upload: {img.source_path}")
            except Exception as e:
                print(f"Lỗi khi upload MinIO cho {img.source_path}: {e}")

        # Bước D: Giải phóng bộ nhớ của Batch hiện tại
        batch_data.clear()
        gc.collect() # Ép buộc thu gom rác để giải phóng RAM ngay lập tức
        print(f"Hoàn thành Batch {current_batch_num}. Đã giải phóng bộ nhớ.")

    # 3. Đóng kết nối
    dao_postgresql.close()
    print("\n=== ĐÃ HOÀN THÀNH TẤT CẢ ===")

if __name__ == "__main__":
    main()
