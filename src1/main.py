import numpy as np
from model.Image import Image
import os
import gc


FEATURE_KEYS = ("color", "texture", "hog", "shape")


def _to_1d_float32(vector_like) -> np.ndarray:
    if vector_like is None:
        return np.array([], dtype=np.float32)
    vector = np.asarray(vector_like, dtype=np.float32).flatten()
    if vector.size == 0:
        return np.array([], dtype=np.float32)
    return vector


def _fit_feature_stats(records):
    stats = {}
    for feature_key in FEATURE_KEYS:
        max_dim = max((item["raw_features"][feature_key].size for item in records), default=0)
        if max_dim == 0:
            stats[feature_key] = {
                "dim": 0,
                "mean": np.array([], dtype=np.float32),
                "std": np.array([], dtype=np.float32),
            }
            continue

        matrix = np.zeros((len(records), max_dim), dtype=np.float32)
        for row_index, item in enumerate(records):
            vector = item["raw_features"][feature_key]
            if vector.size == 0:
                continue
            limit = min(vector.size, max_dim)
            matrix[row_index, :limit] = vector[:limit]

        mean = matrix.mean(axis=0).astype(np.float32)
        std = matrix.std(axis=0).astype(np.float32)
        std = np.where(std < 1e-8, 1.0, std).astype(np.float32)

        stats[feature_key] = {"dim": max_dim, "mean": mean, "std": std}
    return stats


def _normalize_vector(vector: np.ndarray, feature_stat: dict) -> np.ndarray:
    dim = feature_stat["dim"]
    if dim == 0:
        return np.array([], dtype=np.float32)

    padded = np.zeros(dim, dtype=np.float32)
    if vector.size:
        limit = min(vector.size, dim)
        padded[:limit] = vector[:limit]

    return ((padded - feature_stat["mean"]) / feature_stat["std"]).astype(np.float32)

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

    # 2. Trích xuất đặc trưng thô cho toàn bộ ảnh
    all_items = []
    print("\n=== Bắt đầu trích xuất đặc trưng thô ===")
    for i in range(0, total_images, BATCH_SIZE):
        batch_paths = image_paths[i:i + BATCH_SIZE]
        current_batch_num = i // BATCH_SIZE + 1
        total_batches = (total_images + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"\n--- Trích xuất Batch {current_batch_num}/{total_batches} (Ảnh {i+1} đến {min(i + BATCH_SIZE, total_images)}) ---")

        for path in batch_paths:
            try:
                img_instance = Image(path)
                img_instance.preprocess()  # resize cho bước trích đặc trưng
                metadata_features = img_instance.ExtractFeatures()

                metadata_basic = {
                    "image_id": img_instance.image_id,
                    "original_filename": img_instance.source_path,
                    "minio_url": img_instance.url_minio,
                    "category": img_instance.category,
                    "description": f"Ảnh lá {img_instance.category}",
                }

                all_items.append(
                    {
                        "basic": metadata_basic,
                        "object_name": img_instance.object_name,
                        "raw_features": {
                            "color": _to_1d_float32(metadata_features.get("color")),
                            "texture": _to_1d_float32(metadata_features.get("texture")),
                            "hog": _to_1d_float32(metadata_features.get("hog")),
                            "shape": _to_1d_float32(metadata_features.get("shape")),
                        },
                    }
                )
                del img_instance
            except Exception as e:
                print(f"Lỗi khi xử lý ảnh {path}: {e}")

        gc.collect()

    if not all_items:
        print("Không có ảnh hợp lệ để xử lý.")
        dao_postgresql.close()
        return

    # 3. Fit thống kê z-score trên toàn bộ tập ảnh
    print("\n=== Tính thống kê z-score (mean/std) trên toàn tập ===")
    feature_stats = _fit_feature_stats(all_items)
    dao_postgresql.upsert_feature_normalization_params(feature_stats)
    print("Đã lưu tham số chuẩn hóa vào bảng Feature_Normalization_Params.")

    # 4. Chuẩn hóa và lưu DB + upload MinIO theo batch
    print("\n=== Chuẩn hóa z-score và đẩy dữ liệu lên hệ thống ===")
    total_items = len(all_items)
    for i in range(0, total_items, BATCH_SIZE):
        batch_data = all_items[i:i + BATCH_SIZE]
        current_batch_num = i // BATCH_SIZE + 1
        total_batches = (total_items + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"\n--- Lưu Batch {current_batch_num}/{total_batches} (Record {i+1} đến {min(i + BATCH_SIZE, total_items)}) ---")

        print(f"Đang lưu metadata + feature chuẩn hóa cho Batch {current_batch_num} vào PostgreSQL...")
        for item in batch_data:
            try:
                normalized_features = {"image_id": item["basic"]["image_id"]}
                for feature_key in FEATURE_KEYS:
                    z_vector = _normalize_vector(item["raw_features"][feature_key], feature_stats[feature_key])
                    normalized_features[feature_key] = z_vector

                dao_postgresql.insert_image_metadata(item["basic"], normalized_features)
            except Exception as e:
                print(f"Lỗi khi lưu DB cho {item['basic']['original_filename']}: {e}")

        print(f"Đang upload ảnh 1200x800 cho Batch {current_batch_num} lên MinIO...")
        for item in batch_data:
            try:
                image_data = Image.get_storage_image_bytes_from_path(item["basic"]["original_filename"])
                if image_data:
                    dao_minio.upload_image_bytes(image_data, item["object_name"])
                else:
                    print(f"Lỗi encode ảnh để upload: {item['basic']['original_filename']}")
            except Exception as e:
                print(f"Lỗi khi upload MinIO cho {item['basic']['original_filename']}: {e}")

        gc.collect()
        print(f"Hoàn thành Batch {current_batch_num}. Đã giải phóng bộ nhớ.")

    # 3. Đóng kết nối
    dao_postgresql.close()
    print("\n=== ĐÃ HOÀN THÀNH TẤT CẢ ===")

if __name__ == "__main__":
    main()
