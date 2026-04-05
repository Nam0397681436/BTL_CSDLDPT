import cv2 as cv
import numpy as np
from model.Image import Image
from services.ExtractFeatureImage import (
    extract_feature_HOG,
    extract_feature_SIFT,
)

import os


def main():
    # quét lấy hết các đường dẫn file ảnh
    image_folder = "./data"
    image_paths = []

    for root, dirs, files in os.walk(image_folder):
        for file in files:
            if file.endswith((".jpg", ".jpeg", ".png",".JPG", ".JPEG", ".PNG")):
                image_paths.append(os.path.join(root, file))

    # tạo đối tượng Image và trích xuất đặc trưng

    list_images = []
    for path in image_paths:
        img_intances = Image(path)

        # tiền xử lý ảnh - > bước này resize ảnh về cùng kích thước để dễ dàng trích xuất đặc trưng
        img_intances.preprocess()
        metadata_features = img_intances.ExtractFeatures()
        list_images.append((img_intances, metadata_features))

    # insert metadata vào psg và file image vào minio
    # insert metadata vào psg
    try:
        from dao.DAOPostgresql import DAOPostgresql

        dao_postgresql = DAOPostgresql(os.getenv("POSTGRES_CONNECTION_STRING"))
        dao_postgresql.connect()

        for img, features in list_images:
            dao_postgresql.insert_image_metadata(img.source_path, features)
        dao_postgresql.close()

    except ImportError as e:
        print(f"Error importing DAOPostgresql: {e}")
    
    try:
        from dao.DAOMinio import DAOMinio
        dao_minio = DAOMinio(
            os.getenv("MINIO_ENDPOINT"),
            os.getenv("MINIO_ACCESS_KEY"),
            os.getenv("MINIO_SECRET_KEY"),
            os.getenv("MINIO_BUCKET_NAME")
        )
        dao_minio.connect()
        for img, features in list_images:
            dao_minio.upload_image(f"{os.getenv('MINIO_BUCKET_NAME')}/{img.source_path}", img.source_path)
        dao_minio.close()
        
    except ImportError as e:
        print(f"Error importing DAOMinio: {e}")
        

if __name__ == "__main__":
    main()
