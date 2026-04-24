import os
import sys
from pathlib import Path

# PHẢI THỰC HIỆN DÒNG NÀY ĐẦU TIÊN ĐỂ NHẬN DIỆN PACKAGE 'src'
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np  
import streamlit as st
from src.model.Image import Image
from src.services.ExtractImageDB import ExtractImageDB
from src.services.ExtractImageDBWeighted import ExtractImageDBWeighted
from src.dao.DAOPostgresql import DAOPostgresql
from src.dao.DAOMinio import DAOMinio
from src.utils.normalVector import normalize_vector_by_feature_name

def main():
    img_path="./test/img_test/potato.jpg"
    img_input=cv2.imread(img_path)
    
    img_object=Image(img_input=img_input)
    img_object.preprocess()
    
    feature_img_input=img_object.ExtractFeatures()
    pg_conn_str = "postgresql://admin:admin123@localhost:5432/mydb"
    feature_img_input={
        "color":normalize_vector_by_feature_name(feature_img_input["color"], "color", connection_string=pg_conn_str),
        "texture":normalize_vector_by_feature_name(feature_img_input["texture"], "texture", connection_string=pg_conn_str),
        "hog":normalize_vector_by_feature_name(feature_img_input["hog"], "hog", connection_string=pg_conn_str),
        "shape":normalize_vector_by_feature_name(feature_img_input["shape"], "shape", connection_string=pg_conn_str),
        "venation":normalize_vector_by_feature_name(feature_img_input["venation"], "venation", connection_string=pg_conn_str)
    }
 
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
    
    try: 
        resulteImageQuery=ExtractImageDBWeighted(dao_minio,dao_postgresql)
        resulteImageQuery.extract_image_postgresql(feature_img_input)
        list_img_url=resulteImageQuery.extract_image_minio(feature_img_input)
        for img_url in list_img_url:
            print(img_url)
    except Exception as e:
        print(f"Lỗi xử lý: {e}")
        return
    
if __name__ == "__main__":
    main()                     
    