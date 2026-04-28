import streamlit as st
import os
import numpy as np
import cv2
from PIL import Image as PILImage
import sys
from pathlib import Path

# Đảm bảo import được module từ thư mục gốc của dự án
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import DAOs and Services
from src.dao.DAOPostgresql import DAOPostgresql
from src.dao.DAOMinio import DAOMinio
from src.model.Image import Image
from src.services.ExtractImgDBHBT import ExtractImageDBHBT

# Cấu hình UI
st.set_page_config(page_title="Tìm kiếm ảnh lá cây (HBT Method)", layout="wide")
st.title("🌿 Hệ thống Tìm kiếm Ảnh Lá Cây (HBT Multi-stage)")
st.markdown("""
Hệ thống này sử dụng phương pháp tìm kiếm đa tầng:
1. **Giai đoạn 1**: Lọc nhanh các ứng viên dựa trên **Shape** và **Texture**.
2. **Giai đoạn 2**: Xếp hạng lại top ứng viên bằng **HOG** và **Color Histogram**.
""")

# --- 1. Khởi tạo kết nối DB ---
@st.cache_resource
def init_db():
    pg_conn_str = os.getenv("POSTGRES_CONNECTION_STRING", "postgresql://admin:admin123@localhost:5432/mydb")
    dao_postgresql = DAOPostgresql(pg_conn_str)
    dao_postgresql.connect()
    
    minio_endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    dao_minio = DAOMinio(minio_endpoint, bucket_name="plantsimage")
    dao_minio.connect()
    
    return dao_postgresql, dao_minio

try:
    dao_postgresql, dao_minio = init_db()
except Exception as e:
    st.error(f"Lỗi kết nối cơ sở dữ liệu: {e}. Đảm bảo Postgres và Minio đang chạy.")
    st.stop()

# --- 2. Các hàm hỗ trợ ---
def _to_1d_float32(vector_like) -> np.ndarray:
    if vector_like is None:
        return np.array([], dtype=np.float32)
    vector = np.asarray(vector_like, dtype=np.float32).flatten()
    if vector.size == 0:
        return np.array([], dtype=np.float32)
    return vector

def _normalize_vector(vector: np.ndarray, feature_stat: dict) -> np.ndarray:
    """Chuẩn hóa Z-score vector theo tham số từ DB."""
    dim = feature_stat.get("dim", 0)
    if dim == 0:
        return np.array([], dtype=np.float32)

    padded = np.zeros(dim, dtype=np.float32)
    if vector.size:
        limit = min(vector.size, dim)
        padded[:limit] = vector[:limit]

    mean = feature_stat.get("mean", np.zeros(dim))
    std = feature_stat.get("std", np.ones(dim))
    
    # Tránh chia cho 0
    std = np.where(std < 1e-8, 1.0, std)
    
    return ((padded - mean) / std).astype(np.float32)

# Load normalization params
@st.cache_data
def load_normalization_params():
    return dao_postgresql.get_feature_normalization_params()

norm_params = load_normalization_params()

# --- 3. UI Tải ảnh lên ---
st.sidebar.header("Tùy chọn tìm kiếm")
top_k = st.sidebar.slider("Số lượng kết quả (k_langgieng)", min_value=5, max_value=50, value=10)
debug_mode = st.sidebar.checkbox("Hiển thị điểm chi tiết (Debug Mode)", value=False)

uploaded_file = st.file_uploader("Chọn một ảnh lá cây...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Ảnh truy vấn")
        image_pil = PILImage.open(uploaded_file)
        st.image(image_pil, caption="Ảnh đầu vào", use_column_width=True)
        
    with col2:
        st.subheader("Tiến trình")
        
        # Lấy nội dung file và đọc vào bộ nhớ OpenCV
        file_bytes = np.asarray(bytearray(uploaded_file.getvalue()), dtype=np.uint8)
        img_cv2 = cv2.imdecode(file_bytes, 1)

        try:
            with st.spinner("Đang trích xuất đặc trưng và chuẩn hóa..."):
                # 1. Khởi tạo đối tượng Image bằng OpenCV matrix
                img_instance = Image(img_input=img_cv2)
                img_instance.preprocess()
                raw_features = img_instance.ExtractFeatures()
                
                # 2. Làm sạch và chuẩn hóa đặc trưng (Z-score)
                query_features_normalized = {}
                feature_keys = ["color", "texture", "hog", "shape"]
                
                for feature_key in feature_keys:
                    raw_vec = _to_1d_float32(raw_features.get(feature_key))
                    if feature_key in norm_params:
                        z_vec = _normalize_vector(raw_vec, norm_params[feature_key])
                        query_features_normalized[feature_key] = z_vec
                    else:
                        st.warning(f"Chưa có tham số chuẩn hóa trong DB cho: {feature_key}. Sẽ bỏ qua đặc trưng này.")
                        query_features_normalized[feature_key] = None
            
            with st.spinner("Đang tìm kiếm đa tầng (HBT Method)..."):
                # 3. Tìm kiếm bằng ExtractImageDBHBT
                search_engine = ExtractImageDBHBT(k_langgieng=top_k, dao_minio=dao_minio, dao_postgresql=dao_postgresql)
                
                # Trả về list: [(distance, image_id, debug_info), ...]
                top_results = search_engine.extract_image_postgresql(query_features_normalized, debug_mode=debug_mode)
            
            if not top_results:
                st.warning("Không tìm thấy kết quả nào. Có thể cơ sở dữ liệu trống.")
            else:
                st.success(f"Tìm thấy {len(top_results)} kết quả phù hợp bằng phương pháp Multi-stage!")
                
                # Lấy chi tiết metadata
                result_ids = [res[1] for res in top_results]
                metadata_list = dao_postgresql.get_metadata_by_ids(result_ids)
                meta_map = {m["image_id"]: m for m in metadata_list}
                
                # 4. Hiển thị kết quả dạng lưới
                st.subheader("Kết quả tìm kiếm")
                cols_per_row = 3
                
                for i in range(0, len(top_results), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j in range(cols_per_row):
                        if i + j < len(top_results):
                            dist, img_id, debug_info = top_results[i + j]
                            meta = meta_map.get(img_id, {})
                            
                            minio_url = meta.get("minio_url")
                            category = meta.get("category", "Unknown")
                            
                            with cols[j]:
                                try:
                                    # Sử dụng object_name theo định dạng lưu trữ trong Image.py
                                    object_name = f"{category}/{img_id}.jpg"
                                    
                                    # Tải trực tiếp bytes từ MinIO thay vì dùng URL công khai
                                    response = dao_minio.connection.get_object(dao_minio.bucket_name, object_name)
                                    img_data = response.read()
                                    st.image(img_data, use_column_width=True)
                                    
                                    # Dọn dẹp connection
                                    response.close()
                                    response.release_conn()
                                except Exception as minio_err:
                                    st.warning("Không thể tải ảnh")
                                    st.caption(str(minio_err))
                                    
                                st.caption(f"**Loài:** {category}")
                                st.caption(f"**Độ lệch (GĐ2):** {dist:.4f}")
                                
                                if debug_mode and debug_info:
                                    st.markdown(f"""
                                    <div style="font-size: 0.8em; color: gray;">
                                    <b>Shape:</b> {debug_info.get('shape', 0):.4f}<br>
                                    <b>Texture:</b> {debug_info.get('texture', 0):.4f}<br>
                                    <b>HOG:</b> {debug_info.get('hog', 0):.4f}<br>
                                    <b>Color:</b> {debug_info.get('color', 0):.4f}
                                    </div>
                                    """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Đã xảy ra lỗi trong quá trình tìm kiếm: {e}")
            import traceback
            st.code(traceback.format_exc())
