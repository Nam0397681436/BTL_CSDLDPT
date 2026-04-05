CREATE TABLE Basic_Metadata (
    Image_ID VARCHAR(50) PRIMARY KEY,
    MinIO_URL VARCHAR(255) NOT NULL,
    Category VARCHAR(50), -- Nhãn loại lá cây (ví dụ: la_phong, la_bang)
    Date_created DATE,
    Description TEXT
);

CREATE TABLE Global_Features (
    Image_ID VARCHAR(50) PRIMARY KEY REFERENCES Basic_Metadata(Image_ID) ON DELETE CASCADE,
    
    Color_Vector REAL[],   -- Lưu vector biểu đồ màu (Color Histogram)
    
    Texture_Vector REAL[], -- Lưu vector kết cấu (Co-occurrence matrix hoặc Gabor)
    
    HOG_Vector REAL[]    -- Lưu vector hình dạng/ Cấu trúc 
);


CREATE TABLE Local_SIFT_Features (
    Feature_ID SERIAL PRIMARY KEY,         -- Mã định danh cho từng điểm SIFT
    Image_ID VARCHAR(50) REFERENCES Basic_Metadata(Image_ID) ON DELETE CASCADE,
    SIFT_Vector REAL[]                     -- Lưu vector đặc trưng SIFT 128 chiều của điểm đó
);