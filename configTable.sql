CREATE TABLE IF NOT EXISTS "Basic_Metadata" (
    image_id VARCHAR(50) PRIMARY KEY,
    original_filename VARCHAR(255),
    minio_url VARCHAR(255) NOT NULL,
    category VARCHAR(50),
    aspect_ratio VARCHAR(20),
    format_file VARCHAR(10),
    description VARCHAR(255)
);

CREATE TABLE IF NOT EXISTS "Images_Features" (
    image_id VARCHAR(50) PRIMARY KEY REFERENCES "Basic_Metadata"(image_id) ON DELETE CASCADE,
    color_vector REAL[],
    texture_vector REAL[],
    hog_vector REAL[],
    shape_vector REAL[]
);

CREATE TABLE IF NOT EXISTS "Feature_Normalization_Params" (
    feature_name VARCHAR(50) PRIMARY KEY,
    mean_vector REAL[] NOT NULL,
    std_vector REAL[] NOT NULL,
    vector_dim INTEGER NOT NULL,
    updated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP
)
