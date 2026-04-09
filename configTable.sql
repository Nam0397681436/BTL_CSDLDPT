CREATE TABLE Basic_Metadata (
    image_id VARCHAR(50) PRIMARY KEY,
    original_filename VARCHAR(255),          
    minio_url VARCHAR(255) NOT NULL,
    category VARCHAR(50),
    description VARCHAR(255)

);

CREATE TABLE Images_Features (
    image_id VARCHAR(50) PRIMARY KEY REFERENCES Basic_Metadata(image_id) ON DELETE CASCADE,
    
    color_vector REAL[],  
    texture_vector REAL[],
    hog_vector REAL[],     
    shape_vector REAL[],   
    venation_vector REAL[] 
);
