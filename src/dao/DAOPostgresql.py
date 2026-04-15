from model.IDatabase import IDatabase
import psycopg2
import numpy as np

class DAOPostgresql(IDatabase):
    @staticmethod
    def _to_pg_float_array(value):
        if value is None:
            return []
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, np.ndarray):
            value = value.flatten().tolist()
        if not isinstance(value, list):
            value = list(value)
        return [float(v) for v in value]

    def connect(self):
        if self.connection is None:
            # psycopg2 uses standard postgresql:// connection strings
            self.connection = psycopg2.connect(self.connection_string)
            self._ensure_schema_and_tables()

    def _ensure_schema_and_tables(self):
        with self.connection.cursor() as cursor:
            cursor.execute(
                '''
                CREATE TABLE IF NOT EXISTS "Basic_Metadata" (
                    image_id VARCHAR(50) PRIMARY KEY,
                    original_filename VARCHAR(255),
                    minio_url VARCHAR(255) NOT NULL,
                    category VARCHAR(50),
                    description VARCHAR(255)
                )
                '''
            )
            cursor.execute(
                '''
                CREATE TABLE IF NOT EXISTS "Images_Features" (
                    image_id VARCHAR(50) PRIMARY KEY REFERENCES "Basic_Metadata"(image_id) ON DELETE CASCADE,
                    color_vector REAL[],
                    texture_vector REAL[],
                    hog_vector REAL[],
                    shape_vector REAL[],
                    venation_vector REAL[]
                )
                '''
            )
            cursor.execute(
                '''
                CREATE TABLE IF NOT EXISTS "Feature_Normalization_Params" (
                    feature_name VARCHAR(50) PRIMARY KEY,
                    mean_vector REAL[] NOT NULL,
                    std_vector REAL[] NOT NULL,
                    vector_dim INTEGER NOT NULL,
                    updated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
                '''
            )
        self.connection.commit()

    def upsert_feature_normalization_params(self, feature_stats):
        if self.connection is None:
            self.connect()

        try:
            with self.connection.cursor() as cursor:
                for feature_name, stats in feature_stats.items():
                    mean_vector = stats.get("mean", [])
                    std_vector = stats.get("std", [])
                    dim = int(stats.get("dim", 0))

                    mean_list = mean_vector.tolist() if hasattr(mean_vector, "tolist") else list(mean_vector)
                    std_list = std_vector.tolist() if hasattr(std_vector, "tolist") else list(std_vector)

                    cursor.execute(
                        """
                        INSERT INTO "Feature_Normalization_Params"
                            (feature_name, mean_vector, std_vector, vector_dim, updated_at)
                        VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
                        ON CONFLICT (feature_name)
                        DO UPDATE SET
                            mean_vector = EXCLUDED.mean_vector,
                            std_vector = EXCLUDED.std_vector,
                            vector_dim = EXCLUDED.vector_dim,
                            updated_at = CURRENT_TIMESTAMP
                        """,
                        (feature_name, mean_list, std_list, dim),
                    )
                self.connection.commit()
        except Exception as e:
            self.connection.rollback()
            print(f"Error upserting normalization params: {e}")
            
    def close(self):
        if self.connection:
            self.connection.close()
            self.connection = None

    def insert_image_metadata(self, metadata_basic, metadata_features):
        if self.connection is None:
            self.connect()
        
        try: 
            with self.connection.cursor() as cursor:
                # Insert Basic Metadata
                cursor.execute(
                    """
                    INSERT INTO "Basic_Metadata" (image_id, original_filename, minio_url, category, description)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (image_id) DO NOTHING
                    """,
                    (
                        metadata_basic["image_id"],
                        metadata_basic["original_filename"],
                        metadata_basic["minio_url"],
                        metadata_basic["category"],
                        metadata_basic["description"]
                    )
                )

                # Insert Features
                color_vector = self._to_pg_float_array(metadata_features.get("color"))
                texture_vector = self._to_pg_float_array(metadata_features.get("texture"))
                hog_vector = self._to_pg_float_array(metadata_features.get("hog"))
                shape_vector = self._to_pg_float_array(metadata_features.get("shape"))
                venation_vector = self._to_pg_float_array(metadata_features.get("venation"))

                cursor.execute(
                    """
                    INSERT INTO "Images_Features" (
                        image_id, color_vector, texture_vector, hog_vector, shape_vector, venation_vector
                    )
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (image_id) DO NOTHING
                    """,
                    (
                        metadata_features["image_id"],
                        color_vector,
                        texture_vector,
                        hog_vector,
                        shape_vector,
                        venation_vector,
                    )
                )
                self.connection.commit()
        except Exception as e:
            self.connection.rollback()
            print(f"Error inserting image metadata: {e}")
            
    def get_image_metadata(self, image_id):
        if self.connection is None:
            self.connect()
        with self.connection.cursor() as cursor:
            cursor.execute('SELECT * FROM "Basic_Metadata" WHERE image_id = %s', (image_id,))
            return cursor.fetchone()
    
    def get_feature_normalization_params(self):
        if self.connection is None:
            self.connect()

        with self.connection.cursor() as cursor:
            cursor.execute(
                '''
                SELECT feature_name, mean_vector, std_vector, vector_dim
                FROM "Feature_Normalization_Params"
                '''
            )
            rows = cursor.fetchall()

        stats = {}
        for feature_name, mean_vector, std_vector, vector_dim in rows:
            stats[feature_name] = {
                "mean": mean_vector,
                "std": std_vector,
                "dim": vector_dim,
            }

        return stats
