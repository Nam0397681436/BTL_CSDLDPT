from model.IDatabase import IDatabase
import psycopg2
from psycopg2.extras import execute_values

class DAOPostgresql(IDatabase):
    def connect(self):
        if self.connection is None:
            # psycopg2 uses standard postgresql:// connection strings
            self.connection = psycopg2.connect(self.connection_string)
            # Create schema if not exists
            with self.connection.cursor() as cursor:
                cursor.execute('CREATE SCHEMA IF NOT EXISTS "ImageMD"')
                self.connection.commit()
            
    def close(self):
        if self.connection:
            self.connection.close()
            self.connection = None

    def insert_image_metadata(self, metadata_basic, metadata_features):
        if self.connection is None:
            self.connect()
        
        try: 
            with self.connection.cursor() as cursor:
                # Set search path to your schema
                cursor.execute('SET search_path TO "ImageMD", public')
                
                # Insert Basic Metadata
                cursor.execute(
                    """
                    INSERT INTO Basic_Metadata (image_id, original_filename, minio_url, category, description)
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
                cursor.execute(
                    """
                    INSERT INTO Images_Features (image_id, color_vector, texture_vector, hog_vector, shape_vector, venation_vector)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (image_id) DO NOTHING
                    """,
                    (
                        metadata_features["image_id"],
                        metadata_features["color"].tolist() if hasattr(metadata_features["color"], 'tolist') else metadata_features["color"],
                        metadata_features["texture"], # Already a list from _compute_texture_features
                        metadata_features["hog"].tolist() if hasattr(metadata_features["hog"], 'tolist') else metadata_features["hog"],
                        metadata_features["shape"].tolist() if hasattr(metadata_features["shape"], 'tolist') else metadata_features["shape"],
                        metadata_features["venation"].tolist() if hasattr(metadata_features["venation"], 'tolist') else metadata_features["venation"]
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
            cursor.execute('SELECT * FROM "ImageMD".Basic_Metadata WHERE image_id = %s', (image_id,))
            return cursor.fetchone()
