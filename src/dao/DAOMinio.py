from model.IDatabase import IDatabase
import minio
import boto3

class DAOMinio(IDatabase):
    def __init__(self, connection_string, bucket_name="plantsimage"):
        super().__init__(connection_string)
        self.bucket_name = bucket_name

    def connect(self):
        if self.connection is None:
            import os
            access_key = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
            secret_key = os.getenv("MINIO_SECRET_KEY", "minioadmin")
            # Standard MinIO setup
            self.connection = minio.Minio(
                self.connection_string,
                access_key=access_key,
                secret_key=secret_key,
                secure=False
            )
            
            # Đảm bảo Bucket tồn tại
            try:
                if not self.connection.bucket_exists(self.bucket_name):
                    self.connection.make_bucket(self.bucket_name)
                    print(f"Created bucket: {self.bucket_name}")
            except Exception as e:
                print(f"Warning: Could not verify/create bucket {self.bucket_name}: {e}")

    def upload_image(self, local_path, object_name):
        if self.connection is None:
            self.connect()
        try:
            self.connection.fput_object(
                self.bucket_name,
                object_name,
                local_path,
                content_type="image/jpeg"
            )
        except Exception as e:
            print(f"Error uploading image {object_name}: {e}")

    def upload_image_bytes(self, data, object_name):
        if self.connection is None:
            self.connect()
        try:
            import os
            # Lấy kích thước data stream
            data.seek(0, os.SEEK_END)
            size = data.tell()
            data.seek(0)
            
            self.connection.put_object(
                self.bucket_name,
                object_name,
                data,
                size,
                content_type="image/jpeg"
            )
        except Exception as e:
            print(f"Error uploading image bytes {object_name}: {e}")
    
        