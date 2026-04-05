from model.IDatabase import IDatabase
import minio
import boto3

class DAOMinio(IDatabase):
    def connect(self):
        if self.connection is None:
            self.connection = minio.Minio(self.connection_string, secure=False)
    def close(self):
        super().close()

        