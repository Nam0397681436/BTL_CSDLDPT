import postgresql
import sqlmachine
import minio
import boto3
from abc import ABC, abstractmethod

class IDatabase:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.connection = None
    @abstractmethod
    def connect(self):
        pass
    @abstractmethod
    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None