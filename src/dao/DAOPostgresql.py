from model.IDatabase import IDatabase
import postgresql

class DAOPostgresql(IDatabase):
    def connect(self):
        if self.connection is None:
            self.connection = postgresql.open(self.connection_string)
            self.connection.execute('CREATE SCHEMA IF NOT EXISTS "ImageMD"')
            self.connection.execute('SET search_path TO "ImageMD", public')
            
    def close(self):
        super().close()

    def insert_image_metadata(self, image_path, features):
        if self.connection is None:
            raise Exception("Database connection is not established.")
        
        pass

    def get_image_metadata(self, image_id):
        if self.connection is None:
            raise Exception("Database connection is not established.")
        pass   
