from minio import Minio

class MinioClient:
    def __init__(self, host: str, access_key: str, secret_key: str, secure_bool: bool = False):
        self.host = host
        self.access_key = access_key
        self.secret_key = secret_key
        self.secure_bool = secure_bool
    
    def get_client(self):
        client = Minio(self.host,
            access_key=self.access_key,
            secret_key=self.secret_key,
            secure=self.secure_bool
        )
        return client
