import os

class Config:
    UPLOAD_DIR = "uploads"
    VECTOR_STORE_DIR = "storage"
    VECTOR_STORE_NAME = "faiss_index"
    
    LLM_MODEL = "llama3.2:3b" 
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    RETRIEVER_K = 3 
    
    @classmethod
    def get_vector_store_path(cls):
        return os.path.join(cls.VECTOR_STORE_DIR, cls.VECTOR_STORE_NAME)

os.makedirs(Config.UPLOAD_DIR, exist_ok=True)
os.makedirs(Config.VECTOR_STORE_DIR, exist_ok=True)