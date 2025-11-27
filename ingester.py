import os
import shutil
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from config import Config
from ocr_processor import OCRProcessor

class IngestionPipeline:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )

    def _process_pdf(self, file_path):
        loader = PyPDFLoader(file_path)
        return loader.load()

    def _process_image(self, file_path):
        text = OCRProcessor.extract_text(file_path)
        if text:
            return [Document(page_content=text, metadata={"source": os.path.basename(file_path)})]
        return []

    def get_session_index_path(self, session_id):
        return os.path.join(Config.VECTOR_STORE_DIR, session_id)

    def clear_vector_store(self, session_id):
        path = self.get_session_index_path(session_id)
        if os.path.exists(path):
            shutil.rmtree(path)
            return True
        return False

    def ingest(self, file_paths: List[str], session_id: str, reset_index: bool = False):
        if not session_id:
            return {"status": "error", "message": "Session ID is missing."}

        if reset_index:
            self.clear_vector_store(session_id)

        documents = []
        for path in file_paths:
            ext = os.path.splitext(path)[1].lower()
            try:
                if ext == ".pdf":
                    documents.extend(self._process_pdf(path))
                elif ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                    documents.extend(self._process_image(path))
            except Exception as e:
                print(f"Error processing {path}: {e}")

        if not documents:
            return {"status": "error", "message": "No text extracted."}

        chunks = self.text_splitter.split_documents(documents)
        
        index_path = self.get_session_index_path(session_id)
        
        if os.path.exists(index_path) and os.path.exists(f"{index_path}/index.faiss"):
            vector_store = FAISS.load_local(index_path, self.embeddings, allow_dangerous_deserialization=True)
            vector_store.add_documents(chunks)
        else:
            vector_store = FAISS.from_documents(chunks, self.embeddings)
            
        vector_store.save_local(index_path)
        
        return {
            "status": "success", 
            "chunks": len(chunks), 
            "files": len(file_paths),
            "session_id": session_id
        }   