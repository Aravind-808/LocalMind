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

    def clear_vector_store(self):
        # delete vectror store directory
        path = Config.get_vector_store_path()
        if os.path.exists(path):
            shutil.rmtree(path)
            return True
        return False

    def ingest(self, file_paths: List[str], reset_index: bool = False):
        # either append to or reset the vector store index

        if reset_index:
            self.clear_vector_store()

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
        
        vector_store_path = Config.get_vector_store_path()
        
        if os.path.exists(vector_store_path) and os.path.exists(f"{vector_store_path}/index.faiss"):
            vector_store = FAISS.load_local(vector_store_path, self.embeddings, allow_dangerous_deserialization=True)
            vector_store.add_documents(chunks)
        else:
            vector_store = FAISS.from_documents(chunks, self.embeddings)
            
        vector_store.save_local(vector_store_path)
        
        return {
            "status": "success", 
            "chunks": len(chunks), 
            "files": len(file_paths),
            "action": "reset" if reset_index else "append"
        }