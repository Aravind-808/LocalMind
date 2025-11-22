import os
import shutil
from typing import List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse 
from pydantic import BaseModel

from config import Config
from ingester import IngestionPipeline
from rag_engine import RAGEngine

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ingest_pipeline = IngestionPipeline()
rag_engine = RAGEngine()

class QueryRequest(BaseModel):
    question: str

@app.get("/system/status")
async def get_system_status():
    path = Config.get_vector_store_path()
    exists = os.path.exists(path) and os.path.exists(f"{path}/index.faiss")
    return {"index_exists": exists}

@app.post("/system/clear")
async def clear_index():
    ingest_pipeline.clear_vector_store()
    rag_engine.vector_store = None 
    return {"message": "Vector store cleared successfully."}

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...), action: str = Form("append")):
    saved_paths = []
    for file in files:
        file_path = os.path.join(Config.UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved_paths.append(file_path)
    
    should_reset = (action == "reset")
    try:
        result = ingest_pipeline.ingest(saved_paths, reset_index=should_reset)
        if should_reset:
            rag_engine.vector_store = None
        for path in saved_paths:
            os.remove(path)
        return result
    except Exception as e:
        return HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_question(request: QueryRequest):
    return StreamingResponse(
        rag_engine.answer_question_stream(request.question), 
        media_type="text/plain"
    )

app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)