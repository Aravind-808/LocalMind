import os
import shutil
import json
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from config import Config
from ingester import IngestionPipeline
from rag_engine import RAGEngine
from history import HistoryManager

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
    session_id: str 

async def stream_and_save(generator, session_id):
    full_response = ""
    sources_data = []
    
    async for chunk in generator:
        if "__SOURCES__:" in chunk:
            parts = chunk.split("__SOURCES__:")
            text_part = parts[0]
            json_part = parts[1]
            full_response += text_part
            yield text_part
            try:
                sources_data = json.loads(json_part)
                yield f"__SOURCES__:{json_part}" 
            except: pass
        else:
            full_response += chunk
            yield chunk

    if session_id:
        HistoryManager.add_message(session_id, "bot", full_response, sources_data)

@app.get("/history/list")
async def list_chats():
    return HistoryManager.list_sessions()

@app.get("/history/{session_id}")
async def get_chat(session_id: str):
    return HistoryManager.load_session(session_id)

@app.delete("/history/{session_id}")
async def delete_chat(session_id: str):
    HistoryManager.delete_session(session_id)
    ingest_pipeline.clear_vector_store(session_id)
    return {"status": "deleted"}

@app.post("/chat/new")
async def create_chat():
    session_id = HistoryManager.create_session()
    return {"session_id": session_id}

@app.get("/system/status/{session_id}")
async def get_index_status(session_id: str):
    """Check if index exists for THIS session"""
    path = ingest_pipeline.get_session_index_path(session_id)
    exists = os.path.exists(path) and os.path.exists(f"{path}/index.faiss")
    return {"index_exists": exists}

@app.post("/system/clear/{session_id}")
async def clear_index(session_id: str):
    ingest_pipeline.clear_vector_store(session_id)
    return {"message": "Session index cleared."}

@app.post("/upload")
async def upload_files(
    files: List[UploadFile] = File(...), 
    action: str = Form("append"),
    session_id: str = Form(...)
):
    saved_paths = []
    for file in files:
        file_path = os.path.join(Config.UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved_paths.append(file_path)
    
    should_reset = (action == "reset")
    try:
        result = ingest_pipeline.ingest(saved_paths, session_id, reset_index=should_reset)
        for path in saved_paths:
            os.remove(path)
        return result
    except Exception as e:
        return HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_question(request: QueryRequest):
    sess_id = request.session_id
    
    HistoryManager.add_message(sess_id, "user", request.question)

    session_data = HistoryManager.load_session(sess_id)
    recent_history = session_data["messages"][-8:] if session_data else []

    generator = rag_engine.answer_question_stream(request.question, sess_id, recent_history)
    
    return StreamingResponse(
        stream_and_save(generator, sess_id), 
        media_type="text/plain"
    )

app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)