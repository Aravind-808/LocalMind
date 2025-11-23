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
    session_id: Optional[str] = None

# --- Wrapper for Stream Saving ---
async def stream_and_save(generator, session_id):
    """
    Yields chunks to client immediately, but accumulates them 
    to save to history file at the end.
    """
    full_response = ""
    sources_data = []
    
    async for chunk in generator:
        # Check if this chunk is the sources block
        if "__SOURCES__:" in chunk:
            parts = chunk.split("__SOURCES__:")
            text_part = parts[0]
            json_part = parts[1]
            
            full_response += text_part
            yield text_part # Yield text part
            
            try:
                sources_data = json.loads(json_part)
                # We yield the source marker to frontend so it can render badges
                yield f"__SOURCES__:{json_part}" 
            except:
                pass
        else:
            full_response += chunk
            yield chunk

    # Save to history after stream ends
    if session_id:
        HistoryManager.add_message(session_id, "bot", full_response, sources_data)

# --- History Routes ---
@app.get("/history/list")
async def list_chats():
    return HistoryManager.list_sessions()

@app.get("/history/{session_id}")
async def get_chat(session_id: str):
    return HistoryManager.load_session(session_id)

@app.delete("/history/{session_id}")
async def delete_chat(session_id: str):
    HistoryManager.delete_session(session_id)
    return {"status": "deleted"}

@app.post("/chat/new")
async def create_chat():
    session_id = HistoryManager.create_session()
    return {"session_id": session_id}

# --- Existing Routes ---
@app.get("/system/status")
async def get_system_status():
    path = Config.get_vector_store_path()
    exists = os.path.exists(path) and os.path.exists(f"{path}/index.faiss")
    return {"index_exists": exists}

@app.post("/system/clear")
async def clear_index():
    ingest_pipeline.clear_vector_store()
    rag_engine.vector_store = None 
    return {"message": "Vector store cleared."}

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
    # 1. Ensure Session Exists
    sess_id = request.session_id
    if not sess_id:
        sess_id = HistoryManager.create_session()

    # 2. Save User Question
    HistoryManager.add_message(sess_id, "user", request.question)

    # 3. Stream & Save Bot Response
    generator = rag_engine.answer_question_stream(request.question)
    
    return StreamingResponse(
        stream_and_save(generator, sess_id), 
        media_type="text/plain"
    )

app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)