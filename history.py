import os
import json
import uuid
from datetime import datetime
from typing import List, Dict

HISTORY_DIR = "history"
os.makedirs(HISTORY_DIR, exist_ok=True)

class HistoryManager:
    @staticmethod
    def create_session(title: str = "New Chat"):
        session_id = str(uuid.uuid4())
        data = {
            "id": session_id,
            "title": title,
            "created_at": datetime.now().isoformat(),
            "messages": []
        }
        HistoryManager.save_session(session_id, data)
        return session_id

    @staticmethod
    def save_session(session_id: str, data: Dict):
        path = os.path.join(HISTORY_DIR, f"{session_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load_session(session_id: str):
        path = os.path.join(HISTORY_DIR, f"{session_id}.json")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def add_message(session_id: str, role: str, content: str, sources: List[str] = None):
        data = HistoryManager.load_session(session_id)
        if not data:
            return
        
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        if sources:
            message["sources"] = sources
            
        data["messages"].append(message)
        
        # Update title if it's the first user message and title is default
        if role == "user" and len(data["messages"]) == 1:
            data["title"] = content[:30] + "..." if len(content) > 30 else content
            
        HistoryManager.save_session(session_id, data)

    @staticmethod
    def list_sessions():
        sessions = []
        if not os.path.exists(HISTORY_DIR):
            return []
            
        for filename in os.listdir(HISTORY_DIR):
            if filename.endswith(".json"):
                path = os.path.join(HISTORY_DIR, filename)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        sessions.append({
                            "id": data["id"],
                            "title": data.get("title", "Untitled"),
                            "created_at": data.get("created_at", "")
                        })
                except:
                    continue
        # Sort by newest first
        sessions.sort(key=lambda x: x["created_at"], reverse=True)
        return sessions

    @staticmethod
    def delete_session(session_id: str):
        path = os.path.join(HISTORY_DIR, f"{session_id}.json")
        if os.path.exists(path):
            os.remove(path)