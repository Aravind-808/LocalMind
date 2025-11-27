import os
import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from config import Config

class RAGEngine:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
        self.llm = ChatOllama(
            model=Config.LLM_MODEL,
            temperature=0.1,
            keep_alive="10m",
            num_ctx=4096 
        )
        
        self.prompt = ChatPromptTemplate.from_template("""
        You are a helpful AI assistant. Use the provided context and conversation history to answer the user's question.
        If the answer is not in the context, say you don't know.
        
        Previous Conversation:
        {chat_history}
        
        Context from Documents:
        {context}
        
        Current Question: {question}
        """)

    def get_index_path(self, session_id):
        return os.path.join(Config.VECTOR_STORE_DIR, session_id)

    async def answer_question_stream(self, question: str, session_id: str, chat_history: list):
        path = self.get_index_path(session_id)
        if not os.path.exists(path) or not os.path.exists(f"{path}/index.faiss"):
            yield "No documents found for this chat. Please upload files to start."
            return

        vector_store = FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)
        
        retriever = vector_store.as_retriever(search_kwargs={"k": Config.RETRIEVER_K})
        docs = retriever.invoke(question)
        context_text = "\n\n".join([doc.page_content for doc in docs])
        
        history_text = ""
        for msg in chat_history:
            role = "User" if msg['role'] == 'user' else "AI"
            history_text += f"{role}: {msg['content']}\n"

        sources = []
        seen = set()
        for doc in docs:
            src = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", None)
            label = f"{src} (Page {page + 1})" if page is not None else src
            if label not in seen:
                sources.append(label)
                seen.add(label)

        chain = self.prompt | self.llm
        
        async for chunk in chain.astream({
            "context": context_text, 
            "chat_history": history_text,
            "question": question
        }):
            yield chunk.content

        yield f"\n\n__SOURCES__:{json.dumps(sources)}"