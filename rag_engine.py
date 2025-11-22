import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from config import Config
import os

class RAGEngine:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
        self.llm = ChatOllama(
            model=Config.LLM_MODEL,
            temperature=0.1,
            keep_alive="10m",
            num_ctx=4096
        )
        self.vector_store = None
        self.prompt = ChatPromptTemplate.from_template("""
        Answer the question based ONLY on the following context. Use markdown formatting for any code snippets, tables, or lists,
        or to emphasize text.
        If the answer is not contained within the context, respond with "I don't know.":
        {context}
        
        Question: {question}
        """)

    def load_index(self):
        path = Config.get_vector_store_path()
        if os.path.exists(path) and os.path.exists(f"{path}/index.faiss"):
            self.vector_store = FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)
            return True
        return False

    async def answer_question_stream(self, question: str):
        # streams the answer to the qn 
        if not self.vector_store:
            loaded = self.load_index()
            if not loaded:
                yield "Index not found. Please upload documents first."
                return

        retriever = self.vector_store.as_retriever(search_kwargs={"k": Config.RETRIEVER_K})
        docs = retriever.invoke(question)
        
        context_text = "\n\n".join([doc.page_content for doc in docs])
        
        sources = []
        seen = set()
        for doc in docs:
            src = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", None)
            label = f"{src} (Page {page + 1})" if page is not None else src
            if label not in seen:
                sources.append(label)
                seen.add(label)

        chain = self.prompt | self.llm # using | for chaining the prompt and llm
        
        async for chunk in chain.astream({"context": context_text, "question": question}):
            yield chunk.content

        yield f"\n\n__SOURCES__:{json.dumps(sources)}"
