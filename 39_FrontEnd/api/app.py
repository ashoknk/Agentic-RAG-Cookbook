import sys
import time
from pathlib import Path
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project root to sys.path to resolve src imports
sys.path.append(str(Path(__file__).parent.parent))

from src.config.config import Config
from src.document_ingestion.document_processor import DocumentProcessor
from src.vectorstore.vectorstore import VectorStore
from src.graph_builder.graph_builder import GraphBuilder

app = FastAPI(title="Agentic RAG Backend API", description="API backend for Agentic RAG system")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict this to specific origins (e.g. ["http://localhost:5173"]) in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# App state to hold initialized RAG components
app.state.rag_system = None
app.state.num_chunks = 0

class SearchRequest(BaseModel):
    question: str

class DocumentResponse(BaseModel):
    page_content: str
    metadata: dict

class SearchResponse(BaseModel):
    answer: str
    time: float
    retrieved_docs: List[DocumentResponse]

class InitResponse(BaseModel):
    status: str
    num_chunks: int

@app.get("/api/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "initialized": app.state.rag_system is not None}

@app.post("/api/init", response_model=InitResponse)
def init_rag():
    """Initializes the Agentic RAG system and caches it in memory."""
    if app.state.rag_system is not None:
        return InitResponse(status="already_initialized", num_chunks=app.state.num_chunks)
    
    try:
        # Load LLM
        llm = Config.get_llm()
        
        # Process document URLs
        doc_processor = DocumentProcessor(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        vector_store = VectorStore()
        
        urls = Config.DEFAULT_URLS
        documents = doc_processor.process_urls(urls)
        
        # Create vector store
        vector_store.create_vectorstore(documents)
        
        # Build LangGraph reasoning graph
        graph_builder = GraphBuilder(
            retriever=vector_store.get_retriever(),
            llm=llm
        )
        graph_builder.build()
        
        # Cache in app state
        app.state.rag_system = graph_builder
        app.state.num_chunks = len(documents)
        
        return InitResponse(status="initialized", num_chunks=app.state.num_chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize RAG: {str(e)}")

@app.post("/api/search", response_model=SearchResponse)
def search_rag(request: SearchRequest):
    """Executes a search query against the initialized RAG system."""
    if app.state.rag_system is None:
        raise HTTPException(
            status_code=400, 
            detail="RAG system not initialized. Please call /api/init first."
        )
    
    try:
        start_time = time.time()
        
        # Run query through the graph
        result = app.state.rag_system.run(request.question)
        
        elapsed_time = time.time() - start_time
        
        # Format the retrieved documents for response
        docs = []
        for doc in result.get('retrieved_docs', []):
            docs.append(DocumentResponse(
                page_content=doc.page_content,
                metadata=doc.metadata or {}
            ))
            
        return SearchResponse(
            answer=result.get('answer', ''),
            time=elapsed_time,
            retrieved_docs=docs
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search execution failed: {str(e)}")
