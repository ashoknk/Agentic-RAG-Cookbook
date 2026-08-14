"""Streamlit UI for Agentic RAG System - Simplified Version"""

import streamlit as st
from pathlib import Path
import sys
import time

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.config.config import Config
from src.document_ingestion.document_processor import DocumentProcessor
from src.vectorstore.vectorstore import VectorStore
from src.graph_builder.graph_builder import GraphBuilder

# Page configuration
st.set_page_config(
    page_title="🤖 RAG Search",
    page_icon="🔍",
    layout="centered"
)

# Simple CSS
st.markdown("""
    <style>
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

def init_session_state():
    """Initialize session state variables"""
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    if 'history' not in st.session_state:
        st.session_state.history = []


def load_sources_from_file() -> list[str]:
    """Load sources from data/urls.txt and add a local PDF folder if it exists."""
    sources = []

    urls_file = Path("data/urls.txt")
    if urls_file.exists():
        sources.extend(
            [line.strip() for line in urls_file.read_text().splitlines() if line.strip()]
        )

    pdf_dir = Path("data")
    if pdf_dir.exists() and pdf_dir.is_dir():
        sources.append(str(pdf_dir))

    print(f"streamlit_app.py- Loading sources: {sources}")        

    return sources or Config.DEFAULT_URLS

# to cache global, non-serializable objects—such as ML models/ DB connections so they are only loaded once and shared across all users, sessions
@st.cache_resource
def initialize_rag():
    """Initialize the RAG system (cached)"""
    try:
        # Initialize components
        llm = Config.get_llm()
        doc_processor = DocumentProcessor(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        vector_store = VectorStore()
        
        # Load sources from file or fallback to defaults
        sources = load_sources_from_file()
        
        # Process documents from the configured sources
        documents = doc_processor.process_sources(sources)
        
        # Create vector store
        vector_store.create_vectorstore(documents)
        
        # Build graph
        graph_builder = GraphBuilder(
            retriever=vector_store.get_retriever(),
            llm=llm
        )
        graph_builder.build()
        
        return graph_builder, len(documents)
    except Exception as e:
        st.error(f"Failed to initialize: {str(e)}")
        return None, 0

def main():
    """Main application"""
    init_session_state()
    
    # Title ans sample questions
    st.title("🧠Agentic Retrieval-Augmented Generation (RAG) System")
    st.caption("Ask questions, extract insights, and cite sources from your loaded documents.")
    st.markdown("---")
    st.caption("What is the difference between short-term and long-term memory in LLM agents?")
    st.caption("What are the core components of the OpenClaw execution framework?")
    st.caption("How does the interaction loop operate between agent memory, the LLM, and tools?")
    st.caption("What are the main challenges when extending diffusion models to video generation?")
    st.caption("What strategies make video diffusion models more computationally efficient?")
    st.caption("How do Diffusion Transformers differ from U-Net architectures for video processing?")
    st.caption("What are the primary security vulnerabilities when deploying autonomous OpenClaw agents?")
    st.caption("How can prompt injection attacks be prevented from hijacking an agent loop?")

    
    # Initialize system
    if not st.session_state.initialized:
        with st.spinner("Loading system..."):
            rag_system, num_chunks = initialize_rag()
            if rag_system:
                st.session_state.rag_system = rag_system
                st.session_state.initialized = True
                st.success(f"✅ System ready! ({num_chunks} document chunks loaded)")
    
    st.markdown("---")
    
    # Chat interface: show previous messages and use native chat input
    # Display existing conversation messages
    # st.chat_message("user") in Streamlit Docs creates a visual container styled for a user chat bubble
    # "assistant" sets up the correct visual layout, alignment, and default robot-style avatar for an AI or bot
    for item in st.session_state.history:
        with st.chat_message("user"):
            st.write(item.get('question', ''))
        with st.chat_message("assistant"):
            st.write(item.get('answer', ''))

    # Chat input
    user_input = st.chat_input("Ask a question...")
    if user_input:
        if st.session_state.rag_system:
            with st.spinner("Searching..."):
                start_time = time.time()

                # Get answer
                result = st.session_state.rag_system.run(user_input)

                elapsed_time = time.time() - start_time

                # Add to history
                st.session_state.history.append({
                    'question': user_input,
                    'answer': result['answer'],
                    'time': elapsed_time
                })

                # Immediately display the new chat messages
                with st.chat_message("user"):
                    st.write(user_input)
                with st.chat_message("assistant"):
                    st.write(result['answer'])

                # Show retrieved docs in expander
                with st.expander("📄 Source Documents"):
                    for i, doc in enumerate(result.get('retrieved_docs', []), 1):
                        st.text_area(
                            f"Document {i}",
                            doc.page_content[:300] + "...",
                            height=100,
                            disabled=True
                        )

                # Display metrics in two columns: response time and retrieval count
                retrieved_count = len(result.get('retrieved_docs', []))
                col1, col2 = st.columns(2)
                col1.metric(label="Response Time", value=f"{elapsed_time:.2f}s")
                col2.metric(label="Retrieved Docs", value=str(retrieved_count))
    
    # Recent Searches in the sidebar 
    if st.session_state.history:
        with st.sidebar:
            st.markdown("---")
            st.markdown("### 📜 Recent Searches")
            for item in reversed(st.session_state.history[-3:]):  # Show last 3
                st.markdown(f"**Q:** {item['question']}")
                st.markdown(f"**A:** {item['answer'][:100]}...")
                st.markdown("")

if __name__ == "__main__":
    main()