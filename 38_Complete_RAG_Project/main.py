"""
Main entry point for the Agentic RAG application.

This file 
- initializes the document ingestion pipeline, 
- creates a vector store,
- builds the retrieval and reasoning graph, and 
- provides a simple interface for asking questions or running the system in interactive mode.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.config.config import Config
from src.document_ingestion.document_processor import DocumentProcessor
from src.vectorstore.vectorstore import VectorStore
from src.graph_builder.graph_builder import GraphBuilder

class AgenticRAG:
    """Main Agentic RAG application"""
    
    def __init__(self, urls=None):
        """
        Initialize Agentic RAG system including state and dependencies
        
        Args:
            urls: List of URLs to process (uses defaults if None)
        """
        print("🚀 Initializing Agentic RAG System...")
        
        # Use default URLs if none provided
        # self.urls = urls or Config.DEFAULT_URLS
        self.sources = urls or Config.DEFAULT_URLS
        
        # Initialize components
        self.llm = Config.get_llm()
        # we need doc_processor for _setup_vectorstore
        self.doc_processor = DocumentProcessor(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        self.vector_store = VectorStore()
        
        # Process documents and create vector store
        self._setup_vectorstore()
        
        # Build graph
        self.graph_builder = GraphBuilder(
            retriever=self.vector_store.get_retriever(),
            llm=self.llm
        )
        self.graph_builder.build()
        
        print("✅ System initialized successfully!\n")
    


    def _setup_vectorstore(self):
        """Setup vector store with processed documents. Handles the actual document ingestion"""
        print(f"📄 Processing {len(self.sources)} URLs...")
        documents = self.doc_processor.process_sources(self.sources)
        print(f"📊 Created {len(documents)} document chunks")
        print("🔍 Creating vector store...")
        self.vector_store.create_vectorstore(documents)

    
    def ask(self, question: str) -> str:
        """
        Ask a question to the RAG system
        
        Args:
            question: User question
            
        Returns:
            Generated answer
        """
        print(f"❓ Question: {question}\n")
        print("🤔 Processing...")
        
        result = self.graph_builder.run(question)
        answer = result['answer']
        
        print(f"✅ Answer: {answer}\n")
        return answer
    
    def interactive_mode(self):
        """Run in interactive mode"""
        print("💬 Interactive Mode - Type 'quit' to exit\n")
        
        while True:
            question = input("Enter your question (type q to quit): ").strip()
            
            if question.lower() in ['quit', 'exit', 'q', 'x']:
                print("👋 Goodbye!")
                break
            
            if question:
                self.ask(question)
                print("-" * 80 + "\n")

    
def load_sources_from_file():    
    sources = []

    urls_file = Path("data/urls.txt")
    if urls_file.exists():
        with open(urls_file, "r") as f:
            sources.extend([line.strip() for line in f if line.strip()])

    pdf_dir = Path("data")  
    if pdf_dir.exists() and pdf_dir.is_dir():
        sources.append(str(pdf_dir))

    # print(f"main()- Loading sources: {sources}")   #NOTE Testing print statement to verify sources loaded from file and PDF directory      

    return sources or Config.DEFAULT_URLS


def main():
    """Main function"""

    # Initialize RAG system
    sources = load_sources_from_file()
    rag = AgenticRAG(urls=sources)
    
    ##### ---- Example questions ---- ##### 

    # Group 1: Autonomous Agents & Memory Architectures
    # Answered by:
    # - Memory_for_Autonomous_LLM_Agents_Mechanisms,_Evaluation,_and_Emerging_Frontiers.pdf
    # - Security_of_OpenClaw_Agents.pdf
    # - https://arxiv.org/abs/2603.07670
    # - https://arxiv.org/html/2602.22406v1
    query1 = "What is the difference between short-term and long-term memory in LLM agents?"

    # Answered by:
    # - Openclaw_Review_.pdf
    # - Security_of_OpenClaw_Agents.pdf
    query2 = "What are the core components of the OpenClaw execution framework?"

    # Answered by:
    # - Memory_for_Autonomous_LLM_Agents_Mechanisms,_Evaluation,_and_Emerging_Frontiers.pdf
    # - Openclaw_Review_.pdf
    # - Security_of_OpenClaw_Agents.pdf
    # - https://arxiv.org/html/2602.22406v1
    query3 = "How does the interaction loop operate between agent memory, the LLM, and tools?"

    # Group 2: Generative Video & Diffusion Models
    # Answered by:
    # - Diffusion_Models_Comprehensive_Survey_of_Methods_and_Applications.pdf
    # - Efficient_Video_Diffusion_Models_Advancements_and_.pdf
    # - https://arxiv.org/html/2601.07823v1
    # - https://arxiv.org/abs/2505.14357
    query4 = "What are the main challenges when extending diffusion models to video generation?"

    # Answered by:
    # - Efficient_Video_Diffusion_Models_Advancements_and_.pdf
    # - https://arxiv.org/html/2601.07823v1
    query5 = "What strategies make video diffusion models more computationally efficient?"

    # Answered by:
    # - Efficient_Video_Diffusion_Models_Advancements_and_.pdf
    # - https://arxiv.org/html/2601.07823v1
    query6 = "How do Diffusion Transformers differ from U-Net architectures for video processing?"

    # Group 3: Agent Security & Edge Cases
    # Answered by:
    # - Openclaw_Review_.pdf
    # - Security_of_OpenClaw_Agents.pdf
    query7 = "What are the primary security vulnerabilities when deploying autonomous OpenClaw agents?"

    # Answered by:
    # - Memory_for_Autonomous_LLM_Agents_Mechanisms,_Evaluation,_and_Emerging_Frontiers.pdf
    # - Openclaw_Review_.pdf
    # - Security_of_OpenClaw_Agents.pdf
    query8 = "How can prompt injection attacks be prevented from hijacking an agent loop?"

    example_questions = [
        query1,
        query2,
        query3,
        query4,
        query5,
        query6,
        query7,
        query8
    ]
    
    print("=" * 80)
    print("📝 Running example questions:")
    print("=" * 80 + "\n")
    
    for question in example_questions:
        rag.ask(question)
        print("=" * 80 + "\n")
    
    # Optional: Run interactive mode
    print("\n" + "=" * 80)
    user_input = input("Would you like to enter interactive mode? (y/n): ")
    if user_input.lower() == 'y':
        rag.interactive_mode()

if __name__ == "__main__":
    main()