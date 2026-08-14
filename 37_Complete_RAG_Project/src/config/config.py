"""Configuration module for Agentic RAG system"""

import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for RAG system"""
    
    os.environ["USER_AGENT"] = "38_Complete_RAG_Project/1.0"
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Model Configuration
    LLM_MODEL = "openai:gpt-4o"

    PDF_FOLDER = "data"
    
    # Document Processing
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    
    # Default URLs
    DEFAULT_URLS = [
        "https://arxiv.org/html/2602.22406v1",
        "https://arxiv.org/html/2601.07823v1",
        "https://arxiv.org/abs/2603.07670",
        "https://arxiv.org/abs/2505.14357"
    ]
    
    # @classmethod is used to create methods bound to the class itself rather than a single object instance
    @classmethod
    def get_llm(cls):
        """Initialize and return the LLM model"""
        os.environ["OPENAI_API_KEY"] = cls.OPENAI_API_KEY
        return init_chat_model(cls.LLM_MODEL)