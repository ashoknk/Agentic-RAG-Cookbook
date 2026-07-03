import os
import warnings
from dotenv import load_dotenv

# Core LangChain Imports
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Ignore clean-up or deprecation warnings
warnings.filterwarnings('ignore')
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
FAISS_INDEX = "faiss_index"

# ==========================================
# 1. INITIALIZE EMBEDDING MODEL
# ==========================================
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=1536
)

# ==========================================
# 2. DEFINE RAW DOCUMENTS
# ==========================================
sample_documents = [
    Document(
        page_content="""
        Artificial Intelligence (also called AI) is the simulation of human intelligence in machines.
        These systems are designed to think like humans and mimic their actions.
        AI can be categorized into narrow AI and general AI.
        """,
        metadata={"source": "AI Introduction", "page": 1, "topic": "AI"}
    ),
    Document(
        page_content="""
        Machine Learning (ML) is a subset of AI that enables systems to learn from data.
        Instead of being explicitly programmed, ML algorithms find patterns in data.
        Common types include supervised, unsupervised, and reinforcement learning.
        """,
        metadata={"source": "ML Basics", "page": 1, "topic": "ML"}
    ),
    Document(
        page_content="""
        Deep Learning (DL) is a subset of machine learning based on artificial neural networks.
        It uses multiple layers to progressively extract higher-level features from raw input.
        Deep learning has revolutionized computer vision, NLP, and speech recognition.
        """,
        metadata={"source": "Deep Learning", "page": 1, "topic": "DL"}
    ),
    Document(
        page_content="""
        Natural Language Processing (NLP) is a branch of AI that helps computers understand human language.
        It combines computational linguistics with machine learning and deep learning models.
        Applications include chatbots, translation, sentiment analysis, and text summarization.
        """,
        metadata={"source": "NLP Overview", "page": 1, "topic": "NLP"}
    )
]

# ==========================================
# 3. TEXT SPLITTING & CHUNKING
# ==========================================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
    separators=[" "]
)

chunks = text_splitter.split_documents(sample_documents)
print(f"Total Chunks Created: {len(chunks)}\n")

# ==========================================
# 4. CREATE AND SAVE FAISS VECTOR STORE
# ==========================================
print("Building FAISS Vector Store...")
vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)
print(f"FAISS Vector store built with {vectorstore.index.ntotal} vectors.")

# Serialize index locally to disk
vectorstore.save_local(FAISS_INDEX)
print(f"Vector store saved successfully to the '{FAISS_INDEX}/' directory.\n")