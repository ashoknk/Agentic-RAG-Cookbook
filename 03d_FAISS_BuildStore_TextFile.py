import os
import warnings
from dotenv import load_dotenv

# LangChain Core and Embedding Imports
from langchain_openai import OpenAIEmbeddings

# Document Processing & FAISS Components
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS

# Ignore clean-up or deprecation warnings
warnings.filterwarnings('ignore')
load_dotenv()

# ==============================================================================
# 1. INITIAL PREPARATION: Environment & Directory Config
# ==============================================================================

# Fix for potential macOS OpenMP duplication crash
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

DATA_DIR = "data"
# FAISS saves to a directory folder containing a .faiss and .pkl file
FAISS_PERSIST_DIRECTORY = "./faiss_textfile" 

# Initialize Embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=1536
)

# ==============================================================================
# 2. DOCUMENT LOADING & TRANSFORMATION
# ==============================================================================
# Load raw text documents from the local data directory
loader = DirectoryLoader(
    DATA_DIR, 
    glob="*.txt", 
    loader_cls=TextLoader,
    loader_kwargs={'encoding': 'utf-8'}
)
documents = loader.load()
print(f"Loaded {len(documents)} documents from '{DATA_DIR}'.")

# Split documents into smaller semantic chunks for better retrieval
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=[" "]
)
chunks = text_splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks from local files.")

# ==============================================================================
# 3. VECTOR STORE CREATION (FAISS Local Persistence)
# ==============================================================================

# Generate vectors and load them into a local FAISS index
print(f"Building FAISS Vector Store from text files in {DATA_DIR}...")
vectorstore = FAISS.from_documents(
    documents=chunks, 
    embedding=embeddings
)

# Explicitly save FAISS database index locally to disk
vectorstore.save_local(FAISS_PERSIST_DIRECTORY)

print(f"FAISS Vector store built with {vectorstore.index.ntotal} vectors.")
print(f"Vector store saved successfully to the '{FAISS_PERSIST_DIRECTORY}/' directory.\n")