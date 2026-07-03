import os
import warnings
from dotenv import load_dotenv

# Core LangChain Imports
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

warnings.filterwarnings('ignore')
load_dotenv()

# Add this BEFORE importing any vector store or heavy math libraries
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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
# 2. LOAD FAISS VECTOR STORE FROM DISK
# ==========================================
if not os.path.exists(FAISS_INDEX):
    raise FileNotFoundError(f"The local index path '{FAISS_INDEX}' was not found. Please run the build script first!")

print(f"Loading FAISS index from '{FAISS_INDEX}'...")
loaded_vectorstore = FAISS.load_local(
    FAISS_INDEX,
    embeddings,
    allow_dangerous_deserialization=True  # Required to safely unpack pickle files locally
)
print(f"Loaded FAISS vector store contains {loaded_vectorstore.index.ntotal} vectors.\n")
print("-" * 60)

# ==========================================
# 3. SIMILARITY SEARCH VARIATIONS
# ==========================================
query = "What is deep learning"
print(f"Target Query: '{query}'\n")

# --- Option A: Standard Similarity Search ---
results = loaded_vectorstore.similarity_search(query, k=3)

print("--- A. Top 3 Similar Chunks (Standard Search) ---")
for i, doc in enumerate(results):
    print(f"{i+1}. Source: {doc.metadata['source']}")
    print(f"   Content: {doc.page_content.strip()[:180]}...\n")

# --- Option B: Similarity Search with Distance Scores ---
# Note: FAISS uses L2 distance (Euclidean). Lower scores mean closer/more similar items.
results_with_scores = loaded_vectorstore.similarity_search_with_score(query, k=3)

print("--- B. Similarity Search with Scores (L2 Distance) ---")
for doc, score in results_with_scores:
    print(f"Score (Lower=Better): {score:.3f} | Source: {doc.metadata['source']}")
    print(f"Content Preview: {doc.page_content.strip()[:100]}...\n")

# --- Option C: Search with Metadata Filtering ---
# NOTE: FAISS filters post-retrieval. If your top 'k' results don't match the criteria, 
# you will receive fewer results than requested.
print("--- C: Search with Metadata Filtering ---")
filter_dict = {"topic": "ML"}
filtered_results = loaded_vectorstore.similarity_search(
    query,
    k=3,
    filter=filter_dict
)

print(f"Filtered Similarity Search (Topic Filter: {filter_dict})")
if filtered_results:
    for i, doc in enumerate(filtered_results):
        print(f"Match {i+1}: Source: {doc.metadata['source']} | Topic: {doc.metadata['topic']}")
        print(f"Content: {doc.page_content.strip()[:120]}...\n")
else:
    print("No chunks found matching the filter criteria.")