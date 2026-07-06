import os
import warnings
import numpy as np
from dotenv import load_dotenv

# Core LangChain Imports
from langchain_openai import OpenAIEmbeddings

# Ignore clean-up or deprecation warnings
warnings.filterwarnings('ignore')
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# ==========================================
# 1. INITIALIZE EMBEDDING MODEL
# ==========================================
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=1536
)

# ==========================================
# 2. SEMANTIC SIMILARITY LOGIC (Cosine Similarity)
# ==========================================
def compare_embeddings(text1: str, text2: str):
    """
    Computes the cosine similarity between two text strings using OpenAI Embeddings.
    A score near 1.0 indicates high contextual/semantic similarity.
    A score near 0.0 indicates no semantic relationship.
    """
    emb1 = np.array(embeddings.embed_query(text1))
    emb2 = np.array(embeddings.embed_query(text2))
    
    # Calculate cosine similarity: (A · B) / (||A|| * ||B||)
    # `np.dot()` 
    #   computes the dot product (sum of element-wise products) of two arrays, 
    # `np.linalg` 
    #   provides a suite of linear algebra functions, including `norm()` to calculate vector length. 
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return similarity

# ==========================================
# 3. RUN SIMILARITY EXAMPLES
# ==========================================
print("===== Semantic Similarity Testing =====")
print(f"'AI' vs 'Artificial Intelligence': {compare_embeddings('AI', 'Artificial Intelligence'):.3f}")
print(f"'AI' vs 'Pizza':                 {compare_embeddings('AI', 'Pizza'):.3f}")
print(f"'Machine Learning' vs 'ML':     {compare_embeddings('Machine Learning', 'ML'):.3f}")
print("-" * 40)