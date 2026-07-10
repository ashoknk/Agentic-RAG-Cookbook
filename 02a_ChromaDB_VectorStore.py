"""
================================================================================
CORE INGESTION & PERSISTENT VECTOR STORE CONFIGURATION
================================================================================
Purpose:
    Demonstrates how to build a complete production-grade data ingestion pipeline
    using an on-disk vector database (ChromaDB) to turn raw text files into 
    searchable mathematical representations.

What it does:
    1. Ingestion: Dynamically batches and loads `.txt` documents from a local directory.
    2. Chunking: Applies a chunking strategy (`RecursiveCharacterTextSplitter`)
       to slice raw data into clean, semantic fragments optimized for LLM windows.
    3. Vector Storage: Employs OpenAI's text-embedding model to create 
       high-dimensional dense vectors and stores them persistently on the disk.
    4. Retrieval Mechanics: Demonstrates standard context lookups (`similarity_search`) 
       and advanced vector analysis queries (`similarity_search_with_score`).

Crucial Database Insight:
    Unlike general cosine similarity metrics where higher numbers represent better alignment, 
    ChromaDB natively uses **Squared L2 (Euclidean) Distance**. In this framework, **LOWER 
    scores represent high semantic alignment**, where 0.0 is a mathematically exact match.

Role in Agentic RAG:
    This file establishes the persistent "Non-Parametric Knowledge Base" that 
    the final RAG agent can query via standard vectors to answer domain-specific questions.
================================================================================
"""

import os
import warnings
from dotenv import load_dotenv

# LangChain Imports
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma

warnings.filterwarnings('ignore')
load_dotenv()

# https://reference.langchain.com/python/langchain-chroma/vectorstores/Chroma

# ==============================================================================
# 1. INITIAL PREPARATION: Environment & Sample Data
# ==============================================================================

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
DATA_DIR = "data"
PERSISTENT_DIRECTORY = "./chroma_db"
COLLECTION_NAME = "rag_collection"  # collection_name acts as a unique namespace or table name that groups and stores your specific embeddings

# ==============================================================================
# 2. DOCUMENT LOADING & TRANSFORMATION
# ==============================================================================

# Load documents from the local directory

loader = DirectoryLoader(
    DATA_DIR, 
    glob="*.txt", 
    loader_cls=TextLoader,
    loader_kwargs={'encoding': 'utf-8'}
)

# https://reference.langchain.com/python/langchain-community/document_loaders/directory/DirectoryLoader/load
documents = loader.load()
print(f"Loaded {len(documents)} documents.")

# Split documents into smaller chunks for better retrieval
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=[" "]
)
chunks = text_splitter.split_documents(documents)
print(f'Created {len(chunks)} chunks from {len(documents)} documents.')

# ==============================================================================
# 3. VECTOR STORE CREATION (ChromaDB)
# ==============================================================================

# Initialize embeddings and the persistent Chroma store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Unlike InMemoryVectorStore, Chroma persists data to disk
# Chroma.from_documents(...) It is an initialization method used for ingesting and saving new data.
# collection_name acts as a unique namespace or table name that groups and stores your specific embeddings
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=PERSISTENT_DIRECTORY,
    collection_name=COLLECTION_NAME
)

print(f"Vector store created with {vectorstore._collection.count()} vectors.")
print(f"Data persisted to: {PERSISTENT_DIRECTORY}")

# ==============================================================================
# 4. UTILITY: Formatted Output Helper
# ==============================================================================

def display_results(query: str, results: list, with_scores: bool = False):
    print(f"\n{'='*10} QUERY: {query} {'='*20}")
    for i, item in enumerate(results):
        # Handle both similarity_search (doc only) and similarity_search_with_score (doc, score)
        doc = item[0] if with_scores else item
        score = item[1] if with_scores else None
        
        print(f"\n--- Chunk {i+1} ---")
        print(f"Content: {doc.page_content[:150]}...")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        if with_scores:
            print(f"Similarity Score: {score:.4f}")

# ==============================================================================
# 5. EXECUTION: Similarity Search Examples
# ==============================================================================

#TODO put in loop format once with similar score and other with just similar
# test_questions = [
#     "What are the three types of machine learning?",
#     "What is deep learning and how does it relate to neural networks?",
#     "What are CNNs best used for?"
# ]


# Example 1: Standard Similarity Search
# ------------------------------------------------------------------------------
# HOW similarity_search() WORKS:
# 1. It takes your plain text string query.
# 2. It automatically sends that text to OpenAI's "text-embedding-3-small" model 
#    to convert it into a temporary query vector.
# 3. It runs a mathematical nearest-neighbor comparison against all the vectors 
#    saved inside your ChromaDB index to find the closest semantic matches.
#
# WHAT k=2 DOES:
# - 'k' stands for the number of matching documents you want to retrieve.
# - Setting k=2 restricts ChromaDB to return only the top 2 highest-ranked text 
#    chunks that best match the query. 
# ------------------------------------------------------------------------------

# Example 1: Standard Similarity Search
query_1 = "What are the types of machine learning?"
res1 = vectorstore.similarity_search(query_1, k=2)
display_results(query_1, res1)

# Example 2: Search with Relevance Scores
# ------------------------------------------------------------------------------
# HOW similarity_search_with_score() DIFFERS:
# - Instead of just returning a list of matching Document objects, it returns 
#   a list of Python tuples containing: (Document, Distance_Score).
#
# UNDERSTANDING THE SCORE (ChromaDB Specific):
# - ChromaDB defaults to Squared L2 (Euclidean) distance as its math metric.
# - Because it measures DISTANCE rather than absolute similarity percentage, 
#   a LOWER score indicates a closer vector match (Better/More Similar). 
#   A score of 0.0 would represent a perfect word-for-word identity match.
# ------------------------------------------------------------------------------
# Example 2: Search with Relevance Scores
query_2 = "What is Deep Learning?"
res2 = vectorstore.similarity_search_with_score(query_2, k=2)
display_results(query_2, res2, with_scores=True)

# Example 3: Search with Relevance Scores
query_3="what is NLP?"
results_scores=vectorstore.similarity_search_with_score(query_3,k=2)
display_results(query_3, results_scores, with_scores=True)


"""==============================================================================
CHROMADB RELEVANCE SCORE CHEAT SHEET (Squared L2 Distance)
------------------------------------------------------------------------------
Because ChromaDB measures the DISTANCE between vectors, LOWER scores mean 
the text chunks are closer together (Better/More Similar). 
Use these general brackets to judge your search results:

  0.0 to 0.5  --> Exceptional, near-exact word match.
  0.5 to 0.8  --> The "Sweet Spot". Excellent semantic match. The conceptual 
                  answer is definitely in this chunk.
  0.8 to 1.1  --> Loose match. Shares some related vocabulary but might miss 
                  the specific point.
  > 1.1       --> Weak match. Likely "filler" context returned because 'k' 
                  forced the database to return a document.
=============================================================================="""