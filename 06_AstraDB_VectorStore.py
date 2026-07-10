"""
================================================================================
This script transitions the codebase from volatile local RAM to a persistent, 
enterprise-ready production architecture using DataStax Astra DB. Astra DB is a 
cloud-native, serverless vector database built on Apache Cassandra, providing 
scalable cloud storage for vectorized applications.

1. SECURITY & CONNECTION: Validates and authenticates secure cloud connection 
   endpoints and tokens loaded safely from environmental configurations.
2. CLOUD INGESTION: Instantiates `AstraDBVectorStore`. Passing raw text documents 
   automatically triggers a network call to the OpenAI API to calculate embeddings 
   before committing them over the wire to the cloud collection.
3. ADVANCED METADATA FILTERING: Demonstrates database-level filtering using metadata 
   properties (e.g., scoping search spaces strictly to `{"source": "tweet"}`).
4. ADVANCED RETRIEVAL SEARCH TYPES:
   - **Similarity Score Threshold**: Introduces a fallback strictness ceiling, 
     pruning away irrelevant matches that drop below a semantic confidence metric.
   - **MMR (Maximum Marginal Relevance)**: Minimizes redundancy in the retrieved output, 
     balancing strict query alignment with informational diversity.
================================================================================
"""

import os
import warnings
from dotenv import load_dotenv

# LangChain Core and Vector Store Imports
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_astradb import AstraDBVectorStore

# Suppress unnecessary warnings and load environment variables
warnings.filterwarnings('ignore')
load_dotenv()

# ==============================================================================
# 1. INITIAL PREPARATION: Configuration & Credentials
# ==============================================================================

# Ensure your .env file contains OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["ASTRA_DB_API_ENDPOINT"] = os.getenv("ASTRA_DB_API_ENDPOINT")
os.environ["ASTRA_DB_APPLICATION_TOKEN"] = os.getenv("ASTRA_DB_APPLICATION_TOKEN")

# Astra DB Connection Details

# Initialize OpenAI Embeddings
# text-embedding-3-small is cost-effective and high-performing
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=512)

# Initialize the AstraDB Vector Store
# Astra DB is a cloud-native, serverless vector database
vector_store = AstraDBVectorStore(
    embedding=embeddings,
    api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
    collection_name="astra_vector_langchain",
    token=os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
    namespace=None,
)

# ==============================================================================
# 2. DATA INGESTION: Documents Creation & Upload
# ==============================================================================
documents = [
    Document(page_content="Just spilled coffee all over my keyboard right before a major demo. Peak Monday vibes.", metadata={"source": "tweet"}),
    Document(page_content="Can anyone recommend a good lightweight code editor for Mac? Thinking of switching.", metadata={"source": "tweet"}),
    Document(page_content="Finally hit 10k followers today! Huge thank you to everyone who supports my tech journey.", metadata={"source": "tweet"}),
    Document(page_content="Spent the weekend refactoring my codebase and honestly, it's so satisfying.", metadata={"source": "tweet"}),
    Document(page_content="Local authorities have issued a flash flood warning for the tri-state area until midnight.", metadata={"source": "news"}),
    Document(page_content="The global tech summit kicks off tomorrow with major announcements expected in AI regulation.", metadata={"source": "news"}),
    Document(page_content="Scientists discover a new deep-sea coral species off the coast of Australia.", metadata={"source": "news"}),
    Document(page_content="The city transit authority announced a 10% fare increase starting next month.", metadata={"source": "news"}),
    Document(page_content="A beginner's guide to mastering CSS Grid and Flexbox layouts in 2026.", metadata={"source": "website"}),
    Document(page_content="Top 5 meal prep strategies for busy working professionals looking to eat healthier.", metadata={"source": "website"}),
    Document(page_content="Click here for our comprehensive, step-by-step tutorial on deploying your first Next.js app.", metadata={"source": "website"}),
    Document(page_content="Discover the best budget-friendly hiking gear for your next outdoor adventure.", metadata={"source": "website"}),
]

# Add documents to Astra DB (this embeds and stores them in the cloud)
vector_store.add_documents(documents=documents)
print(f"✅ Successfully added {len(documents)} documents to Astra DB.")

# ==============================================================================
# 3. DIRECT SEARCH: Similarity & Filtering
# ==============================================================================

print("\n--- 1. Similarity Search (Filtered by Source: Tweet) similarity_search() ---")
# Matches Doc 9 (CSS Grid/Flexbox guide) and Doc 11 (Next.js deployment tutorial)
question1 = "Where can I find step-by-step web development guides and coding tutorials?"

# filter allows you to narrow down search results based on metadata
results = vector_store.similarity_search(
    question1,
    k=1,
    filter={"source": "tweet"},
)
print(f"Question: {question1}")
for res in results:
    print(f'* Content: "{res.page_content}" | Source: {res.metadata["source"]}')

print("\n--- 2. Similarity Search with Scores similarity_search_with_score()---")
# Returns a tuple of (Document, Score)
results_with_score = vector_store.similarity_search_with_score(
    question1,
    k=1,
    filter={"source": "tweet"},
)
print(f"Question: {question1}")
for res, score in results_with_score:
    print(f'* [Score={score:.4f}] "{res.page_content}"')

# ==============================================================================
# 4. ADVANCED RETRIEVER STRATEGIES
# ==============================================================================

print("\n--- 3. Retriever: Similarity Score Threshold ---")

# Matches Doc 4 (refactoring codebase) and Doc 6 (tech summit/AI regulation news)
question2 = "What major software engineering tasks or global tech topics are currently being discussed?"

# Strategy A: Score Threshold
# Only returns documents that meet a minimum similarity score
retriever_threshold = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 1, "score_threshold": 0.5},
)

print(f"Question: {question2}")
threshold_res = retriever_threshold.invoke(question2, filter={"source": "news"})
for doc in threshold_res:
    print(f'* Found: "{doc.page_content}"')

# Strategy B: MMR (Maximum Marginal Relevance)
# MMR tries to find a balance between relevance and diversity in results
retriever_mmr = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 1},
)

print("\n--- 4. Retriever: MMR (Diversity Search) ---")
mmr_res = retriever_mmr.invoke(question2, filter={"source": "news"})
for doc in mmr_res:
    print(f'* Found: "{doc.page_content}"')