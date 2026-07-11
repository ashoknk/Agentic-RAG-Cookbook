"""
================================================================================
This script explores cloud-native vector persistence utilizing Pinecone, a 
purpose-built, high-performance serverless vector database engine. It introduces 
explicit **Index Management** workflows to teach students how remote database indices 
are provisioned and maintained at scale.

1. ENVIRONMENT VALIDATION: Performs defensive checkups to protect against empty 
   or mock API keys before consuming cloud compute resources.
2. REMOTE INDEX PROVISIONING: Uses the native Pinecone client to dynamically query the 
   cloud environment. If the targeted index doesn't exist, it configures and 
   spins up a new serverless vector cluster matching precise embedding dimensions.
3. EMBEDDING PIPELINE: Pairs the Pinecone index with an OpenAI 1024-dimension 
   embedding model via `PineconeVectorStore` to orchestrate automatic cloud ingestion.
4. RELEVANCE SEARCH TESTING: Demonstrates deep database-level metadata filtering 
   and handles edge-case testing by exposing how score thresholds successfully prune 
   out irrelevant background noise.
================================================================================
"""

import os
import warnings
from dotenv import load_dotenv

# 1. Load environment variables FIRST to ensure API keys are available
load_dotenv()

# LangChain Core and Model Imports
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

# Vector Store - Using the modern class name for LangChain 1.x
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. INITIAL PREPARATION: Configuration & Credentials
# ==============================================================================

# Credentials from environment
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Safety Check: Ensure keys are not empty strings or placeholders
if not OPENAI_API_KEY or OPENAI_API_KEY.startswith("sk-proj-XXXX"):
    raise ValueError("Valid OPENAI_API_KEY not found. Please check your .env file.")

# Pinecone Setup
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
INDEX_NAME = "pulseindex"

# Initialize OpenAI Embeddings
# We use 1024 dimensions to match your previous setup
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=1024
)

# Initialize Pinecone Client
pc = Pinecone(api_key=PINECONE_API_KEY)

# ==============================================================================
# 2. INDEX MANAGEMENT: Ensuring the index exists
# ==============================================================================

# Check for existing index using modern list_indexes().names()
if INDEX_NAME not in pc.list_indexes().names():
    print(f"Creating new index: {INDEX_NAME}...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=1024,  # Must match embedding dimensions
        metric="cosine",
        spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION)
    )
else:
    print(f"Index '{INDEX_NAME}' already exists. Connecting...")

# Connect to the index and initialize the VectorStore
index = pc.Index(INDEX_NAME)
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# ==============================================================================
# 3. DATA INGESTION: Documents Creation
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

# Add documents to Pinecone (triggers the OpenAI embedding call)
vector_store.add_documents(documents=documents)
print(f"✅ Successfully added {len(documents)} documents to Pinecone.")

# ==============================================================================
# 4. RETRIEVAL: Similarity Search & Filtering
# ==============================================================================

print("\n--- 1.Similarity Search (Tweets only) similarity_search() ---")
# question1 = "How to build AI agents?"

question1 = "Where can I find step-by-step web development guides and coding tutorials?"

results = vector_store.similarity_search(
    question1,
    k=1,
    filter={"source": "tweet"} # Metadata filtering happens at the DB level
)
print(f"Question: {question1}")
for doc in results:
    print(f"* {doc.page_content} [{doc.metadata}]")


print("\n--- 2.Similarity Search with Scores (News only) similarity_search_with_score() ---")
# question2 = "Was there a robbery?"
question2 = "What major software engineering tasks or global tech topics are currently being discussed?"
results_with_score = vector_store.similarity_search_with_score(
    question2, 
    k=1, 
    filter={"source": "news"}
)
print(f"Question: {question2}")
for doc, score in results_with_score:
    print(f"* [Score={score:.4f}] {doc.page_content}")

# ==============================================================================
# 5. RETRIEVER: LangChain Runnable Interface
# ==============================================================================

# Strategy A: Score Threshold
# Threshold search ensures we only get relevant results
retriever_threshold = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 1, "score_threshold": 0.4},
)

print("\n--- 3.Retriever Result as_retriever() ---")

question3 = "Tell me about breakfast"
retrieved_docs = retriever_threshold.invoke(question3)
print(f"Question: {question3}")
for doc in retrieved_docs:
    print(f"Found: {doc.page_content}")



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