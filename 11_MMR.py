"""
Maximal Marginal Relevance
MMR (Maximal Marginal Relevance) is a powerful diversity-aware retrieval technique used in information retrieval and RAG pipelines to balance relevance and novelty when selecting documents.
Imagine you ask a friend for book recommendations about space, and instead of giving you three different books on astronauts, they give you one on astronauts, one on planets, and one on alien life. That is exactly what **MMR** does for your AI search engine. While standard search only looks for the closest matching answers, MMR actively penalizes repetitive information. It forces the database to select chunks that are highly relevant to your question, but completely different from one another in text content. This ensures your LLM gets a well-rounded, diverse snapshot of information instead of reading the exact same fact repeated three times.
"""

"""
================================================================================
This script introduces **Maximal Marginal Relevance (MMR)**, an advanced, 
diversity-aware retrieval algorithm. While normal vector search exclusively 
grabs the closest documents in vector space (which often leads to repetitive text), 
MMR mathematically penalizes redundancy. It selects document chunks that are 
highly relevant to the user query yet distinctly different from one another, 
ensuring the downstream LLM receives a comprehensive, well-rounded overview.


1. TEXT INGESTION: Loads a multi-topic cybersecurity reference dataset and splits it into fine-grained character segments.
2. FAISS EMBEDDING ROUTING: Populates a local FAISS index using vector representations 
   generated via a local HuggingFace embedding engine (`all-MiniLM-L6-v2`).
3. RETRIEVER DIVERSIFICATION: Switches the vector store strategy by explicitly setting 
   `search_type="mmr"`. This instructs the underlying database engine to swap 
   its retrieval math to filter out overlapping context clusters automatically.
4. END-TO-END VALIDATION: Executes complex multi-domain infrastructure queries 
   to demonstrate how MMR extracts highly distinct, non-repetitive chunks  across disparate topics.
================================================================================
"""

import os
import logging
import warnings
from dotenv import load_dotenv

# LangChain Core & Text Splitters
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser 

# Models & Vector Stores
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import init_chat_model

# Chains & Pipelines
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain

# ==============================================================================
# 1. INITIAL PREPARATION: Environment & Logging
# ==============================================================================

# Suppress noise: warnings, transformer loading logs, and progress bars
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# ==============================================================================
# 2. DATA INGESTION & CHUNKING
# ==============================================================================
# TODO change question to cybersecurity_data if needed
FILE_NAME = "cybersecurity_data/cybersecurity_dataset.txt"
# Step 1: Load and split the document into smaller segments
try:
    loader = TextLoader(FILE_NAME)
    raw_docs = loader.load()
    
    # Using a smaller chunk size to see the impact of MMR diversity
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_documents(raw_docs)
except Exception as e:
    print(f"Error loading dataset: {e}. Ensure '{FILE_NAME}' exists.")
    chunks = []

# ==============================================================================
# 3. VECTOR STORE & MMR RETRIEVER SETUP
# ==============================================================================

MODEL_EMBEDDING = "all-MiniLM-L6-v2"
# Step 2: Initialize embeddings via HuggingFace
embedding_model = HuggingFaceEmbeddings(model_name=MODEL_EMBEDDING)

# Build the FAISS vector index
vectorstore = FAISS.from_documents(chunks, embedding_model)

# Step 3: Create MMR Retriever
# 'search_type="mmr"' triggers the Maximal Marginal Relevance algorithm
# 'k=3' selects the 3 most diverse and relevant documents
# MMR is a formula that balances relevance with diversity. It prevents your AI from reading the same information over 
# and over again by forcing the search engine to pick documents that are both relevant to your question and 
# distinct from each other.
retriever = vectorstore.as_retriever(
    search_type="mmr", 
    search_kwargs={"k": 3}
)

# When you set search_type="mmr", LangChain's native vector store class (like FAISS or Chroma) automatically catches that keyword parameter. Instead of calling its standard Euclidean distance or Cosine similarity method, the vector store shifts its internal mathematical routing to execute its built-in MMR diversification function.

# You get the full benefit of the algorithm purely through that single configuration string without needing any extra imports!

# ==============================================================================
# 4. RAG PIPELINE CONSTRUCTION
# ==============================================================================

# Step 4: Setup Prompt and LLM
prompt = PromptTemplate.from_template("""
Answer the question based on the context provided.

Context:
{context}

Question: {input}
""")

# Initialize high-speed LLM (Groq Llama 3.1)
MODEL_NAME_CHAT = "groq:llama-3.1-8b-instant"
llm = init_chat_model(MODEL_NAME_CHAT)

# Step 5: Construct the RAG Pipeline
# 'create_stuff_documents_chain' formats documents into the prompt
document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

# 'create_retrieval_chain' links the retriever to the document processor
rag_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=document_chain)

# ==============================================================================
# 5. EXECUTION & OUTPUT
# ==============================================================================

# Step 6: Define and run the Query
# Expected MMR Pull: One chunk from IAM/PAM , one from DevSecOps , and one from EDR/XDR
query = {"input": "How can an enterprise protect its data across identities, software pipelines, and user endpoints?"}

# Expected MMR Pull: One chunk from API Security/OWASP flaws , one from CSPM/Cloud misconfigurations , and one from automated pipeline remediation
query = {"input": "What are the primary ways attackers exploit corporate vulnerabilities, and how do automated tools mitigate these risks?"}

# Expected MMR Pull: One chunk from Zero Trust Architecture , one from CSPM , and one from EDR/XDR behavior tracking
query = {"input": "What is the modern strategy for securing enterprise infrastructure both on-premises and in the cloud?"}

response = rag_chain.invoke(query)
print("🔍 Query :{}".format(query["input"]))
print("\n✅ Final Answer:")
print("-" * 20)
print(response["answer"])

# Because MMR was active, it forced a balance of relevance and diversity:
# First Pick (Relevance): It grabbed a highly relevant chunk about how LangChain handles Agents 
# (Sources 6, 14, or 16).
# Next Picks (Diversity Penalty): When it looked at the remaining chunks, it saw other paragraphs 
# talking about agents. It penalized them for being redundant and intentionally picked chunks that were 
# different—safely capturing Conversational Memory (Source 5) and ConversationSummaryMemory (Source 15).

# Demonstrate MMR:
print("\n📚 Raw Chunks Retrieved by MMR (Notice the structural diversity):")
print("-" * 50)
for i, doc in enumerate(response["context"]):
    print(f"Chunk {i+1}: {doc.page_content}")