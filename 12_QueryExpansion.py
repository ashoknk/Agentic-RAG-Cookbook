
"""
================================================================================
This script showcases **Query Expansion**
🎯 What is Query Expansion/Enhancement?
Query enhancement (Query Expansion Techniques) refers to techniques used to improve or reformulate the user query to retrieve better, more relevant documents from the knowledge base. 
It is especially useful when:
- The original query is short, ambiguous, or under-specified.
- You want to broaden the scope to catch synonyms, related phrases, or spelling variants
- By expanding the query, you can increase the chances of retrieving relevant documents that may not match the original query verbatim but are still highly pertinent to the user’s intent.

This architecture forces an LLM to preprocess and rewrite the prompt, loading it 
with rich technical vocabulary, domain synonyms, and deeper context *before* the vector database is ever touched.

1. VECTOR PREPARATION: Mounts a cybersecurity text knowledge base into a FAISS  vector store leveraging HuggingFace embedding calculations.
2. QUERY EXPANSION CHAIN: Configures an independent, isolated LCEL sequence that 
   intercepts the user's raw string and instructs an LLM (`o4-mini`) to expand it 
   into a technically dense search paragraph.
3. LCEL RUNNABLEMAP COORDINATION: Chains the components into a single pipeline. 
   The output of the query expansion chain is dynamically routed as the search criteria for the vector store's MMR retriever.
4. STREAMLINED RESPONSE: Passes the rich, expanded-context documents into the final 
   answering prompt template to generate highly accurate summaries.
================================================================================
"""

import os
import logging
import warnings
from dotenv import load_dotenv

# LangChain Core & Text Splitters
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser 
from langchain_core.runnables import RunnableMap

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
# https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ==============================================================================
# 2. DATA INGESTION & VECTOR STORE SETUP
# ==============================================================================
FILE_NAME = "cybersecurity_data/cybersecurity_dataset.txt"
# Step 1: Load and split the dataset
try:
    loader = TextLoader(FILE_NAME)
    raw_docs = loader.load()
    
    # Split text into small chunks for granular retrieval
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_documents(raw_docs)

    # Step 2: Initialize Vector Store with HuggingFace Embeddings
    # https://reference.langchain.com/python/langchain-huggingface/embeddings/huggingface/HuggingFaceEmbeddings
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    # https://sj-langchain.readthedocs.io/en/latest/vectorstores/langchain.vectorstores.faiss.FAISS.html
    vectorstore = FAISS.from_documents(chunks, embedding_model)

    # Step 3: Setup MMR Retriever (Diversity-aware retrieval)
    # This helps find diverse documents related to the expanded query
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5})
except Exception as e:
    print(f"Initialization error: {e}. Please ensure '{FILE_NAME}' exists.")
    chunks = []

# ==============================================================================
# 3. QUERY EXPANSION & RAG PROMPTS
# ==============================================================================

# Initialize the LLM (Using OpenAI o4-mini for expansion and reasoning)
CHAT_MODEL = "openai:o4-mini"
llm = init_chat_model(CHAT_MODEL)

# Step 4: Define the Query Expansion Chain
# This chain transforms short user inputs into descriptive search queries
# By passing the raw, brief user query into this prompt first, the code forces 
# the LLM to output a paragraph dense with technical terminology, alternative phrasings, 
# and synonyms before any vector search happens.

query_expansion_prompt = PromptTemplate.from_template("""
You are a helpful assistant. Expand the following query to improve document retrieval by adding relevant synonyms, technical terms, and useful context.

Original query: "{query}"

Expanded query:
""")

query_expansion_chain = query_expansion_prompt | llm | StrOutputParser()

# Step 5: Define the Final Answering Chain
answer_prompt = PromptTemplate.from_template("""
Answer the question based on the context below.

Context:
{context}

Question: {input}
""")

document_chain = create_stuff_documents_chain(llm=llm, prompt=answer_prompt)

# ==============================================================================
# 4. FULL RAG PIPELINE WITH QUERY EXPANSION
# ==============================================================================

# This pipeline uses RunnableMap to first expand the query, then retrieve context, 
# and finally generate the answer.
rag_pipeline = (
    RunnableMap({
        "input": lambda x: x["input"],
        "context": lambda x: retriever.invoke(
            query_expansion_chain.invoke({"query": x["input"]})
        )
    })
    | document_chain
)

# ==============================================================================
# 5. EXECUTION & TESTING
# ==============================================================================

def run_query(query_text):
    print(f"\n--- 🔍 Testing Query: '{query_text}' ---")
    
    # Optional: Preview the expanded query
    expanded_preview = query_expansion_chain.invoke({"query": query_text})
    print(f"✨ Expanded Query Preview: {expanded_preview.strip()}")
    
    # Run full RAG pipeline
    response = rag_pipeline.invoke({"input": query_text})
    print(f"\t✅ Final AI Answer:\n{response}")

# Run tests
run_query("IAM vs PAM for enterprise security?")
run_query("CI/CD for DevSecOps?")