"""
# ### HyDE # 🧠 What is HyDe?

# HyDE (Hypothetical Document Embeddings) is a retrieval technique where, instead of embedding 
# the user’s query directly, you first generate a hypothetical answer (document) to 
# the query using an LLM — and then embed that hypothetical document to search your vector store.

# ➡️ HyDE bridges the gap between user intent and relevant content, especially when:
# 1. Queries are short
# 2. Language mismatch between query and documents
# 3. You want to retrieve based on answer content, not question words
================================================================================
This script introduces **Hypothetical Document Embeddings (HyDE)** via a clear, 
step-by-step manual implementation. Standard vector search embeds a 
"Question," which often looks completely different from the "Answers" stored in a 
database. HyDE solves this by using an LLM to generate a "Fake Answer" 
(hypothetical document) first. We then embed that fake answer to query 
the database, matching the target document vocabulary with far greater precision.


1. DATASET SETUP: Uses a `WikipediaLoader` to pull biographical historical data 
   and chunks it into a persistent local Chroma vector database.
2. HYPOTHETICAL GENERATION (`get_hyde_doc`): Captures the short user prompt and 
   forces an LLM (`llama-3.1-8b-instant`) to draft an ungrounded essay on the topic.
3. BASELINE VS. HYDE DEMONSTRATION:
   - **Baseline Test**: Passes the raw question directly into the retriever.
   - **HyDE Test**: Converts the user query into the dense, fake essay text, 
     then passes that essay to the database to reveal superior contextual precision.
================================================================================
"""

import os
import logging
import warnings
from dotenv import load_dotenv

from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import SystemMessagePromptTemplate, ChatPromptTemplate
from langchain.chat_models import init_chat_model

# Suppress standard python warnings
warnings.filterwarnings("ignore")
# Suppress Transformer loading logs and general warnings
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
# Suppress the progress bar (The 100%|███ bar)
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# 1. Load and chunk your dataset
chunk_size = 300
chunk_overlap = 100

# loading data
loader = WikipediaLoader(query="Steve Jobs", load_max_docs=5)
documents = loader.load()

# text splitting
text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
docs = text_splitter.split_documents(documents=documents)

# Initialize Embeddings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# Set up the LLM you’ll use to generate hypothetical answers
load_dotenv()
os.environ["HF_TOKEN"]=os.getenv("HF_TOKEN")

os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
CHAT_MODEL="groq:llama-3.1-8b-instant"
llm = init_chat_model(CHAT_MODEL)

## creating vector store
# OUTPUT_DIR = "output/steve_jobs_for_hyde.db"
OUTPUT_DIR = "output/Abraham_Lincoln_for_hyde.db"
db = Chroma.from_documents(documents = docs,embedding=embeddings,persist_directory = OUTPUT_DIR)
##create the retriever
base_retriever=db.as_retriever(search_kwargs = {"k":5})

## Generating a prompt for generating HyDE. Instructs the LLM to write a fake response.
def get_hyde_doc(query):
    template = """Imagine you are an expert writing a detailed explanation on the topic: '{query}'
    create a hypothetical answer for the topic"""

    system_message_prompt = SystemMessagePromptTemplate.from_template(template = template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt])
    messages = chat_prompt.format_prompt(query = query).to_messages()
    response = llm.invoke(messages)
    hypo_doc = response.content
    return hypo_doc

# ==============================================================================
# PIPELINE EXECUTION & HYDE ADVANTAGE SHOWCASE
# ==============================================================================
# query = 'When was Steve Jobs fired from Apple?'
query = "When did Abraham Lincoln serve as the President of the United States?"
hypo_doc = get_hyde_doc(query=query)

print("\n" + "="*80)
print(f"🚀 HYDE BENEFIT DEMONSTRATION")
print("="*80)
print(f"Original Short User Query: '{query}'")

# --- Step A: Simulate Standard Retrieval (Baseline) ---
# We pass the raw question to the retriever to see the baseline
baseline_docs = base_retriever.invoke(query)

print("\n❌ CROSS-COMPARISON: Chunks retrieved via STANDARD query search:")
print("------------------------------------------------------------------------")
for i, doc in enumerate(baseline_docs[:2], 1): # Preview top 2 for brevity
    print(f"  Standard Match {i} | Source: {doc.metadata.get('title', 'Wiki')}")
    print(f"  Content Snippet: {doc.page_content.strip()[:140]}...")

# --- Step B: Show HyDE Expansion ---
print("\n✨ STEP 1: LLM generates a dense, answer-like 'Hypothetical Document':")
print("------------------------------------------------------------------------")
# Indent the hypothetical document text so it stands out cleanly
indented_hypo = "\n".join([f"    {line}" for i, line in enumerate(hypo_doc.split("\n")) if i < 15]) 
print(f"{indented_hypo}\n    ... [Truncated for Display] ...")

# --- Step C: Show HyDE Retrieval (The Winner) ---
# Call your database retriever using the fake response text as the input string
matched_doc = base_retriever.invoke(hypo_doc)

print("\n✅ STEP 2: Chunks retrieved via HYDE (Embedding the Answer Essay):")
print("------------------------------------------------------------------------")
for i, doc in enumerate(matched_doc[:2], 1): # Preview top 2 for comparison
    print(f"  HyDE Match {i} | Source: {doc.metadata.get('title', 'Wiki')}")
    print(f"  Content Snippet: {doc.page_content.strip()[:140]}...")

print("\n💡 ARCHITECTURAL LESSON:")
print("Standard search embeds a 'Question' which mathematically misaligns with stored 'Answers'.")
print("HyDE embeds a 'Fake Answer'. Because both the fake document and real database documents")
print("share an 'Answer format/vocabulary', vector similarity search achieves far higher precision!")
print("="*80)