"""
================================================================================
This script transitions from the raw manual function mapping of the previous example 
to LangChain's production-ready abstraction class: **`HypotheticalDocumentEmbedder`**. 
It implements a pre-configured preset called `"web_search"`, which forces the internal 
LLM to generate hypothetical text structured like a standard, high-quality search engine answer page.


1. INGESTION ENGINE: Splices a multi-paragraph cybersecurity dataset into 
   granular 300-character segments.
2. WRAPPED HYDE INTERFACE: Instantiates `HypotheticalDocumentEmbedder.from_llm()`. 
   This seamlessly binds your 
    a.basic HuggingFace embedding calculations and 
    b. your Groq LLM 
        into a single, unified `hyde_embedding_function` object.
   Uses a pre-configured default template built into LangChain called "web_search".
3. SILENT QUERY ROUTING: The new object wraps around the vector store ingestion. 
   When `similarity_search()` is called, the class intercepts your short text query, 
   silently queries the LLM for a web-styled response, embeds it, and runs the 
   database vector lookup completely behind the scenes.
4. RAG RESPONSE GENERATION: Feeds the extracted semantic matches into a document 
   stuffing chain to return the final grounded answer.
================================================================================
"""

import logging
import os
import warnings
from dotenv import load_dotenv

from langchain_classic.chains.hyde.base import HypotheticalDocumentEmbedder
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain.chat_models import init_chat_model

from huggingface_hub import utils
import transformers

# Suppress warnings and heavy logs
warnings.filterwarnings("ignore")
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
utils.disable_progress_bars()
transformers.utils.logging.set_verbosity_error()

load_dotenv()
os.environ["HF_TOKEN"]=os.getenv("HF_TOKEN")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
llm = init_chat_model("groq:llama-3.1-8b-instant")

FILE_NAME = "cybersecurity_data/cybersecurity_dataset.txt"
# Step 1: Load and split documents
loader = TextLoader(FILE_NAME)
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# Step 2: Set up base embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
base_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# Step 3: Build HyDE Embedder using standard 'web_search' preset
# This uses a pre-defined prompt that mimics a web search result.
# Merge your standard embedding engine (HuggingFaceEmbeddings) and 
# your LLM into a singular object called hyde_embedding_function

OUTPUT_DIR = "output/langchain_web"
# prompt_key="web_search" - Write a hypothetical document in the style of a high-quality web search result 
# It doesn't perform a "web search."
#  HypotheticalDocumentEmbedder class is wrapped directly around your database.
hyde_embedding_function = HypotheticalDocumentEmbedder.from_llm(
    llm=llm,
    base_embeddings=base_embeddings,
    prompt_key="web_search" # <--- Built-in LangChain preset
)

# Step 4: Store chunks into Chroma DB using HyDE embeddings
# When you load documents into the database with Chroma.from_documents(..., embedding=hyde_embedding_function), 
# HyDE runs on every single document chunk as it is saved! The LLM generates a hypothetical question for every chunk 
# of your text file, and embeds those answer.
print("📦 Ingesting documents into  database...")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=hyde_embedding_function,
    persist_directory=OUTPUT_DIR
)


# Step 5: Construct the RAG Answer Generation Chain
rag_prompt = PromptTemplate.from_template("""
Use the context below to answer the question.

Context:
{context}

Question: {input}
""")
# Create a chain for passing a list of Documents to a model.
# https://reference.langchain.com/python/langchain-classic/chains/combine_documents/stuff/create_stuff_documents_chain
rag_chain = create_stuff_documents_chain(llm=llm, prompt=rag_prompt)

# Step 6: Define full pipeline search and synthesis loop
# Under the hood, the HypotheticalDocumentEmbedder intercepts your question, silently ships it 
# out to the LLM to generate the fake text using standard prompt structures (like web_search), 
# immediately converts that hidden result into numbers, and queries the database.

def hyde_rag_pipeline(query):
    # Pass query; HyDE silently maps query -> hypothetical text -> vector match
    # When similarity_search(query) is called, the query string is intercepted by the underlying 
    # HypotheticalDocumentEmbedder class. It silently sends that query to the LLM behind the scenes using 
    # the defined prompt (prompt_key="web_search" or your custom_prompt) to generate the hidden hypothetical text.
    matched_docs = vectorstore.similarity_search(query, k=2)
    
    print("\n📚 [Web Search Mode] Retrieved Context Chunks:")
    for i, doc in enumerate(matched_docs):
        print(f"--- Chunk {i+1} ---\n{doc.page_content}\n")
        
    response = rag_chain.invoke({
        "input": query,
        "context": matched_docs
    })
    return response

# Step 7: Run execution
query = "What OWASP API Top 10 represent in App Security and how are they applied?"
answer = hyde_rag_pipeline(query)
print("✅ Final Answer (Web Search):\n", answer)