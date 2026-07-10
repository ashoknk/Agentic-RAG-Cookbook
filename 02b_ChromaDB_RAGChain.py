"""
==============================================================================
ARCHITECTURAL COMPARISON: 02a_ChromaDB_Sample vs. 02b_ChromaDB_RAGChainSample
------------------------------------------------------------------------------
While both files connect to ChromaDB, file 02b introduces a Large Language 
Model (LLM) and LangChain's retrieval chains. This shifts the codebase from 
a simple document search engine to an intelligent question-answering system.

KEY IMPROVEMENTS OF 02b OVER 02a:

1. RAW RETRIEVAL VS. NATURAL RESPONSES
   - 02a (Vector Search Only): Only locates and prints raw, jagged text chunks 
     exactly as they are written in your files. If the answer is split or messy, 
     the user has to read through the raw data manually to find the answer.
   - 02b (Integrated LLM): Takes those raw text chunks and feeds them to 
     "gpt-4o-mini". The LLM synthesizes, filters, and rewrites the data into 
     a highly concise, accurate, and human-readable response.

2. CONTEXT STUFFING (create_stuff_documents_chain)
   - 02b introduces a unified pipeline that automatically "stuffs" all matching 
     database text chunks directly into a single system prompt template context 
     window before communicating with OpenAI.

3. END-TO-END AUTOMATION (create_retrieval_chain)
   - In 02a, you have to query the database, capture the documents, and handle 
     the output manually. 
   - In 02b, calling `rag_chain.invoke()` handles the entire lifecycle in a single 
     step: it converts the query to a vector, searches ChromaDB, passes the hits 
     to the prompt template, queries the LLM, and returns a clean dictionary 
     containing both the final humanized answer and the source document history.

SUMMARY:
File 02a is the "Search Engine" (Finds the data). 
File 02b is the "Smart Assistant" (Finds the data AND explains it to the user).
==============================================================================
"""

import os
import warnings
from dotenv import load_dotenv


from langchain.chat_models.base import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_classic.chains import create_retrieval_chain
# from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

warnings.filterwarnings('ignore')
load_dotenv()

# ==============================================================================
# 1. INITIAL PREPARATION: Environment & Sample Data
# ==============================================================================

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
# NOTE https://reference.langchain.com/python/langchain/chat_models/base/init_chat_model
# gpt-4o-mini is faster, significantly more capable, and cheaper than gpt-3.5-turbo
llm=init_chat_model("openai:gpt-4o-mini")

DATA_DIR = "data"
# 1. Define the same directory and collection name used during creation
PERSISTENT_DIRECTORY = "./chroma_db"

# collection_name acts as a unique namespace or table name that groups and stores your specific embeddings
COLLECTION_NAME = "rag_collection"

# ==============================================================================
# 3. VECTOR STORE loading (ChromaDB)
# ==============================================================================

# Initialize embeddings and the persistent Chroma store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 3. Load the existing vectorstore from disk
# Note: We do NOT use .from_documents() here 
#  a constructor method used for loading and querying existing data.
vectorstore = Chroma(
    persist_directory=PERSISTENT_DIRECTORY,
    embedding_function=embeddings,
    collection_name=COLLECTION_NAME
)

# 4. Verify it loaded correctly by checking the count
print(f"Loaded vectorstore with {vectorstore._collection.count()} vectors.")
print(f"Data persisted to: {PERSISTENT_DIRECTORY}")

## Convert vector store to retriever
retriever=vectorstore.as_retriever(
    search_kwarg={"k":3} ## Retrieve top 3 relevant chunks
)

## Create a prompt template
# ChatPromptTemplate.from_messages
#    - EXPECTS: A Python list of explicit role-content tuples or special message 
#      objects (e.g., [("system", "..."), ("human", "...")]).
#    - BEHAVIOR: It explicitly segregates text based on actor identity. It forces 
#      the model to strictly distinguish between System instructions (guardrails), 
#      AI tracking, and User inputs.
#    - BEST FOR: Production chat interfaces, multi-turn conversations, handling 
#      chat history placeholders, or when strict system behavior enforcement is 
#      required from the LLM


system_prompt="""You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.

Context: {context}"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

### Create a document chain
# create_stuff_documents_chain creates a chain that "stuffs" (inserts) all retrieved documents 
# into a single prompt and sends it to the LLM. 
# It's called "stuff" because it literally stuffs all the documents into the context window at once.
document_chain=create_stuff_documents_chain(llm,prompt)

# This chain:
# - Takes retrieved documents
# - "Stuffs" them into the prompt's {context} placeholder
# - Sends the complete prompt to the LLM
# - Returns the LLM's response

### Create The Final RAG Chain
# create_retrieval_chain is a function that combines a retriever (which fetches relevant documents) 
# with a document chain (which processes those documents with an LLM) to create a complete RAG pipeline.
rag_chain=create_retrieval_chain(retriever,document_chain)

# NOTE: Just for testing
# response=rag_chain.invoke({"input":"What is Deep Learning"})
# print(f"\nQuestion: What is Deep Learning")
# print(f"Answer: {response['answer']}")

# Function to query the modern RAG system
def query_rag_modern(question):
    print(f"\nQuestion: {question}")
    print("-" * 50)
    
    # Using create_retrieval_chain approach
    result = rag_chain.invoke({"input": question})
    
    print(f"Answer: {result['answer']}")
    print("\nRetrieved Context:")
    for i, doc in enumerate(result['context']):
        print(f"\n--- Source {i+1} ---")
        print(doc.page_content[:200] + "...")
    
    return result

# Test queries
test_questions = [
    "What are the three types of machine learning?",
    "What is deep learning and how does it relate to neural networks?",
    "What are CNNs best used for?"
]

for question in test_questions:
    result = query_rag_modern(question)
    print("\n" + "="*60 + "\n")