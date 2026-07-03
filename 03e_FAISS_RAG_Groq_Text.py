import os
import warnings
from typing import List
from dotenv import load_dotenv

# LangChain Core and Models
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# Components
from langchain_community.vectorstores import FAISS

warnings.filterwarnings('ignore')
load_dotenv()

# ==============================================================================
# 1. INITIAL PREPARATION: Environment & Directory Config
# ==============================================================================

# Fix for potential macOS OpenMP duplication crash
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Target directory where the pre-built FAISS index files reside
FAISS_PERSIST_DIRECTORY = "./faiss_textfile" 

# Initialize LLM (Groq)
llm = init_chat_model(
    model="groq:llama-3.1-8b-instant",
    temperature=0 
)

# Initialize Embeddings (Must match the dimensions used during store creation)
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=1536
)

# ==============================================================================
# 2. LOAD FAISS VECTOR STORE FROM DISK
# ==============================================================================
if not os.path.exists(FAISS_PERSIST_DIRECTORY):
    raise FileNotFoundError(
        f"The local index path '{FAISS_PERSIST_DIRECTORY}' was not found. "
        f"Please run your FAISS build script first!"
    )

print(f"Loading FAISS index from '{FAISS_PERSIST_DIRECTORY}'...")
loaded_vectorstore = FAISS.load_local(
    FAISS_PERSIST_DIRECTORY,
    embeddings,
    allow_dangerous_deserialization=True  # Required to safely unpack pickle files locally
)
print(f"Loaded FAISS vector store contains {loaded_vectorstore.index.ntotal} vectors.\n")
# print("-" * 60)

# Convert vector store to retriever
retriever = loaded_vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Helper function to format context
def format_docs(docs: List[Document]) -> str:
    formatted = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get('source', 'Unknown')
        formatted.append(f"Document {i+1} (Source: {source}):\n{doc.page_content}")
    return "\n\n".join(formatted)

# ==============================================================================
# 3. PROMPT TEMPLATES
# ==============================================================================
simple_prompt = ChatPromptTemplate.from_template("""
Use the following context to answer the question. 
If you don't know the answer based on the context, say you don't know.
Provide specific details from the context to support your answer.

Context: {context}

Question: {question}

Answer:""")

conversational_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Use the provided context to answer questions."),
    ("placeholder", "{chat_history}"),
    ("human", "Context: {context}\n\nQuestion: {input}"),
])

# ==============================================================================
# 4. IMPLEMENTATION: RAG Chains (LCEL)
# ==============================================================================

# Standard RAG Chain returning clean text string output
simple_rag_chain_lcel = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | simple_prompt
    | llm
    | StrOutputParser()
)

# Streaming RAG Chain returning raw message chunks
streaming_rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | simple_prompt
    | llm
)

# Conversational RAG Chain tracking manual chat history input arrays
conversational_rag = (
    RunnablePassthrough.assign(
        context=lambda x: format_docs(retriever.invoke(x["input"]))
    )
    | conversational_prompt
    | llm
    | StrOutputParser()
)

# ==============================================================================
# 5. EXECUTION AND TESTING
# ==============================================================================
def test_standard_chains_lcel(question: str):
    print(f"--- Testing Standard RAG (LCEL) ---")
    print(f"Question: {question}")
    # print("-" * 50)
    
    answer = simple_rag_chain_lcel.invoke(question)
    print(f"Answer: {answer}")
    
    docs = retriever.invoke(question)
    print("\nSource Documents:")
    for i, doc in enumerate(docs):
        print(f"\n     --- Source {i+1} ---")
        print(doc.page_content[:200] + "...")

def test_standard_chains_stream(question: str):
    print(f"\n--- Testing Simple and Streaming Chains ---")
    print(f"\tQuestion: {question}\n")
    
    print("\t 1. Simple RAG Answer:")
    print(streaming_rag_chain.invoke(question).content)  # .content extracts string from AIMessage

    print("\n\t 2. Streaming RAG Answer:")
    for chunk in streaming_rag_chain.stream(question):
        print(chunk.content, end="", flush=True)
    print("\n")

def test_conversational_chain(question: str, followup: str):
    print(f"--- Testing Conversational RAG ---")
    history = []
    
    # Round 1
    a1 = conversational_rag.invoke({"input": question, "chat_history": history})
    print(f"\tQ1: {question}\n\tA1: {a1}\n")
    
    history.extend([HumanMessage(content=question), AIMessage(content=a1)])
    
    # Round 2 (Context-dependent)
    a2 = conversational_rag.invoke({"input": followup, "chat_history": history})
    print(f"\tQ2: {followup}\n\tA2: {a2}")

if __name__ == "__main__":
    test_query = "What is the difference between AI and machine learning?"
    test_standard_chains_lcel(test_query)
    print("="*50)

    test_standard_chains_stream(test_query)
    print("="*50)
    
    conv_query = "What is machine learning?"
    conv_followup = "What are its main types?" # Corrected follow-up query to show off history context resolution
    test_conversational_chain(conv_query, conv_followup)