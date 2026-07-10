"""
================================================================================
This script is a comprehensive architectural guide demonstrating three core 
flavors of Retrieval-Augmented Generation (RAG) using LangChain Expression Language 
(LCEL) and a cloud-hosted Llama-3.1 model via Groq. It takes static, in-memory documents 
and binds them to interactive, intelligent system runtime components.

1. STANDARD/SYNCHRONOUS RAG: Collects context, queries the LLM, and suppresses 
   terminal streaming output to return a single, cleanly parsed string at the 
   very end of generation.
2. STREAMING RAG: Leverages the generator patterns of LLMs to display response tokens 
   piece-by-piece in real-time for highly responsive terminal or web user interfaces.
3. CONVERSATIONAL (CHAT HISTORY) RAG: Implements dialogue state persistence. It uses 
   historical message arrays to ground follow-up questions contextually, enabling 
   accurate multi-turn interaction.
================================================================================
"""

import os
import warnings
from typing import List
from dotenv import load_dotenv

# LangChain Core and Models
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# Components
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

warnings.filterwarnings('ignore')
load_dotenv()

# ==============================================================================
# 1. INITIAL PREPARATION: Models, Documents, and Vector Store
# ==============================================================================

# Initialize LLM (Groq)
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
llm = init_chat_model(
    model="groq:llama-3.1-8b-instant",
    temperature=0 
)

# Initialize Embeddings
# You can use the dimensions parameter to compress text-embedding-3-small down to a smaller size, like 512 dimensions:
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=512
)

#TODO load from text file 
# Sample Data
sample_documents = [
    Document(
        page_content="Artificial Intelligence (AI) is the simulation of human intelligence in machines...",
        metadata={"source": "AI Introduction", "topic": "AI"}
    ),
    Document(
        page_content="Machine Learning is a subset of AI that enables systems to learn from data...",
        metadata={"source": "ML Basics", "topic": "ML"}
    ),
    Document(
        page_content="Deep Learning is a subset of machine learning based on artificial neural networks...",
        metadata={"source": "Deep Learning", "topic": "DL"}
    ),
    Document(
        page_content="Natural Language Processing (NLP) is a branch of AI that helps computers understand human language...",
        metadata={"source": "NLP Overview", "topic": "NLP"}
    )
]

# Text Splitting
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(sample_documents)

# Vector Store and Retriever
vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Helper function to format context
def format_docs(docs: List[Document]) -> str:
    formatted = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get('source', 'Unknown')
        formatted.append(f"Document {i+1} (Source: {source}):\n{doc.page_content}")
    return "\n\n".join(formatted)

# Base Prompt for Simple/Streaming
simple_prompt = ChatPromptTemplate.from_template("""
Use the following context to answer the question. 
If you don't know the answer based on the context, say you don't know.
Provide specific details from the context to support your answer.

Context: {context}

Question: {question}

Answer:""")


# ==============================================================================
# 2. IMPLEMENTATION 1: Simple RAG Chain (Standard/Synchronous RAG)
#    -  This chain processes the query silently behind the scenes and waits until the entire text answer is fully generated before printing the final result all at once.
#    - KEY VARIABLE: The `StrOutputParser()` at the end is crucial because it automatically unpacks the final complex AI message object and returns a clean, plain Python string.

# ==============================================================================

simple_rag_chain_lcel = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | simple_prompt
    | llm
    | StrOutputParser()
)

# HACK: just for testing purposes
# response=simple_rag_chain_lcel.invoke("What is Deep Learning")
# response
# retriever.invoke("What is Deep Learning")

# ==============================================================================
# 3. IMPLEMENTATION 2: Streaming RAG Chain (Real-Time/Streaming RAG)
#    - This chain emits words piece-by-piece in real-time as they are being thought up by the LLM, creating a highly responsive user experience.
#    - KEY VARIABLE: By removing `StrOutputParser()`, the chain outputs a stream of raw `AIMessageChunk`
#      objects whose `.content` properties can be incrementally extracted and flushed to the terminal using a `stream()` loop.
# ==============================================================================
streaming_rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | simple_prompt
    | llm
)

# ==============================================================================
# 4. IMPLEMENTATION 3: Conversational RAG Chain (Chat History-Aware RAG)
#    - This chain acts like an interactive chatbot by remembering previous questions and answers, allowing users to ask context-dependent follow-up questions.
#    - KEY VARIABLES: It uses `{chat_history}` inside a `from_messages` list to physically inject past dialogue, 
#       and `RunnablePassthrough.assign()` to map incoming dictionary keys like `{input}` dynamically through your custom retriever function.
# ==============================================================================

conversational_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Use the provided context to answer questions."),
    ("placeholder", "{chat_history}"),
    ("human", "Context: {context}\n\nQuestion: {input}"),
])

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

# Query using the LCEL approach - Fixed version
def test_standard_chains_lcel(question):
    print(f"Question: {question}")
    print("-" * 50)
    
    # Method 1: Pass string directly (when using RunnablePassthrough)
    answer = simple_rag_chain_lcel.invoke(question)
    print(f"Answer: {answer}")
    
    # Get source documents separately if needed
    docs = retriever.invoke(question)
    print("\nSource Documents:")
    for i, doc in enumerate(docs):
        print(f"\n--- Source {i+1} ---")
        print(doc.page_content[:200] + "...")
       

def test_standard_chains_stream(question: str):
    print(f"\n--- Testing Simple and Streaming Chains ---")
    print(f"Question: {question}\n")
    
    # Simple Invoke
    print("1. Simple RAG Answer:")
    print(streaming_rag_chain.invoke(question))

    # Stream
    print("\n2. Streaming RAG Answer:")
    for chunk in streaming_rag_chain.stream(question):
        print(chunk.content, end="", flush=True)
    print("\n")


def test_conversational_chain(question: str, followup: str):
    print(f"--- Testing Conversational RAG ---")
    history = []
    
    # Round 1
    a1 = conversational_rag.invoke({"input": question, "chat_history": history})
    print(f"Q1: {question}\nA1: {a1}\n")
    
    history.extend([HumanMessage(content=question), AIMessage(content=a1)])
    
    # Round 2 (Context-dependent)
    a2 = conversational_rag.invoke({"input": followup, "chat_history": history})
    print(f"Q2: {followup}\nA2: {a2}")


if __name__ == "__main__":
    question1 = "What is the difference between AI and machine learning?"
    test_standard_chains_lcel(question1)
    print("="*50)
    test_standard_chains_stream(question1)
    print("="*50)

    question2 = "What is machine learning?"
    question2_followup = "how it is difference from AI?"