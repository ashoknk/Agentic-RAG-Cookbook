"""
================================================================================
CONVERSATIONAL RAG WITH CHAT HISTORY & ANAPHORA RESOLUTION
================================================================================
Purpose:
    This script implements a Retrieval-Augmented Generation (RAG) pipeline 
    capable of maintaining conversational context. It resolves context-dependent 
    pronouns (e.g., "it", "that", "its") by analyzing the user's latest query 
    against the existing chat history and reformulating it into a standalone 
    search query before querying the vector database.

Key LangChain Components & Functions Used:
--------------------------------------------------------------------------------
    * MessagesPlaceholder ("chat_history"): 
        A dynamic prompt component that injects a variable list of past chat 
        messages (Human and AI) into the prompt template at runtime.

    * create_history_aware_retriever(llm, retriever, prompt):
        A specialized LangChain chain that takes the chat history and the 
        latest user question, passes them to an LLM to generate a standalone, 
        context-independent query, and uses that new query to fetch the most 
        relevant documents from the vector store.

================================================================================
"""

import os
import warnings
from dotenv import load_dotenv

# Core LangChain Imports
from langchain.chat_models.base import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage

# Modified Chain Imports using langchain_classic
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

warnings.filterwarnings('ignore')
load_dotenv()

DATA_DIR = 'data'
PERSIST_DIRECTORY = './chroma_db'

# ==========================================
# 1. INITIALIZE COMPONENTS
# ==========================================
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize LLM
llm = init_chat_model("openai:gpt-4o-mini")

# Load existing Chroma DB vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embeddings,
    collection_name="rag_collection"
)

# Convert vector store to retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ==========================================
# 2. CONTEXTUALIZE QUESTION (History-Aware)
# ==========================================
#TODO make this prompt better or different
# contextualize_q_system_prompt = """Given a chat history and the latest user question 
# which might reference context in the chat history, formulate a standalone question 
# which can be understood without the chat history. Do NOT answer the question, 
# just reformulate it if needed and otherwise return it as is."""

contextualize_q_system_prompt = """You are an expert query-reformulation assistant. 
Your task is to analyze a conversation history and a new user question. 

If the latest question references things from the past history (such as using pronouns like 'it', 'that', 'they', 'its', or 'this'), reformulate it into a completely standalone, context-independent question.

CRITICAL RULES:
1. Do NOT answer the question under any circumstances.
2. Only output the reformulated question (or the original question if no changes are needed).
3. Do not include any introductory phrases, explanations, or meta-commentary."""

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Create history aware retriever using langchain_classic
# retriever is used to create the history-aware retriever
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# ==========================================
# 3. QA & DOCUMENT CHAIN
# ==========================================
qa_system_prompt = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.

Context: {context}"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# The create_history_aware_retriever function inherently expects a structured chat prompt 
# that can seamlessly insert an ongoing conversation stream.
#TODO: explain the purpose of create_stuff_documents_chain
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Create conversational RAG chain using langchain_classic
conversational_rag_chain = create_retrieval_chain(
    history_aware_retriever, 
    question_answer_chain
)
print("Conversational RAG chain created using langchain_classic!\n")

# ==========================================
# 4. EXECUTION & CONVERSATION
# ==========================================
chat_history = []

# ===Example 1 ====
# First question
question1 = "What is machine learning?"
result1 = conversational_rag_chain.invoke({
    "chat_history": chat_history,
    "input": question1
})
print(f"Q: {question1}")
print(f"A: {result1['answer']}\n")

# Update history
chat_history.extend([
    HumanMessage(content=question1),
    AIMessage(content=result1['answer'])
])

# Follow up question
followup_question1 = "What are its main types?"
result1_followup = conversational_rag_chain.invoke({
    "chat_history": chat_history,
    "input": followup_question1  # Refers back to ML
})

print(f"Q: {followup_question1}")
print(f"A: {result1_followup['answer']}")

# ===Example 2 ====
print(f"\n=========== Example 2 ===========")
# Vector Embeddings and Vector Databases
# First question
question2 = "What is Vector Embedding?"
result2 = conversational_rag_chain.invoke({
    "chat_history": chat_history,
    "input": question2
})
print(f"Q: {question2}")
print(f"A: {result2['answer']}\n")

# Update history
chat_history.extend([
    HumanMessage(content=question2),
    AIMessage(content=result2['answer'])
])

# Follow up question
followup_question2 = "What are its main types?"
result2_followup = conversational_rag_chain.invoke({
    "chat_history": chat_history,
    "input": followup_question2  # Refers back to ML
})

print(f"Q: {followup_question2}")
print(f"A: {result2_followup['answer']}")

