"""
================================================================================
# ChatBot RAG with Memory
This script showcases **RAG with Memory**, a stateful system architecture that 
enables a retrieval-augmented generation app to manage multi-turn conversations. 
Traditional RAG parses queries isolated from previous turns, preventing users 
from submitting dependent follow-up questions. This pipeline bridges that gap 
by integrating state reducer lists with a persistent checkpoint disk.

Standard RAG processes each question independently. RAG with Memory allows the agent 
to maintain conversation history. This enables the user to ask follow-up questions 
like "Can you explain the second point in more detail?"

1. STATE SCHEMATICS: Configures a `State` dict equipped with an append-only message 
   reducer function (`add_messages`), ensuring conversational history accumulates 
   cleanly across multiple invocations.
2. CONTEXTUAL RETRIEVAL (`retrieve`): Targets the latest user message from the 
   end of the history list and runs a local FAISS semantic lookup.
3. DIALOGUE INJECTION (`generate`): Injects the extracted context *and* the full 
   message history array into a specialized system prompt template using a 
   `MessagesPlaceholder` construct.
4. STATE PERSISTENCE (`MemorySaver`): Compiles the workflow alongside an in-memory 
   checkpointer, enabling thread-isolated conversational memory tracks.
================================================================================
"""

import os
import warnings
from langchain_core._api.deprecation import LangChainPendingDeprecationWarning
# Suppress the specific LangChain serialization warning
warnings.filterwarnings("ignore", category=LangChainPendingDeprecationWarning)

from typing import Annotated, List, TypedDict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ### 1. Setup Vector Store
docs = [
    Document(page_content="LangGraph is a library for building stateful, multi-actor applications with LLMs."),
    Document(page_content="Memory in LangGraph is handled via checkpointers like MemorySaver."),
    Document(page_content="RAG allows LLMs to retrieve external data to answer questions accurately.")
]

embedding=OpenAIEmbeddings(model="text-embedding-3-small", dimensions=512)
vectorstore = FAISS.from_documents(docs, embedding)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# ### 2. Define the State
# We use add_messages to ensure the history is appended, not overwritten.
# When a node returns a new message, do NOT overwrite the existing list of messages; instead, append (add) the new message to the end of the list."
class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    context: List[Document]
    answer: str

# ### 3. Define the Nodes
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# #### Node: Retrieve
def retrieve(state: State):
    # Get the latest user message
    last_message = state["messages"][-1].content
    documents = retriever.invoke(last_message)
    return {"context": documents}

# #### Node: Generate
# NOTE we did not use this in 18a_Chatbot.py 
def generate(state: State):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's question based on the context provided: {context}"),
        MessagesPlaceholder(variable_name="messages"),
    ])
    
    chain = prompt | llm
    context_text = "\n\n".join([d.page_content for d in state["context"]])
    
    response = chain.invoke({
        "messages": state["messages"],
        "context": context_text
    })
    
    return {"messages": [response], "answer": response.content}

# ### 4. Build the Graph
workflow = StateGraph(State)

workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

# IMPORTANT: Add Memory Checkpointer
# NOTE we did not use this in 18a_Chatbot.py 
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# ### 5. Run with Memory (Threads)
config = {"configurable": {"thread_id": "user_123"}}

print("--- Question 1 ---")
req1 = {"messages": [HumanMessage(content="What is LangGraph?")]}
# for chunk in app.stream(req1, config=config):
#     print(chunk)
answer1 = app.invoke(req1, config=config)
print(f"Bot Response 1: {answer1['messages'][-1].content}")    

print("\n--- Question 2 (Follow-up) ---")
req2 = {"messages": [HumanMessage(content="How does it handle memory?")]}
# for chunk in app.stream(req2, config=config):
#     print(chunk)
answer2 = app.invoke(req2, config=config)
print(f"Bot Response 1: {answer2['messages'][-1].content}")        