"""
================================================================================
This script introduces **Corrective RAG (CRAG)**, an advanced self-correcting 
retrieval strategy implemented with LangGraph. Standard RAG blindy 
trusts whatever documents the vector store returns, which often leads to 
hallucinations if the source data is irrelevant. CRAG solves this by injecting 
an automated grading step that evaluates chunk quality, actively branching to 
the web if internal data falls short.


1. DATA INGESTION: Indexes a localized text vector store using high-performance 
   OpenAI embeddings and a FAISS database.
2. VECTOR RETRIEVAL (`retrieve`): Pulls candidate source documents related to 
   the user's incoming query.
3. QUALITY ASSESSMENT (`grade_documents`): Uses an LLM with structured output 
   (`Grade`) to strictly judge the retrieved items with a binary 'yes' or 'no' 
   relevance score. If any chunk is deemed irrelevant, a web search 
   flag is raised.
4. WEB SEARCH FALLBACK (`web_search`): If internal facts are absent or graded 
   as incomplete, the graph branches to a live Tavily API web search to scrape 
   ground-truth external context.
5. CONTEXT SYNTHESIS (`generate`): Assembles all validated context fragments 
   and generates a polished answer.
================================================================================
"""

import os
import logging
import warnings

from langchain_core._api.deprecation import LangChainPendingDeprecationWarning
# Suppress the specific LangChain serialization warning
warnings.filterwarnings("ignore", category=LangChainPendingDeprecationWarning)


from typing import List, TypedDict
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
# TODO from langchain_tavily import TavilySearch
#tavily_tool = TavilySearch(k=2)

from langgraph.graph import StateGraph, START, END

# # Corrective RAG (CRAG)
# 
# ### Concept
# Corrective RAG (CRAG) is a strategy that introduces a self-correction mechanism into the RAG pipeline. 
# It evaluates the relevance of retrieved documents to the query. 
# 
# 1. If the documents are relevant, it proceeds to generation.
# 2. If the documents are irrelevant or ambiguous, it triggers an external search (like Tavily) to supplement the knowledge base before generating an answer.

# Suppress noise: warnings, transformer loading logs, and progress bars
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# Load environment variables
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # Fix for certain environments when using FAISS and OpenAIEmbeddings together

# ### 1. Setup Data and Retriever
# We'll create a simple local vector store to act as our primary knowledge base.

docs = [
    Document(page_content="LangGraph is a library for building stateful, multi-actor applications with LLMs."),
    Document(page_content="Corrective RAG (CRAG) improves RAG by evaluating retrieved documents."),
    Document(page_content="Tavily is a search engine optimized for LLMs and AI agents.")
]

embedding=OpenAIEmbeddings(model="text-embedding-3-small", dimensions=512)
vectorstore = FAISS.from_documents(docs, embedding)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# ### 2. Define the State
class State(TypedDict):
    question: str
    documents: List[Document]
    candidate_answer: str
    run_web_search: bool

# ### 3. Define the Nodes

#NOTE - use gpt-4o-mini if below does not work 
llm = ChatOpenAI(model="gpt-4o", temperature=0)
search_tool = TavilySearchResults(k=2)

# #### Node 1: Retriever
def retrieve(state: State):
    print(" ---RETRIEVING---")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents}

# #### Node 2: Grade Documents (The "Correction" Logic)
class Grade(BaseModel):
    binary_score: str = Field(description="Relevance score 'yes' or 'no'")

def grade_documents(state: State):
    print(" ---CHECKING RELEVANCE---")
    structured_llm = llm.with_structured_output(Grade)
    
    prompt = ChatPromptTemplate.from_template(
        "Grade the relevance of the retrieved document to the user question. "
        "Score 'yes' if the document contains information related to the question, otherwise 'no'.\n"
        "Question: {question}\n"
        "Document: {document}"
    )
    
    chain = prompt | structured_llm
    
    relevant_docs = []
    run_web_search = False
    
    for d in state["documents"]:
        res = chain.invoke({"question": state["question"], "document": d.page_content})
        if res.binary_score == "yes":
            relevant_docs.append(d)
        else:
            run_web_search = True # If even one doc is irrelevant, we might want to search
            
    return {"documents": relevant_docs, "run_web_search": run_web_search}

# #### Node 3: Web Search
def web_search(state: State):
    print(" ---WEB SEARCHING---")
    question = state["question"]
    search_results = search_tool.invoke(question)
    
    joined_results = "\n".join([res["content"] for res in search_results])
    web_doc = Document(page_content=joined_results, metadata={"source": "tavily"})
    
    docs = state["documents"]
    docs.append(web_doc)
    return {"documents": docs}

# #### Node 4: Generate
def generate(state: State):
    print(" ---GENERATING---")
    prompt = ChatPromptTemplate.from_template(
        "Answer the question using the following context:\n{context}\nQuestion: {question}"
    )
    
    chain = prompt | llm
    context = "\n\n".join([d.page_content for d in state["documents"]])
    response = chain.invoke({"question": state["question"], "context": context})
    return {"candidate_answer": response.content}

# ### 4. Define Edge Logic
def decide_to_search(state: State):
    if state["run_web_search"]:
        return "web_search"
    else:
        return "generate"

# ### 5. Build the Graph
workflow = StateGraph(State)

workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("web_search", web_search)
workflow.add_node("generate", generate)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")

# Conditional path based on grading
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_search,
    {
        "web_search": "web_search",
        "generate": "generate"
    }
)

workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()

# ### 6. Run CRAG
# Case 1: Information is in the local vector store
print("---Case 1: Local Knowledge---")
res1 = app.invoke({"question": "What is LangGraph?"})
print(res1["candidate_answer"])

# Case 2: Information is NOT in the local vector store (Triggers Web Search)
print("\n---Case 2: External Knowledge---")
res2 = app.invoke({"question": "Who is the current CEO of Nvidia and what was their 2024 revenue?"})
print(res2["candidate_answer"])