"""
================================================================================
This script presents **Adaptive RAG**, a sophisticated, routing-driven strategy 
designed to dynamically adjust its retrieval method based on the core nature 
and intent of the user prompt. Rather than running a static retrieval sequence, 
it uses an upfront routing gatekeeper node to direct requests along entirely different 
computational paths.


THE THREE STRUCTURAL PATHWAYS:
------------------------------
- **Vector Store (RAG)**: For hyper-focused, specific domain knowledge found 
  in local document databases (e.g., proprietary software docs).
- **Web Search**: For trending queries, current real-time alerts, or wide-spectrum 
  general knowledge outside internal data pools.
- **Direct Response**: (Conceptual) For simple conversational pleasantries, greetings, 
  or baseline logic tasks.

1. GATEKEEPER ROUTER (`router_node`): Intercepts the user query and forces an 
   LLM to commit to a structured metadata choice (`RouteQuery`).
2. DYNAMIC BRANCHING: LangGraph interprets the returned token path (`vectorstore` 
   or `web_search`) and routes the state packet accordingly.
3. SOURCE SELECTION: Either triggers a localized FAISS vector store retrieval 
   or launches a live Tavily web search, preventing wasteful database overhead.
4. KNOWLEDGE SYNTHESIS (`generate`): Feeds the accurately routed context into 
   the final generative generation chain.
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

# from langchain_tavily import TavilySearchResults
from langgraph.graph import StateGraph, START, END

# # Adaptive RAG
# 
# ### Concept
# Adaptive RAG is a sophisticated strategy that routes a query to the most appropriate retrieval method based on its nature.
# 
# It uses a "Router" node at the beginning of the graph to decide:
# 1.  **Web Search**: For trending or current topics not in the internal database.
# 2.  **Vector Store (RAG)**: For specific internal knowledge or documentation.
# 3.  **Direct Response**: For simple greetings or common knowledge.

# Suppress noise: warnings, transformer loading logs, and progress bars
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# Load environment variables
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ### 1. Setup Vector Store
# Creating a small local knowledge base for LangGraph specific info.

docs = [
    Document(page_content="LangGraph is a library for building stateful, multi-actor applications with LLMs."),
    Document(page_content="Adaptive RAG routes queries to web search or vector stores based on intent."),
    Document(page_content="Nodes in LangGraph represent functions, and edges represent the flow of state.")
]

embedding=OpenAIEmbeddings(model="text-embedding-3-small", dimensions=512)
vectorstore = FAISS.from_documents(docs, embedding)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# ### 2. Define the State
class State(TypedDict):
    question: str
    documents: List[Document]
    answer: str
    route: str # Tracks the decision made by the router

# ### 3. Define the Router (The Key Component)
# We use structured output to force the LLM to choose a path.

class RouteQuery(BaseModel):
    """Route a user query to the most appropriate datasource."""
    datasource: str = Field(
        description="Given a user question choose to route it to web_search or vectorstore.",
    )

def router_node(state: State):
    print(" ---ROUTING---")
    #NOTE - use gpt-4o-mini if below does not work 
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    structured_llm = llm.with_structured_output(RouteQuery)
    
    prompt = ChatPromptTemplate.from_template(
        "You are an expert at routing a user question to a vectorstore or web search. "
        "The vectorstore contains documents related to LangGraph. "
        "Use web_search for questions about recent events or general knowledge. "
        "Question: {question}"
    )
    
    router_chain = prompt | structured_llm
    result = router_chain.invoke({"question": state["question"]})
    return {"route": result.datasource}

# ### 4. Define Functional Nodes
#NOTE - use gpt-4o-mini if below does not work 
llm = ChatOpenAI(model="gpt-4o", temperature=0)
search_tool = TavilySearchResults(k=2)

# #### Node: Retrieve from Vector Store
def retrieve(state: State):
    print(" ---RETRIEVING FROM VECTOR STORE---")
    documents = retriever.invoke(state["question"])
    return {"documents": documents}

# #### Node: Web Search
def web_search(state: State):
    print(" ---SEARCHING THE WEB---")
    search_results = search_tool.invoke(state["question"])
    joined_results = "\n".join([res["content"] for res in search_results])
    web_doc = Document(page_content=joined_results)
    return {"documents": [web_doc]}

# #### Node: Generate Answer
def generate(state: State):
    print(" ---GENERATING---")
    prompt = ChatPromptTemplate.from_template(
        "Answer the question using ONLY the provided context:\n{context}\nQuestion: {question}"
    )
    context = "\n\n".join([d.page_content for d in state["documents"]])
    chain = prompt | llm
    response = chain.invoke({"question": state["question"], "context": context})
    return {"answer": response.content}

# ### 5. Build the Graph
workflow = StateGraph(State)

# Add Nodes
workflow.add_node("router_node", router_node)
workflow.add_node("retrieve", retrieve)
workflow.add_node("web_search", web_search)
workflow.add_node("generate", generate)

# Define Entry Point and Conditional Edges
workflow.add_edge(START, "router_node")

workflow.add_conditional_edges(
    "router_node",
    lambda x: x["route"],
    {
        "vectorstore": "retrieve",
        "web_search": "web_search"
    }
)

workflow.add_edge("retrieve", "generate")
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()

# ### 6. Run Adaptive RAG

# Case A: Query routes to Vector Store
print("---Case A: Internal Query---")
res1 = app.invoke({"question": "Explain what a node is in LangGraph."})
print(f"Final Answer: {res1['answer']}")

# Case B: Query routes to Web Search
print("\n---Case B: Trending/General Query---")
res2 = app.invoke({"question": "What is the weather in New York today?"})
print(f"Final Answer: {res2['answer']}")