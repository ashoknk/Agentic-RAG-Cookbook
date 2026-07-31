"""

1. EVOLUTION FROM RETRY LOOPS TO ACTIVE WEB SEARCH FALLBACK (CRAG)
   - 25c_AgenticRAG_Grader.py: Used a closed-loop Corrective RAG design. When 
     retrieved documents were irrelevant, it routed to a `rewrite` node to re-query 
     the local FAISS vector store, capping at max retries before routing to a static 
     `fallback` error message.
   - 25d_AgenticRAG_Grader.py: Implements active Multi-Source Fallback (True CRAG). 
     When local FAISS retrieved documents fail the relevance check,  and directly routes to an external web search tool to fetch real-time data.

     CRAG stands for Corrective Retrieval-Augmented Generation. 
     It is an advanced RAG design pattern designed to prevent "garbage-in, garbage-out"

2. EXTRA AND REMOVED CODE & COMPONENTS
   -----------------------------------------------------------------------------
   A. ADDED COMPONENTS / CODE:
      * Tool: `web_search_tool = TavilySearchResults(k=3)`.
      * Node Function: `web_search(state: AgentState)`
        - Executes live web queries using Tavily when local database documents are graded as irrelevant (`"no"`).
      * Graph Routing:
        - Added edge: `workflow.add_edge("web_search", "generate")` to pass web search 
          context straight into answer generation.

   B. REMOVED COMPONENTS / CODE:
      * Node Function: `rewrite(state: AgentState)` was completely removed.
      * Node Function: `fallback(state: AgentState)` was replaced by the dynamic `web_search` node.

   C. EXTRA / MODIFIED FUNCTIONS:
      * Function: `grade_documents(state: AgentState)`
        - Return Type signature updated: `Literal["generate", "web_search"]` 
          (previously `Literal["generate", "rewrite", "fallback"]`).
        - Routing Logic: Directly returns `"web_search"` upon encountering an irrelevant 
          document score instead of evaluating `retry_count`.

3. KEY STRUCTURAL ADVANTAGES
   - Broader Knowledge Scope: Prevents total failure on out-of-domain queries by 
     supplementing internal knowledge bases with real-time web capabilities.
   - Reduced Latency & API Calls: Removes iterative query rewriting loops over static 
     data sources that are missing the required context.

    
    Demonstrating Corrective RAG (CRAG) with Multi-Source Fallback.
    Pipeline:
    1. Agent retrieves from FAISS vector DB.
    2. Document Grader checks relevance.
    3. If relevant -> Generate answer.
    4. If irrelevant -> Rewrite query and search the Live Web (Tavily).
"""

import os
from typing import Annotated, Literal, Sequence
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()
os.environ["USER_AGENT"] = "Agentic-RAG-Cookbook/2.0"
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

from langchain.chat_models import init_chat_model
from langchain_core.tools import create_retriever_tool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# ==============================================================================
# 1. MULTI-SOURCE DATA PREPROCESSING
# ==============================================================================

embeddings_engine = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=512)
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

waf_urls = [
    "https://www.fastly.com/blog/ua-spoofing-101-detection-defense-with-fastlys-next-gen-waf",
    "https://www.fastly.com/blog/credential-stuffing-attacks-vs-brute-force-attacks-what-is-the-difference",
    "https://www.fastly.com/blog/what-is-cve-2026-23869-react-server-components-security-alert",
    "https://www.fastly.com/blog/ddos-in-december-2025"
]

waf_docs = [item for url in waf_urls for item in WebBaseLoader(url).load()]
vectorstore_waf = FAISS.from_documents(documents=splitter.split_documents(waf_docs), embedding=embeddings_engine)

retriever_tool_waf = create_retriever_tool(
    vectorstore_waf.as_retriever(),
    "retriever_waf_db",
    "Search internal database for Fastly WAF and web security blogs"
)

# Active local tools
tools = [retriever_tool_waf]

# Fallback web tool -->
web_search_tool = TavilySearchResults(k=3)

# ==============================================================================
# 2. STATE & SCHEMAS
# ==============================================================================

shared_llm = init_chat_model("groq:qwen/qwen3.6-27b")

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    # retry_count: int


class DocumentGrade(BaseModel):
    binary_score: str = Field(description="Relevance score 'yes' or 'no'")

# ==============================================================================
# 3. NODES & ROUTERS
# ==============================================================================

def agent(state: AgentState):
    print("---CALL AGENT---")
    messages = state["messages"]
    model_with_tools = shared_llm.bind_tools(tools)
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}

def grade_documents(state: AgentState) -> Literal["generate", "web_search"]:
    print("---CHECK RELEVANCE---")
    # retries = state.get("retry_count", 0)
    
    llm_with_tool = shared_llm.with_structured_output(DocumentGrade)
    prompt = ChatPromptTemplate.from_template(
        """You are a grader assessing relevance of a retrieved document to a user question.
        Document: {context}
        Question: {question}
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant."""
    )

    chain = prompt | llm_with_tool
    messages = state["messages"]
    question = messages[0].content
    docs_content = messages[-1].content
    
    scored_result = chain.invoke({"question": question, "context": docs_content})
    
    #Add web_search in else -->
    if scored_result.binary_score == "yes":
        print("---DECISION: LOCAL DOCS RELEVANT---")
        return "generate"
    else:
        print("---DECISION: LOCAL DOCS IRRELEVANT -> TRIGGERING WEB SEARCH---")
        return "web_search"

#Add web_search -->
def web_search(state: AgentState):
    """Fallback node that executes a live web search if local database retrieval fails."""
    print("---WEB SEARCH FALLBACK---")
    messages = state["messages"]
    question = messages[0].content
    
    # Run live web query
    search_results = web_search_tool.invoke({"query": question})
    web_context = "\n\n".join([res["content"] for res in search_results])
    
    return {"messages": [HumanMessage(content=f"Web Search Results:\n{web_context}")]}


def generate(state: AgentState):
    print("---GENERATE---")
    messages = state["messages"]
    question = messages[0].content
    docs_content = messages[-1].content

    prompt_template = ChatPromptTemplate.from_template(
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question concisely.\n\n"
        "Question: {question}\n"
        "Context: {context}\n"
        "Answer:"
    )
    rag_chain = prompt_template | shared_llm | StrOutputParser()
    response = rag_chain.invoke({"context": docs_content, "question": question})
    return {"messages": [response]}

# ==============================================================================
# 4. GRAPH COMPILATION
# ==============================================================================

workflow = StateGraph(AgentState)

workflow.add_node("agent", agent)
workflow.add_node("retrieve", ToolNode(tools))
workflow.add_node("web_search", web_search)
workflow.add_node("generate", generate)

workflow.add_edge(START, "agent")

workflow.add_conditional_edges(
    "agent",
    tools_condition,
    {"tools": "retrieve", END: END}
)

workflow.add_conditional_edges(
    "retrieve",
    grade_documents,
    {
        "generate": "generate",
        "web_search": "web_search"
    }
)

workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)

graph = workflow.compile()

# Save the file as a PNG
OUTPUT_IMAGE_PATH = "Image_PNGs/25d_AgenticRAG_Grader.png"
graph.get_graph().draw_mermaid_png(output_file_path=OUTPUT_IMAGE_PATH)    
os.system(f"open {OUTPUT_IMAGE_PATH}")


# ==============================================================================
# 5. TEST RUNS
# ==============================================================================

print("\n🛡️ In-Domain Query (Uses Local FAISS DB):")
graph.invoke({"messages": [HumanMessage(content="How does Fastly Next-Gen WAF handle credential stuffing attacks?")]})

print("\n🌐 Out-of-Domain Query (Fails FAISS -> Falls back to Live Web Search):")
graph.invoke({"messages": [HumanMessage(content="What are the main security updates in Python 3.12?")]})