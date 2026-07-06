"""
================================================================================
📌  Testing Different Wrapper Tools with a Multi-Tool Agent
================================================================================
This script builds a multi-tool conversational AI assistant using LangGraph.
It integrates three external lookup capabilities:
  1. ArXiv Query Run (Academic/Scientific Papers lookup)
  2. Wikipedia Query Run (General Encyclopedia Search)
  3. Tavily Search Results (Live Internet/News Search engine)

The agent uses a StateGraph with a Reducer-annotated schema to remember 
the sequential history of conversations and tool executions automatically.
================================================================================
"""

import os
import warnings
import logging
from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv

# LangChain Tool & Wrapper Imports
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
# TODO from langchain_tavily import TavilySearch
#tavily_tool = TavilySearch(k=2)


# Core LangChain & Model Providers
from langchain_groq import ChatGroq
from langchain_core.messages import AnyMessage, HumanMessage

# LangGraph Core State Machine Utilities
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# Suppress standard Python and Transformer warnings/progress logs
warnings.filterwarnings("ignore")
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)


# ==============================================================================
# 1. INITIALIZATION: Environment Variables & Keys Setup
# ==============================================================================
load_dotenv()
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# ==============================================================================
# 2. TOOL DEFINITIONS: Configuring Wrappers & API Frameworks
# ==============================================================================
# 2.1. ArXiv API Wrapper Setup
api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=500)
arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)
print(f"📦 Loaded Tool: {arxiv.name}")

# 2.2. Wikipedia API Wrapper Setup
api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)
print(f"📦 Loaded Tool: {wiki.name}")

# 2.3. Live Tavily Search Setup
tavily = TavilySearchResults()
print(f"📦 Loaded Tool: {tavily.name}")

# Consolidate all instantiated search providers into a singular tools array
tools = [arxiv, wiki, tavily]

# ==============================================================================
# 3. LLM BINDING: Coupling Tools to the LLM Schema Interface
# ==============================================================================
GROQ_MODEL = "qwen/qwen3-32b"
llm = ChatGroq(model=GROQ_MODEL)

# Bind tool definition signatures to the LLM model metadata layer
llm_with_tools = llm.bind_tools(tools)

# ==============================================================================
# 4. LANGGRAPH STATE SCHEMA DEFINITION (With History Reducer)
# ==============================================================================
class State(TypedDict):
    # Annotated + add_messages forces the graph to append updates instead of overriding
    messages: Annotated[list[AnyMessage], add_messages]

# ==============================================================================
# 5. GRAPH ARCHITECTURE: Nodes, Edges, and Compilation
# ==============================================================================
def tool_calling_llm(state: State):
    """Primary inference node executing our multi-tool model over history"""
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Instantiate the StateGraph core builder
builder = StateGraph(State)

# Append structural nodes
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode(tools))

# Establish structural edge logic
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",
    # Routes to 'tools' if the LLM requests a tool execution, else routes directly to END
    tools_condition,
)
builder.add_edge("tools", END)

# Compile into an executable runtime canvas
graph = builder.compile()

# Generate and automatically show system diagram layout on execution
OUTPUT_IMAGE_PATH = "Image_PNGs/ChatbotsWithMultipletools.png"
graph.get_graph().draw_mermaid_png(output_file_path=OUTPUT_IMAGE_PATH)    
os.system(f"open {OUTPUT_IMAGE_PATH}")

# ==============================================================================
# 6. RUNTIME PIPELINE EXECUTION: Testing Diverse Query Formats
# ==============================================================================

# --- Example 1: Academic Paper/ArXiv Query ---
print("\n" + "="*80)
print("📚 TEST CASE 1: Invoking Graph for Arxiv Query (Specific Paper ID)")
print("="*80)
result_arxiv = graph.invoke({"messages": [HumanMessage(content="1706.03762")]})
for m in result_arxiv['messages']:
    m.pretty_print()

# --- Example 2: Live Internet Search / News Query ---
print("\n" + "="*80)
print("🌐 TEST CASE 2: Invoking Graph for Live News Query (Tavily Engine)")
print("="*80)
result_news = graph.invoke({"messages": [HumanMessage(content="Provide me the top 10 recent AI news for March 3rd 2025")]})
for m in result_news['messages']:
    m.pretty_print()

# --- Example 3: General Knowledge / Wikipedia Query ---
print("\n" + "="*80)
print("🔍 TEST CASE 3: Invoking Graph for Wikipedia Query (Encyclopedia Lookup)")
print("="*80)
result_wiki = graph.invoke({"messages": [HumanMessage(content="What is machine learning")]})
for m in result_wiki['messages']:
    m.pretty_print()