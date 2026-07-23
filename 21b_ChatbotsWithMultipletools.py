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
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from dotenv import load_dotenv

# LangChain Tool & Wrapper Imports
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.memory import MemorySaver
# NOTE; In future this might change to below
# from langchain_tavily import TavilySearch
# tavily_tool = TavilySearch(k=2)

# Core LangChain & Model Providers
from langchain_groq import ChatGroq
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
from groq import BadRequestError


# LangGraph Core State Machine Utilities
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from pydantic import BaseModel, Field

import urllib.request
import xml.etree.ElementTree as ET
from langchain_core.tools import tool

# Suppress standard Python and Transformer warnings/progress logs
warnings.filterwarnings("ignore")
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

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
# https://docs.python.org/3/library/urllib.request.html
@tool
def arxiv_search(query: str) -> str:
    """Search ArXiv for scientific papers using a query or paper ID (e.g., '1706.03762' or 'transformer')."""
    # 1. Query arXiv's official HTTPS API directly
    url = f"https://export.arxiv.org/api/query?search_query=all:{query}&max_results=1"
    
    try:
        response = urllib.request.urlopen(url)
        xml_data = response.read().decode('utf-8')
        
        # 2. Parse XML response for paper Title and Summary
        root = ET.fromstring(xml_data)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        entry = root.find('atom:entry', ns)
        
        if entry is None:
            return "No papers found on ArXiv."
            
        title = entry.find('atom:title', ns).text.strip()
        summary = entry.find('atom:summary', ns).text.strip()
        
        return f"Title: {title}\nSummary: {summary[:300]}..."
    except Exception as e:
        return f"ArXiv Search Error: {e}"

# -----------------
# 2.2. Wikipedia API Wrapper Setup
# Define explicit input schema
class WikipediaInput(BaseModel):
    query: str = Field(description="Search query string for Wikipedia lookup, e.g. 'Machine learning'")

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)

@tool("wikipedia_search", args_schema=WikipediaInput)
def wikipedia_tool(query: str) -> str:
    """Look up factual information, definitions, and history from Wikipedia."""
    try:
        return wiki_wrapper.run(query)
    except Exception as e:
        return f"Error retrieving data from Wikipedia: {str(e)}"
# -----------------

# 2.2. Live Tavily Search Setup
tavily = TavilySearchResults(max_results=1, doc_content_chars_max=500)

# Consolidate all instantiated search providers into a singular tools array
# tools = [arxiv_search, wiki, tavily]
tools = [arxiv_search, wikipedia_tool, tavily]
print(f"📦 Loaded Tools : {arxiv_search.name} , {tavily.name} ,{wikipedia_tool.name} ")

# ==============================================================================
# 3. LLM BINDING: Coupling Tools to the LLM Schema Interface
# ==============================================================================
# GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_MODEL = "llama-3.3-70b-versatile" # NOTE - Try with this model if it helps
llm = ChatGroq(model=GROQ_MODEL, temperature=0)

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
    try:
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}
    except BadRequestError as e:
        # Gracefully handle Groq tool validation errors
        print(f"\n⚠️ Handled Groq Tool Error: {e}")
        fallback_msg = AIMessage(
            content="I ran into an issue formatting the search properly. Could you rephrase your query?"
        )
        return {"messages": [fallback_msg]}

# Instantiate the StateGraph core builder
builder = StateGraph(State)

# Append structural nodes
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode(tools, handle_tool_errors=True))

# Establish structural edge logic
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",
    # Routes to 'tools' if the LLM requests a tool execution, else routes directly to END
    tools_condition,
)
builder.add_edge("tools", END)

# Compile into an executable runtime canvas
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)


# Generate and automatically show system diagram layout on execution
# OUTPUT_IMAGE_FOLDER = "Image_PNGs"
# os.makedirs(OUTPUT_IMAGE_FOLDER, exist_ok=True)
# OUTPUT_IMAGE_PATH = OUTPUT_IMAGE_FOLDER + "/ChatbotsWithMultipletoolsb.png"
# graph.get_graph().draw_mermaid_png(output_file_path=OUTPUT_IMAGE_PATH)    
# os.system(f"open {OUTPUT_IMAGE_PATH}")

# ==============================================================================
# 6. RUNTIME PIPELINE EXECUTION: Testing Diverse Query Formats
# ==============================================================================
config = {"configurable": {"thread_id": "conversation_1"}}

# --- Example 1: Academic Paper/ArXiv Query ---
# Look for "Tool Calls: arxiv_search (abc)"
print("\n" + "="*80)
print("📚 TEST CASE 1: Invoking Graph for Arxiv Query (Specific Paper ID)")
print("="*80)
# "1706.03762" is a computer science paper https://arxiv.org/abs/1706.03762 Attention Is All You Need
result_arxiv = graph.invoke({"messages": [HumanMessage(content="1706.03762")]}
,config=config)
result_arxiv['messages'][-1].pretty_print()
# for m in result_arxiv['messages']:
#     m.pretty_print()

# --- Example 2: Live Internet Search / News Query ---
# Look for "Tool Calls: tavily_search_results_json (abc)"
print("\n" + "="*80)
print("🌐 TEST CASE 2: Invoking Graph for Live News Query (Tavily Engine)")
print("="*80)
result_news = graph.invoke({"messages": [HumanMessage(content="Provide me the top 10 recent AI news for March 3rd 2025")]}
,config=config)
result_news['messages'][-1].pretty_print()
# for m in result_news['messages']:
#     m.pretty_print()

# --- Example 3: General Knowledge / Wikipedia Query ---
# Look for "Tool Calls: wikipedia_search (abc)"
print("\n" + "="*80)
print("🔍 TEST CASE 3: Invoking Graph for Wikipedia Query (Encyclopedia Lookup)")
print("="*80)
# Groq models perform tool calling based on system prompts injected under the hood. Override and reinforce valid JSON spacing by adding a SystemMessage or prompt instruction
sys_msg = SystemMessage(
    content="You are a helpful assistant. When calling tools, ensure arguments are formatted as clean, valid JSON with proper spacing."
)
result_wiki = graph.invoke({
    "messages": [
        sys_msg,
        HumanMessage(content="What is machine learning")
    ]
},config=config)
result_wiki['messages'][-1].pretty_print()
# for m in result_wiki['messages']:
#     m.pretty_print()