"""
================================================================================
This script expands upon the core ReAct agent pattern by shifting focus to 
**Dynamic Multi-Domain Information Retrieval**. It showcases how a single 
reasoning engine can dynamically choose between different external research 
apis based on the context of the user's inquiry.

### ReAct Agent Architecture ###
The intuition behind ReAct, a general agent architecture.
1. act - let the model call specific tools
2. observe - pass the tool output back to the model
3. reason - let the model reason about the tool output to decide what to do next 
    (e.g., call another tool or just respond directly)

THE VALUE OF API TOOL DIVERSIFICATION:
--------------------------------------
Instead of forcing a single general-purpose web search engine to answer all 
prompts, this architecture introduces specialized lookup APIs. The model 
autonomously determines the optimal source channel: 
    -Use 'duckduckgo_search' for general queries, history, paper IDs, and biographies. 
    -Use 'tavily_search_results_json' for live news. Keep tool responses succinct and direct.

1. INITIALIZATION: Registers specialized academic, encyclopedia, and web search 
   wrappers ( `DuckDuckGo`, `TavilySearchResults`) alongside arithmetic utilities.
2. GRAPH WIRING: Compiles an identical ReAct state graph architecture backed by the `MemorySaver` checkpointer mechanism.
3. DISPARATE RUNTIME TESTING:
   - **Run 1 : Use Tavily to search for the top 3 recent CyberAttack news from the last 3 months.
   - **Run 2 : Search DuckDuckGo for the arXiv paper '1706.03762' and summarize its architecture.
================
"""

import os
import warnings
import logging
import time

from typing import Annotated
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from typing_extensions import TypedDict
from dotenv import load_dotenv

# Set a custom User-Agent identifying your application
os.environ["USER_AGENT"] = "Agentic-RAG-Cookbook/1.0 (contact: ash@codeaiwashnaiku.com)"

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools.tavily_search import TavilySearchResults
# NOTE for future
#from langchain_tavily import TavilySearch
#tavily_tool = TavilySearch(k=2)

from langchain_groq import ChatGroq
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool


# Suppress standard Python and Transformer warnings/progress logs
warnings.filterwarnings("ignore")
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

# Load environment variables
load_dotenv()
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "ReAct-agent"

# 2.1. DuckDuckGo API Setup
ddg_search = DuckDuckGoSearchRun()
ddg_search.name = "duckduckgo_search"
ddg_search.description = (
    "A search engine for general knowledge, encyclopedia entries, historical figures, "
    "and paper summaries. Use this tool for queries about history, people, or facts."
)
print(f"📦 Loaded Tool: {ddg_search.name}")

# 2.2. Tavily Search Setup
tavily = TavilySearchResults(max_results=1, doc_content_chars_max=500)
print(f"📦 Loaded Tool: {tavily.name}")

# # ### Tavily Search Tool
# tavily = TavilySearchResults(
#     max_results=1,
#     include_answer=False
# )

# NOTE : These tools are used for testing purpose here to show they are not being called 

# ### Custom Functions
# LangChain needs to inspect type hints and docstrings of custom Python Functions to generate the JSON schema for the LLM.
# Under the hood, bind_tools() and ToolNode() automatically convert plain functions into tools for you.
# But best practice to use @tool on custom functions, even though modern LangChain can often automatically infer tools in bind_tools().
@tool("add")
def add(a: int, b: int) -> int:
    """Adds a and b.
    Args:
        a: first int
        b: second int
    """
    return a + b

@tool("subtract")
def subtract(a: int, b: int) -> int:
    """Subtracts b from a.
    Args:
        a: first int
        b: second int
    """
    return a - b

@tool("multiply")
def multiply(a: int, b: int) -> int:
    """Multiply a and b.
    Args:
        a: first int
        b: second int
    """
    return a * b

@tool("divide")
def divide(a: int, b: int) -> float:
    """Divide a and b.
    Args:
        a: first int
        b: second int
    """
    return a / b

#Why no @tool
# Pre-built Tool Objects (ddg_search, tavily) are instances of specialized classes (DuckDuckGoSearchRun, TavilySearchResults)
# that are already LangChain tools.

# ### Combine all the tools in the list
tools = [ddg_search, tavily, add, subtract, multiply, divide]

#Initialize LLM model
GROQ_MODEL = "llama-3.1-8b-instant"
# llm = ChatGroq(model="qwen/qwen3.6-27b")
llm = ChatGroq(model=GROQ_MODEL, temperature=0)
# Decision-Making - `tools` to teach the LLM what tools exist.
llm_with_tools = llm.bind_tools(tools)

# ## State Schema
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

#Node definition
def tool_calling_llm(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Build graph
builder = StateGraph(State)
builder.add_node("tool_calling_llm", tool_calling_llm)
# `handle_tool_errors` automatically catches tool exceptions and routes the error message back to the LLM
# Execution - `tools` for actually running the Python functions.
builder.add_node("tools", ToolNode(tools, handle_tool_errors=True))

builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", "tool_calling_llm")
builder.add_edge("tools", END)

# ### Agent Memory
memory = MemorySaver()
graph_memory = builder.compile(checkpointer=memory)

# Example Invocation with Memory
config1 = {"configurable": {"thread_id": "1"}}
# --- Prompt 1: Targets Tavily (Recent Live News) ---
sys_prompt = SystemMessage(
    content=(
        "You are a concise assistant. Use 'duckduckgo_search' for general queries, history, paper IDs, and biographies. "
        "Use 'tavily_search_results_json' for live news. Keep tool responses succinct and direct."
    )
)

first_query = [
    sys_prompt,
    HumanMessage(content="Use Tavily to search for the top 3 recent CyberAttack news from the last 3 months.")
]
# output1 = graph_memory.invoke({"messages": first_query}, config=config1)
# output1['messages'][-1].pretty_print()
# for m in output1['messages']:
#     m.pretty_print()
print("\nSleeping 10s to respect Groq TPM rate limits...")
# time.sleep(10)  # 👈 Pauses execution so tokens per minute drop

print("\n---------1st tool run using Agentic RAG is done --------- ")


# Follow-up question using memory
# "1706.03762" is a computer science paper https://arxiv.org/abs/1706.03762 Attention Is All You Need
# --- Prompt 2: Targets DuckDuckGo (Academic/General Lookup) ---
config2 = {"configurable": {"thread_id": "2"}}
second_query = [
    sys_prompt,
    HumanMessage(content="Search DuckDuckGo for the arXiv paper '1706.03762' and summarize its architecture.")
]
output_2 = graph_memory.invoke({"messages": second_query}, config=config2)
output_2['messages'][-1].pretty_print()
# for m in output_2['messages']:
#     m.pretty_print()
print("\nSleeping 10s to respect Groq TPM rate limits...")
time.sleep(10)  # 👈 Pauses execution so tokens per minute drop

print("\n---------2nd tool run using Agentic RAG is done --------- ")

# Follow-up question using memory
# https://en.wikipedia.org/wiki/Alan_Turing
# --- Prompt 3: Targets DuckDuckGo (Biographical/Fact Lookup) ---
# config3 = {"configurable": {"thread_id": "3"}}
# third_query = [
#     sys_prompt,
#     HumanMessage(content="Search DuckDuckGo for 'J. Robert Oppenheimer' and state his exact birth date.")
# ]
# output_3 = graph_memory.invoke({"messages": third_query}, config=config3)
# # output_3['messages'][-1].pretty_print()
# for m in output_3['messages']:
#     m.pretty_print()    


# print("\n---------3rd tool run using Agentic RAG is done --------- ")