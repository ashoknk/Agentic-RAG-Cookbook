"""
================================================================================
This advanced script scales the ReAct agent framework to support complex, 
highly technical research domains by integrating professional enterprise search 
toolkits. It maps a robust network of APIs capable of parsing software 
engineering exceptions, computing formal calculus integrations, and defining language etymologies.

THE SPECIALIZED TOOL SUITE:
---------------------------
https://reference.langchain.com/python/langchain-community/utilities
https://reference.langchain.com/python/langchain-community/utilities/stackexchange/StackExchangeAPIWrapper


- `StackExchangeTool`: Interrogates technical developer forums to extract verified debugging discussions and exception remedies.

1. SUITE PROVISIONING: Configures API wrappers with distinct result ceilings and 
   registers them to a single tool list alongside custom math components.
2. ARCHITECTURE COMPILATION: Wraps the ecosystem inside a persistent `StateGraph` 
   checkpointing layer with automated exception protection.
3. AGENT TUNNEL EXECUTION: Launches targeted validation runs to track model 
   routing: querying Python attribute errors (StackExchange), processing numeric 
   definite integrals 
================================================================================
"""

import os
import warnings
import logging
import time

from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv

# Set a custom User-Agent identifying your application
os.environ["USER_AGENT"] = "Agentic-RAG-Cookbook/1.0 (contact: ash@codeaiwashnaiku.com)"

# New Tech, Math, and Dictionary Tool Imports
from langchain_community.tools import StackExchangeTool
from langchain_community.utilities import StackExchangeAPIWrapper
from langchain_community.utilities import PubMedAPIWrapper
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from langchain_community.utilities import NasaAPIWrapper
from langchain_community.tools.nasa.tool import NasaAction


from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

# Suppress standard Python and Transformer warnings/progress logs
warnings.filterwarnings("ignore")
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

# ==============================================================================
# 1. INITIAL PREPARATION: Environment & Keys Setup
# ==============================================================================
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

os.environ["NASA_API_KEY"] = os.getenv("NASA_API_KEY")


os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "ReAct-agent"

# ==============================================================================
# 2. TOOL INITIALIZATION (StackExchange, Merriam-Webster)
# ==============================================================================

# 2.1. Initialize Stack Exchange (Developer Q&A)
api_wrapper_stack = StackExchangeAPIWrapper(max_results=2)
stack = StackExchangeTool(api_wrapper=api_wrapper_stack)
print(f"📦 Loaded Tool: {stack.name}")


# 2.2. Initialize PubMed and Nasa 


# Initialize PubMed Wrapper (Parameters like top_k_results control retrieved output)
api_wrapper_pubmed = PubMedAPIWrapper(top_k_results=2)

# Bind the wrapper to the tool
pubmed_tool = PubmedQueryRun(api_wrapper=api_wrapper_pubmed)

print(f"📦 Loaded Tool: {pubmed_tool.name}")


# Initialize NASA Wrapper
api_wrapper_nasa = NasaAPIWrapper()

# Bind the wrapper to the tool
nasa_tool = NasaAction(
    api_wrapper=api_wrapper_nasa, 
    mode="search_media"  # 👈 Required parameter (e.g., 'search_media')
)
nasa_tool.name = "nasa_action"

print(f"📦 Loaded Tool: {nasa_tool.name}")

# ==============================================================================
# 3. CUSTOM MATH TOOLS & LLM BINDING
# NOTE - Might not be used during run time but kept here to show that the correct tool is being caled 
# ==============================================================================
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

# Combine all active tools into a unified list

# Combine active tools into unified list
tools = [
    stack, 
    pubmed_tool,
    nasa_tool,
    add, 
    subtract, 
    multiply, 
    divide
]

# Initialize LLM model and bind the tools

# GROQ_MODEL = "llama-3.1-8b-instant"
# GROQ_MODEL = "llama-3.3-70b-versatile"
# GROQ_MODEL="qwen/qwen3.6-27b"
# GROQ_MODEL="openai/gpt-oss-120b""
# llm = ChatGroq(
#     model=GROQ_MODEL, 
#     temperature=0,
#     request_timeout=15  # 👈 Fails gracefully if Groq takes > 15s
# )

OPENAI_MODEL="gpt-4o-mini"
llm = ChatOpenAI(
    model=OPENAI_MODEL,
    temperature=0,
    timeout=15,  # 👈 Modern OpenAI parameter
)

llm_with_tools = llm.bind_tools(tools)

# ==============================================================================
# 4. LANGGRAPH SCHEMA & GRAPH CONSTRUCT
# ==============================================================================
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def tool_calling_llm(state: State):
    """Primary inference node executing our multi-tool model over history"""
    print("⏳ Sending request to LLM...")
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Build graph structure
builder = StateGraph(State)
builder.add_node("tool_calling_llm", tool_calling_llm)
# Enforce native tool error catching so network/API errors don't crash the script
builder.add_node("tools", ToolNode(tools, handle_tool_errors=True))

builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",
    tools_condition,
)
builder.add_edge("tools", "tool_calling_llm")
builder.add_edge("tools", END)

# Compile with persistent memory checkpointing
memory = MemorySaver()
graph_memory = builder.compile(checkpointer=memory)

# ==============================================================================
# 5. RUNTIME AGENTIC TUNNEL EXECUTION
# ==============================================================================


sys_prompt = SystemMessage(
    content=(
        "You are a helpful and concise research assistant with access to specialized tools:\n"
        "1. 'stack_exchange': Use strictly for software development questions, code errors, debugging, and programming exceptions.\n"
        "2. 'PubMed': Use for biomedical, clinical, medical literature, and life sciences research queries.\n"
        "3. 'nasa_action': Use for searching NASA space media, satellite/planetary datasets, and astronomical imagery.\n"
        "4. 'add', 'subtract', 'multiply', 'divide': Use for basic arithmetic calculations.\n\n"
        "CRITICAL INSTRUCTIONS:\n"
        "- NEVER invent or call tools outside this list (e.g., do NOT call 'google_search' or 'merriam_webster').\n"
        "- Choose the single best tool suited for the request.\n"
        "- Keep final responses clear, professional, and succinct."
    )
)

# --- Query 1: Targeting StackExchange ---
print("\n" + "="*80)
print("💻 TEST CASE 1: Targeting StackExchange (Developer Exception Lookup)")
print("="*80)
config1 = {"configurable": {"thread_id": "1"}}
first_query = [sys_prompt, HumanMessage(content="Search StackExchange for discussions on how to resolve a python AttributeError.")]
output1 = graph_memory.invoke({"messages": first_query}, config=config1)
output1['messages'][-1].pretty_print()
for m in output1['messages']:
    m.pretty_print()
print("\n--------- First tool run using Agentic RAG is done --------- ")


print("\nSleeping to respect TPM rate limits...")
time.sleep(3)  # 👈 Pauses execution so tokens per minute drop

# --- Query 2: Targeting PubMed (Medical Research) ---
print("\n" + "="*80)
print("🩺 TEST CASE 2: Targeting PubMed (Biomedical Research Lookup)")
print("="*80)
config2 = {"configurable": {"thread_id": "2"}}
second_query = [
    sys_prompt, 
    HumanMessage(content="Search PubMed for recent research abstracts regarding mRNA vaccine mechanisms.")
]
output_2 = graph_memory.invoke({"messages": second_query}, config=config2)
output_2['messages'][-1].pretty_print()
print("\n--------- 2nd tool run (PubMed) is done --------- ")

time.sleep(3)  # 👈 Pauses execution so tokens per minute drop

# --- Query 3: Targeting NASA (Space & Astronomy Search) ---
print("\n" + "="*80)
print("🚀 TEST CASE 3: Targeting NASA (Space Media Search)")
print("="*80)
config3 = {"configurable": {"thread_id": "3"}}
third_query = [
    sys_prompt, 
    HumanMessage(content="Search NASA for high-resolution images and media related to the James Webb Space Telescope Nebulae observations.")
]
output_3 = graph_memory.invoke({"messages": third_query}, config=config3)
output_3['messages'][-1].pretty_print()
print("\n--------- 3rd tool run (NASA) is done --------- ")