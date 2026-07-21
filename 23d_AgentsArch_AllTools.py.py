"""
================================================================================
This final script demonstrates the ultimate benefit of the ReAct blueprint: 
**Compound Tool Orchestration**. It challenges the agent with complex, 
hybrid user prompts that require executing multiple distinct tools sequentially 
(e.g., merging real-time news retrieval with concurrent multi-step arithmetic) 
within a single conversational turn.


1. TOTAL INTEGRATION: Assembles the entire array of search engines, academic 
   scanners, and mathematical operators into a unified execution ecosystem.
2. MODEL SCHEDULING: Binds the combined tools list to the LLM engine and 
   compiles the network graph with persistent checkpoint memory.
3. COMPOUND PROMPT EVALUATION: Submits layered queries containing disparate instruction sets 
   (e.g., extracting recent cyberattack news *and* adding numbers *and* multiplying 
   the intermediate results).
4. EXECUTION FLOW TRACKING: Traces how the graph loops multiple times between the 
   LLM and `ToolNode`, collecting diverse payloads before formatting the final combined answer.
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

# from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
# from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools.tavily_search import TavilySearchResults
# NOTE- For future
# from langchain_tavily import TavilySearch
#tavily_tool = TavilySearch(k=2)

from langchain_groq import ChatGroq
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# Suppress standard Python and Transformer warnings/progress logs
warnings.filterwarnings("ignore")
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

# ### ReAct Agent Architecture
# 1. act - let the model call specific tools
# 2. observe - pass the tool output back to the model
# 3. reason - let the model reason about the tool output to decide what to do next (e.g., call another tool or just respond directly)

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


# ### Custom Functions
def multiply(a: int, b: int) -> int:
    """Multiply a and b.
    Args:
        a: first int
        b: second int
    """
    return a * b

def add(a: int, b: int) -> int:
    """Adds a and b.
    Args:
        a: first int
        b: second int
    """
    return a + b

def divide(a: int, b: int) -> float:
    """Divide a and b.
    Args:
        a: first int
        b: second int
    """
    return a / b

# ### Tavily Search Tool
# tavily = TavilySearchResults()
tavily = TavilySearchResults(
    max_results=2,
    include_answer=False
)

# ### Combine all the tools in the list
tools = [ddg_search, tavily, add, divide, multiply]

# ## Initialize LLM model
llm = ChatGroq(model="qwen/qwen3.6-27b")
llm_with_tools = llm.bind_tools(tools)

# ## State Schema
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# ### Entire Chatbot With LangGraph

# ### Node definition
def tool_calling_llm(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Build graph
builder = StateGraph(State)
builder.add_node("tool_calling_llm", tool_calling_llm)
# builder.add_node("tools", ToolNode(tools))
# This automatically catches tool exceptions and routes the error message back to the LLM
builder.add_node("tools", ToolNode(tools, handle_tool_errors=True))

builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", "tool_calling_llm")

# ### Agent Memory
# #### Aim
# Let's introduce Agent With Memory
memory = MemorySaver()
graph_memory = builder.compile(checkpointer=memory)



# Example Invocation with Memory
config = {"configurable": {"thread_id": "1"}}


from langchain_core.messages import SystemMessage

sys_prompt = SystemMessage(
    content=(
        "You are an expert ReAct reasoning agent capable of compound tool orchestration. "
        "When handling multi-part user prompts, break down the instructions into distinct sequential steps and execute all necessary tools.\n\n"
        
        "TOOL SELECTION RULES:\n"
        "1. 'tavily_search_results_json': Use exclusively for live real-time web searches, breaking news, recent events, and cyberattack reports.\n"
        "2. 'duckduckgo_search': Use for general knowledge, academic paper lookups (e.g., arXiv IDs), historical figures, and encyclopedic facts.\n"
        "3. 'add', 'multiply', 'divide': Use these dedicated math tools strictly for arithmetic operations rather than calculating in head.\n\n"
        
        "COMPOUND WORKFLOW INSTRUCTIONS:\n"
        "- If a prompt requires both information retrieval and math, trigger the search and initial arithmetic operations simultaneously or in logical sequence.\n"
        "- Always complete intermediate math operations before calling dependent math tools (e.g., call 'add' before multiplying the result).\n"
        "- When generating tool calls, ensure parameters are output as clean, valid JSON with proper spacing after the function name.\n"
        "- Keep final responses concise, structured, and clearly separated by sub-task."
    )
)

first_query = [sys_prompt, HumanMessage(content="Provide me the top 3 recent CyberAttack news from last 3 months,add 7 plus 13 and then multiply by 10")]
output1 = graph_memory.invoke({"messages": first_query}, config=config)
output1['messages'][-1].pretty_print()
# for m in output1['messages']:
#     m.pretty_print()

print("\nSleeping 10s to respect Groq TPM rate limits...")
time.sleep(10)  # 👈 Pauses execution so tokens per minute drop

print("\n---------1st tool run using Agentic RAG is done --------- ")

# Follow-up question using memory
second_query = [sys_prompt, HumanMessage(content="Search for the academic arXiv paper ID 1706.03762 and summarize its architecture details and subtract 12 plus 22 and then divide by 2.")]
output_2 = graph_memory.invoke({"messages": second_query}, config=config)
output1['messages'][-1].pretty_print()
# for m in output_2['messages']:
#     m.pretty_print()

print("\nSleeping 10s to respect Groq TPM rate limits...")
time.sleep(10)  # 👈 Pauses execution so tokens per minute drop

print("\n---------2nd tool run using Agentic RAG is done --------- ")

# input_msg = [HumanMessage(content="Provide me the top 3 recent CyberAttack news from last 3 months,add 13 plus 12 and then multiply by 10.")]
# Follow-up question using memory
# follow_up = [HumanMessage(content="Provide me the top 3 recent Google stock news from last 6 months,multiply 3 plus 8 and then divide by 6.")]

