import os
import warnings
import logging

from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv

# Set a custom User-Agent identifying your application
os.environ["USER_AGENT"] = "MyAgenticRAGApp/1.0 (contact: ashoknkjp@gmail.com)"

from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
# TODO from langchain_tavily import TavilySearch
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
#
# #### Aim
# This is the intuition behind ReAct, a general agent architecture.
#
# 1. act - let the model call specific tools
# 2. observe - pass the tool output back to the model
# 3. reason - let the model reason about the tool output to decide what to do next (e.g., call another tool or just respond directly)

# Initialize Arxiv tool
api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=500)
arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)
print(f"Tool name: {arxiv.name}")

# Initialize Wikipedia tool
api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)
print(f"Tool name: {wiki.name}")

# NOTE : These 2 tools are used for testing purpose here tos show they are not being called when using math problems in prompt

# Load environment variables
load_dotenv()
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "ReAct-agent"

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
tools = [arxiv, wiki, tavily, add, divide, multiply]

# ## Initialize LLM model
llm = ChatGroq(model="qwen/qwen3-32b")
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

# Example Invocation with Memory
config = {"configurable": {"thread_id": "1"}}
first_query = [HumanMessage(content="Provide me the top 3 recent CyberAttack news from last 3 months,add 7 plus 13 and then multiply by 10")]
output1 = graph_memory.invoke({"messages": first_query}, config=config)

for m in output1['messages']:
    m.pretty_print()

print("\n---------1st tool run using Agentic RAG is done --------- ")

# Follow-up question using memory
second_query = [HumanMessage(content="Search for the academic arXiv paper ID 1706.03762 and summarize its architecture details and subtract 12 plus 22 and then divide by 2.")]
output_2 = graph_memory.invoke({"messages": second_query}, config=config)

for m in output_2['messages']:
    m.pretty_print()


print("\n---------2nd tool run using Agentic RAG is done --------- ")

# input_msg = [HumanMessage(content="Provide me the top 3 recent CyberAttack news from last 3 months,add 13 plus 12 and then multiply by 10.")]
# Follow-up question using memory
# follow_up = [HumanMessage(content="Provide me the top 3 recent Google stock news from last 6 months,multiply 3 plus 8 and then divide by 6.")]

