"""
================================================================================
This script delivers the solution to the state override dilemma explored in `20a`. 
It introduces the specialized **LangGraph Reducer framework** to create a resilient, 
append-only state architecture that preserves conversational history across nodes.


THE REDUCER MECHANISM:
----------------------
Using `Annotated[list[AnyMessage], add_messages]` modifies how LangGraph manages 
internal updates. Instead of performing a destructive dictionary replacement, 
it routes all incoming node updates through the `add_messages` function, automatically 
appending updates to the core conversation history list.

1. SCHEMA CONFIGURATION: Constructs a `State` blueprint equipped with an 
   annotated message append reducer.
2. SIMULATION RUNTIME: Validates the state accumulation process by calling `add_messages` 
   manually to prove list persistence.
3. AGENT TOOL PIPELINE COMPILATION: Assembles a state graph mapping an LLM chatbot node 
   to a pre-built `ToolNode` array using a conditional router link (`tools_condition`).
4. END-TO-END VERIFICATION: Launches a live tool call prompt to confirm that the 
   compiled graph successfully tracks user inputs, intermediary tool requests, 
   and final generated answers within a unified, non-destructive history stack.
================================================================================
"""

import os
from dotenv import load_dotenv
from typing_extensions import TypedDict
from typing import Annotated

# LangChain & LangGraph Utilities
from langchain_groq import ChatGroq
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# ==============================================================================
# 1. INITIAL PREPARATION: Environment & Keys Setup
# ==============================================================================
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

GROQ_MODEL = "llama-3.1-8b-instant"
llm = ChatGroq(model=GROQ_MODEL)

def add(a: int, b: int) -> int:
    """Add a and b"""
    return a + b

llm_with_tools = llm.bind_tools([add])

# ==============================================================================
# 2. DEFINING THE REDUCER SCHEMA
# ==============================================================================
print("======================================================================")
print("✨ PART 1: THE SOLUTION - STATE REDUCER FUNCTION")
print("======================================================================")

# By using Annotated alongside the pre-built 'add_messages' reducer function,
# we instruct LangGraph to APPEND incremental changes to the state history array
# instead of completely overriding the parameter keys.
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# Mocking manual list append processing check
initial_messages = [HumanMessage(content="What is 2 plus 2", name="Ash")]
new_ai_message = AIMessage(content="Let me look up that calculation for you.", name="LLMModel")

print("Executing manual 'add_messages' operation simulation:")
accumulated_result = add_messages(initial_messages, new_ai_message)
print(f"Combined Accumulation Array Size: {len(accumulated_result)} elements.")
for msg in accumulated_result:
    print(f"  - {msg.__class__.__name__}: {msg.content}")

# ==============================================================================
# 3. WORKFLOW NODE DEFINITIONS & COMPILATION
# ==============================================================================
def llm_tool_node(state: State):
    """Chatbot execution node"""
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Initialize StateGraph with the Reducer-annotated structural schema
builder = StateGraph(State)

# Append computational logical segments
builder.add_node("llm_tool", llm_tool_node)
builder.add_node("tools", ToolNode([add]))

# Build Edge Router Channels
builder.add_edge(START, "llm_tool")
builder.add_conditional_edges(
    "llm_tool",
    tools_condition  # Routes to 'tools' if tool_calls exist, else routes to END
)
builder.add_edge("tools", END)

# Compile Execution Canvas
graph_compiled = builder.compile()

OUTPUT_IMAGE_PATH = "Image_PNGs/State_Graph_Reducers.png"
graph_compiled.get_graph().draw_mermaid_png(output_file_path=OUTPUT_IMAGE_PATH)    
# 2. Automatically display/open the image on macOS
os.system(f"open {OUTPUT_IMAGE_PATH}")

# ==============================================================================
# 4. RUNTIME SYSTEM EXECUTION (Verifying the Accumulation)
# ==============================================================================
print("\n======================================================================")
print("🚀 PART 2: GRAPH INVOKE LOGS (VERIFYING APPEND OVER MEMORY KEY)")
print("======================================================================")

user_prompt = "What is 2 plus 2?"
print(f"Invoking graph workflow with prompt: '{user_prompt}'\n")

# graph_compiled.invoke() triggers and manages the overall graph execution, 
# which internally calls llm_with_tools.invoke() when it enters the llm_tool node.
execution_history_output = graph_compiled.invoke(
    {"messages": [HumanMessage(content=user_prompt, name="Ash")]}
)

print("📬 FINAL STATE MESSAGES ARRAY OBJECT RECOVERED:")
print("----------------------------------------------------------------------")
# Notice how the array contains the initial question, the model's tool call,
# the tool's result payload execution, and the final analytical answer.
# The reducer successfully preserved everything!
for idx, message in enumerate(execution_history_output["messages"], 1):
    print(f"[{idx}] {message.__class__.__name__} (Author: {message.name})")
    print(f"    Content: {message.content}")
    if hasattr(message, 'tool_calls') and message.tool_calls:
        print(f"    Triggered Tool Arguments: {message.tool_calls}")
    print("-" * 70)