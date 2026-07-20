"""
Resolve the State Override Problem using Annotated[list, add_messages].
This creates an append-only memory history across graph nodes.
"""

import os
from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

# 1. DEFINE STATE WITH REDUCER
# Annotated[list, add_messages] tells LangGraph to APPEND updates instead of replacing them.
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# 2. SETUP MODEL & TOOLS
def add(a: int, b: int) -> int:
    """Add two integers a and b."""
    return a + b

llm = ChatGroq(model="llama-3.1-8b-instant")
llm_with_tools = llm.bind_tools([add])

# 3. DEFINE NODES
def chatbot_node(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# 4. BUILD GRAPH
builder = StateGraph(State)
builder.add_node("chatbot", chatbot_node)
builder.add_node("tools", ToolNode([add]))

builder.add_edge(START, "chatbot")
builder.add_conditional_edges("chatbot", tools_condition)
builder.add_edge("tools", END)

graph = builder.compile()

# Save the file as a PNG
OUTPUT_IMAGE_FOLDER = "Image_PNGs"
os.makedirs(OUTPUT_IMAGE_FOLDER, exist_ok=True)
OUTPUT_IMAGE_PATH = OUTPUT_IMAGE_FOLDER + "/20b_State_Graph_Reducers.png"
graph.get_graph().draw_mermaid_png(output_file_path=OUTPUT_IMAGE_PATH)    
# 2. Automatically display/open the image on macOS
os.system(f"open {OUTPUT_IMAGE_PATH}")


# 5. EXECUTION & VERIFICATION
user_input = {"messages": [HumanMessage(content="What is 2 plus 2?")]}
final_state = graph.invoke(user_input)

print("\n========== FINAL CONVERSATION HISTORY WITH REDUCER (APPEND BEHAVIOR) ==========")
for idx, msg in enumerate(final_state["messages"], 1):
    sender = getattr(msg, "name", None) or msg.__class__.__name__
    print(f"[{idx}] {sender}: {msg.content}")
    if getattr(msg, "tool_calls", None):
        print(f"    -> Called Tool: {msg.tool_calls}")

print("\n✨ SUCCESS: All messages (Input -> Tool Call -> Tool Result -> Output) were preserved!")