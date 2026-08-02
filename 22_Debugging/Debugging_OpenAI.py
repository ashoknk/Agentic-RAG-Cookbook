"""
This script builds a tool-calling AI agent using LangGraph and LangChain. 
It demonstrates a foundational agentic pattern: giving an LLM access to external
Python functions (tools) and using a conditional execution loop to process user
requests dynamically.

KEY CONCEPTS DEMONSTRATED:
1. Tool Definition (@tool): Wrapping custom Python functions for LLM execution.
2. Model Binding (bind_tools): Informing the LLM about available tools and schemas.
3. Prebuilt Tool Execution (ToolNode): Automatically running tool calls requested by the model.
4. Conditional Routing (add_conditional_edges): Inspecting LLM messages to decide whether 
   to execute a tool or finish the conversation loop.

What is LangSmith?
LangSmith is an observability, tracing, and debugging platform built specifically for LLM applications and AI agents.

https://smith.langchain.com/


1.Full Visibility (Tracing): You can see the exact sequence of thoughts, 
tool calls, raw prompts, and LLM responses at every node in the graph.

2. Local-to-Cloud Connection: Even though your code runs locally on your computer
 (http://127.0.0.1:2024), the LangSmith UI renders the visual graph and provides a live debugging interface.

3.State Inspection & Time Travel: You can click on any step in a conversation thread , 
inspect the State dictionary at that exact moment
================================================================================
"""

import os
from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
from langgraph.graph import END, START
from langgraph.graph.state import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


# Load Environment Variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")

class State(TypedDict):
    # 'add_messages' ensures new messages are appended to history rather than overwriting it
    messages: Annotated[list[BaseMessage], add_messages]

# Initialize the model explicitly using model name to avoid model permission conflicts
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def make_alternative_graph():
    """Make a tool-calling agent for debugging."""

    # Define a simple custom tool with clear docstrings and type 
    # call_7tka5c3A2YoSsqBWkw9GOKeG
    @tool
    def add(a: float, b: float):
        """Adds two numbers."""
        return a + b

    # ==========================================================================
    # TOOL INTEGRATION & MODEL BINDING .
    # ==========================================================================
    # 1. ToolNode: A prebuilt LangGraph node that automatically executes tool calls 
    tool_node = ToolNode([add])

    # 2. bind_tools: So the model knows 'add' exists and when to invoke it.
    model_with_tools = model.bind_tools([add])

    # 3. Agent Node: Passes current state/chat history to the tool-aware model
    def call_model(state: State):
        return {"messages": [model_with_tools.invoke(state["messages"])]}

    # 4. Conditional Router: Inspects the latest message.
    #    If the model added 'tool_calls' to its message, route to 'tools' node.
    #    Otherwise, complete the execution loop by routing to END.
    def should_continue(state: State):
        if state["messages"][-1].tool_calls:
            return "tools"
        else:
            return END

    graph_workflow = StateGraph(State)

    # Add execution nodes
    graph_workflow.add_node("agent", call_model)
    graph_workflow.add_node("tools", tool_node)
    
    
    graph_workflow.add_edge(START, "agent")
    # Return edge: After executing a tool, return to the 'agent' to interpret the tool result
    graph_workflow.add_edge("tools", "agent")
    graph_workflow.add_conditional_edges(
            "agent", 
            should_continue,
            {
                "tools": "tools",  # If should_continue returns "tools" -> go to 'tools' node
                END: END           # If should_continue returns END     -> finish execution
            }
        )

    return graph_workflow.compile()

# Instantiate the graph for execution/debugging
agent = make_alternative_graph()