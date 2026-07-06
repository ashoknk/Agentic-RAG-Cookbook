import os
import logging
import warnings
from dotenv import load_dotenv

# LangGraph and Pydantic Imports
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel

# --- Configuration ---
# Load environment variables (useful for API keys if adding LLMs later)
load_dotenv()

# --- State Definition ---
# Using Pydantic's BaseModel for the State Schema.
# This ensures runtime data validation for the graph state.
class State(BaseModel):
    name: str

# --- Node Functions ---
def example_node(state: State):
    """
    A simple node function that takes the current state 
    and returns an update for the 'name' key.
    """
    print("--- Example node called ---")
    return {"name": "Hello"}

# --- Graph Construction ---
# 1. Initialize the StateGraph with the Pydantic State schema
builder = StateGraph(State)

# 2. Add nodes to the graph
builder.add_node("example_node", example_node)

# 3. Define the flow logic (edges)
builder.add_edge(START, "example_node")
builder.add_edge("example_node", END)

# 4. Compile the builder into an executable graph
graph = builder.compile()

# --- Execution ---

# Successful invocation: "Peter" is a valid string
print("Result with valid input:")
print(graph.invoke({"name": "Peter"}))

# Error Case Demonstration:
# The following call would trigger a Pydantic Validation Error because 123 is an int, not a str.
# graph.invoke({"name": 123})
# Input should be a valid string [type=string_type, input_value=123, input_type=int]
# For further information visit https://errors.pydantic.dev/2.12/v/string_type