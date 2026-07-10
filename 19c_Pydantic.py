"""
================================================================================
This lab introduces **Pydantic `BaseModel`** state definitions within LangGraph to 
enforce runtime data validation and schema protection across all nodes.


THE PRODUCTION SAFEGUARD:
-------------------------
Unlike `TypedDict` and basic `dataclasses` (which allow improper parameter types 
to leak into nodes), a Pydantic state graph validates inputs at the boundary, 
instantly rejecting improper data types before any graph logic executes.

1. STRONG SHIELD DEFINITION: Designs a robust `State` structure inheriting from 
   Pydantic's foundational `BaseModel` class.
2. NODE INTERFACE: Implements a sample processing node that securely processes 
   the verified state object.
3. WORKFLOW RUNTIME: Compiles the state map into an executable entity.
4. COMPARATIVE ASSESSMENTS:
   - **Success Condition**: Submits a valid text string payload to verify correct execution.
   - **Error Catch Demonstration**: Documents how submitting invalid input types 
     instantly triggers a explicit `ValidationError` crash to protect the pipeline.
================================================================================
"""

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