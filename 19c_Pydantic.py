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

# LangGraph and Pydantic Imports
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel
from IPython.display import Image,display

# --- State Definition ---
# Using Pydantic's BaseModel for the State Schema.
# This ensures runtime data validation for the graph state.
# https://pydantic.dev/docs/validation/latest/api/pydantic/base_model/
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

# Save the file as a PNG
OUTPUT_IMAGE_FOLDER = "Image_PNGs"
os.makedirs(OUTPUT_IMAGE_FOLDER, exist_ok=True)

OUTPUT_IMAGE_PATH = OUTPUT_IMAGE_FOLDER + "/Pydantic1.png"
graph.get_graph().draw_mermaid_png(output_file_path=OUTPUT_IMAGE_PATH)    
# 2. Automatically display/open the image on macOS
os.system(f"open {OUTPUT_IMAGE_PATH}")

# --- Execution ---

# Successful invocation: "Peter" is a valid string
print("Result with valid input:")
print(graph.invoke({"name": "PeterPan"}))

# Error Case Demonstration:
# The following call would trigger a Pydantic Validation Error because 123 is an int, not a str.
# graph.invoke({"name": 123})
# Input should be a valid string [type=string_type, input_value=123, input_type=int]
# For further information visit https://errors.pydantic.dev/2.12/v/string_type