"""
================================================================================
This script builds directly upon the foundational Pydantic concepts of `19c`, 
showcasing an advanced execution flow built over a non-linear conditional 
graph topology. It highlights **Auto-Coercion** and fallback parameter defaults 
as core data management advantages within an AI workflow.

THE DUAL REASONING BENEFITS:
- **Default Value Fallbacks**: Allows fields to automatically fall back to preset 
  values (e.g., `game: Literal["soccer"] = "soccer"`) if omitted from inputs.
- **Data Type Coercion**: Smart casting mechanics where parsing an unexpected data 
  type (like an integer `123` into a string slot) is safely cast into its valid representation 
  (`"123"`) instead of crashing the pipeline.

1. PRODUCTION ARCHITECTURE: Declares a Pydantic `State` model containing default options.
2. RECURSIVE LAYOUT DEFINITION: Assembles a multi-node workflow using object dot-notation 
   and links them via conditional edge configurations.
3. VERIFICATION RUNS: Validates standard execution paths, default value processing, 
   and type casting features by checking variable states throughout execution.
================================================================================
"""

import os
import random
from typing import Optional, Literal

# Pydantic Import
from pydantic import BaseModel

# LangChain and LangGraph Imports
from langgraph.graph import StateGraph, START, END

# ### ========= Pydantic BaseModel State ========= ###
# Pydantic's BaseModel provides structured data with strict runtime type enforcement.
# It automatically coerces and validates data entering or moving through the graph.

class State(BaseModel):
    name: str
    # game: Literal["soccer", "pickeball"] = "soccer"  # Default fallback values allowed
    game: Optional[Literal["soccer", "pickeball"]] = None

# Node access uses dot notation (.name) identical to Dataclasses
def play_game_pydantic(state: State):
    print("---Play Game node has been called--")
    return {"name": state.name}

def pickeball_pydantic(state: State):
    print("-- pickeball node has been called--")
    return {"name": state.name, "game": "pickeball"}

def soccer_pydantic(state: State):
    print("-- soccer node has been called--")
    return {"name": state.name, "game": "soccer"}

def decide_play_pydantic(state: State) -> Literal["pickeball", "soccer"]:
    if random.random() < 0.5:
        return "pickeball"
    else:
        return "soccer"

# Building Graph with Pydantic BaseModel
builder_pydantic = StateGraph(State)
builder_pydantic.add_node("playgame", play_game_pydantic)
builder_pydantic.add_node("pickeball", pickeball_pydantic)
builder_pydantic.add_node("soccer", soccer_pydantic)

builder_pydantic.add_edge(START, "playgame")
builder_pydantic.add_conditional_edges("playgame", decide_play_pydantic)
builder_pydantic.add_edge("pickeball", END)
builder_pydantic.add_edge("soccer", END)

graph_pydantic = builder_pydantic.compile()

print("======== Using Pydantic BaseModel =========")
print("-" * 50)

# Invoke calls passing clean Pydantic object states
result_pd1 = graph_pydantic.invoke(State(name="PeterPan"))
print(f"Result 1:\n{result_pd1}\n")

result_pd2 = graph_pydantic.invoke(State(name="TinkerBell"))
print(f"Result 2:\n{result_pd2}\n")

# CRITICAL PRO-TIP DEMONSTRATION: Runtime Coercion vs Error
# NOTE Try to change "123" to 123 . Also try str(123)
# Pydantic catches 'name=123' and safely casts it to a string ("123") before execution!
# result_pd3 = graph_pydantic.invoke(State(name=123))
result_pd3 = graph_pydantic.invoke(State(name="123"))
print(f"Result 3 (Auto-Coerced Integer):")
print(result_pd3)
print(f"Verified Type of 'name' parameter: {type(result_pd3['name'])}")