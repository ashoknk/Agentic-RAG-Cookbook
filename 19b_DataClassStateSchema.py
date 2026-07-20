"""
================================================================================
This script introduces Python's standard **`@dataclass`** decorator as an alternative 
state schema layer within a LangGraph pipeline. It contrasts against 
dictionary lookups by demonstrating object-oriented attribute tracking.

THE NOTATION HIGHLIGHT:
-----------------------
- **Dictionary Lookups (`state['name']`)**: Utilized across traditional `TypedDict` formats.
- **Dot Notation (`state.name`)**: Enabled natively via Dataclasses, offering 
  cleaner, more human-readable object manipulation within graph node logic.

1. CLASS DECORATION: Constructs a state representation class structured around the 
   native `@dataclass` format wrapper.
2. NODE DESIGN: Configures logical worker functions that interface with the state 
   parameters natively using object dot-notation signatures.
3. GRAPH RE-COMPILATION: Registers nodes, hardcodes initialization vectors, and 
   sets up conditional branching paths over the dataclass instance.
4. OBJECT VALIDATION: Instantiates and submits `DataClassState` payload packages 
   directly into the workflow engine to track terminal execution states.
================================================================================
"""

import os
import logging
import warnings
import random
from dataclasses import dataclass
from typing import Optional, Literal
from typing_extensions import TypedDict

# LangChain and LangGraph Imports
from langgraph.graph import StateGraph, START, END


# ### State Schema With DataClasses
# When we define a LangGraph StateGraph, we use a state schema.
# The state schema represents the structure and types of data that our graph will use.
# All nodes are expected to communicate with that schema.


# ### ========= Dataclasses ========= ###
# Python's dataclasses provide another way to define structured data.
# Dataclasses offer a concise syntax for creating classes that are primarily used to store data.

# Standard Python @dataclass decorators do not perform runtime type checking or validation.
# The type annotations inside a dataclass (name: str, game: Literal[...]) act purely as type hints 
# for static type checkers (like Pyright/MyPy) and IDE auto-completion.

@dataclass
class DataClassState:
    name: str
    game: Optional[Literal["soccer", "pickeball"]] = None

def play_game_dc(state: DataClassState):
    print("---Play Game node has been called--")
    return {"name": state.name }

def pickeball_dc(state: DataClassState):
    print("-- Pickeball node has been called--")
    return {"name": state.name , "game": "pickeball"}

def soccer_dc(state: DataClassState):
    print("-- Soccer node has been called--")
    return {"name": state.name , "game": "soccer"}

def decide_play_dc(state: DataClassState) -> Literal["pickeball", "soccer"]:
    if random.random() < 0.5:
        return "pickeball"
    else:
        return "soccer"

# Building Graph with DataClasses
builder_dc = StateGraph(DataClassState)
builder_dc.add_node("playgame", play_game_dc)
builder_dc.add_node("pickeball", pickeball_dc)
builder_dc.add_node("soccer", soccer_dc)

builder_dc.add_edge(START, "playgame")
builder_dc.add_conditional_edges("playgame", decide_play_dc)
builder_dc.add_edge("pickeball", END)
builder_dc.add_edge("soccer", END)

graph_dc = builder_dc.compile()

print("========Using DataClasses =========\n")
# Invoke calls for Dataclass state
result_dc1 = graph_dc.invoke(DataClassState(name="PeterPan"))
print(result_dc1)
print("\n")

result_dc2 = graph_dc.invoke(DataClassState(name="TinkerBell"))
print(result_dc2)
print("\n")

result_dc3 = graph_dc.invoke(DataClassState(name=123))
print(result_dc3)