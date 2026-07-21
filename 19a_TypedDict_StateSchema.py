"""
================================================================================
This script evaluates Python's native **`TypedDict`** class as the state schema 
layer of a LangGraph workflow. It teaches students how state dictionaries act as 
type hints for development environments, but highlights their lack of strict data 
validation at runtime.

1. STATE DEFINITION: Declares a `TypedDictState` containing a user's name string 
   and a strict game option constraint (`Literal["pickeball", "soccer"]`).
2. WORKFLOW ROUTING DESIGN: Connects an initialization play node to secondary nodes 
   via a 50/50 randomized conditional branching router (`decide_play_typed`).
3. VALID STATE EXECUTION: Invokes the compiled state workflow with valid user string 
   arguments, verifying successful execution and routing behavior.
4. RUNTIME LIMITATION PROOF: Invokes the graph passing an integer type into the 
   `name` field. The script successfully runs without crashing, demonstrating 
   that `TypedDict` type constraints are *not* strictly enforced at runtime.
================================================================================
"""

import os
import random

from typing import Literal
from typing_extensions import TypedDict
# https://reference.langchain.com/python/langchain-groq/chat_models/ChatGroq/with_structured_output

# LangChain and LangGraph Imports
from langgraph.graph import StateGraph, START, END

# ### State Schema With DataClasses
# When initializing the graph with builder = StateGraph(TypedDictState), 
# you are telling LangGraph what key-value pairs (name and game) your graph state 
# will hold across nodes. All nodes are expected to communicate with that schema.
# we can use the TypedDict class from python's typing module.

# But, note that these are type hints.
# They can be used by static type checkers (like mypy) or IDEs to catch potential 
# type-related errors before the code is run. But they are not enforced at runtime!

class TypedDictState(TypedDict):
    name: str
    game: Literal["pickeball", "soccer"]

def play_game_typed(state: TypedDictState):
    print("---Play Game node has been called--")
    return {"name": state['name'] }

def pickeball_typed(state: TypedDictState):
    print("-- Pickeball node has been called--")
    return {"name": state["name"] , "game": "pickeball"}

def soccer_typed(state: TypedDictState):
    print("-- Soccer node has been called--")
    return {"name": state["name"] , "game": "soccer"}

def decide_play_typed(state: TypedDictState) -> Literal["pickeball", "soccer"]:
    # Here, let's just do a 50 / 50 split between nodes 2, 3
    if random.random() < 0.5:
        return "pickeball"
    else:
        return "soccer"

# Building Graph with TypedDict
builder = StateGraph(TypedDictState)
builder.add_node("playgame", play_game_typed)
builder.add_node("pickeball", pickeball_typed)
builder.add_node("soccer", soccer_typed)

builder.add_edge(START, "playgame")
builder.add_conditional_edges("playgame", decide_play_typed)
builder.add_edge("pickeball", END)
builder.add_edge("soccer", END)

graph_typed = builder.compile()

# TypedDict cleanly defines the shared schema structure across playgame, pickeball, and soccer nodes,
# allowing LangGraph to merge dictionary updates automatically 
# (e.g., merging {'name': 'PeterPan'} with {'game': 'soccer'}).

print("========Using TypedDict =========\n")
# Invoke calls with string 
result1 = graph_typed.invoke({"name": "PeterPan"})
print(result1)
print("\n")

result2 = graph_typed.invoke({"name": "TinkerBell"})
print(result2)
print("\n")

# Invoke calls with int
result3 = graph_typed.invoke({"name": 123})
print(result3)


# Notice how {'name': 123} passes through smoothly. This proves that if you require strict runtime
# type enforcement (e.g., throwing an error when an integer 123 is passed for a str), 
# you would need to use a Pydantic BaseModel  with runtime validation instead of TypedDict