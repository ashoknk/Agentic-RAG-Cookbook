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
from typing import Literal
from typing_extensions import TypedDict
from dotenv import load_dotenv

# LangChain and LangGraph Imports
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display

# ### State Schema With DataClasses
# When we define a LangGraph StateGraph, we use a state schema.
# The state schema represents the structure and types of data that our graph will use.
# All nodes are expected to communicate with that schema.
# LangGraph offers flexibility in how you define your state schema, 
# accommodating various Python types and validation approaches!

# we can use the TypedDict class from python's typing module.
# It allows you to specify keys and their corresponding value types.
# But, note that these are type hints.
# They can be used by static type checkers (like mypy) or IDEs to catch potential 
# type-related errors before the code is run.
# But they are not enforced at runtime!

# ### ========= Dataclasses ========= ###
# Python's dataclasses provide another way to define structured data.
# Dataclasses offer a concise syntax for creating classes that are primarily 
# used to store data.

@dataclass
class DataClassState:
    name: str
    game: Literal["soccer", "pickeball"]

def play_game_dc(state: DataClassState):
    print("---Play Game node has been called--")
    return {"name": state.name }

def pickeball_dc(state: DataClassState):
    print("-- pickeball node has been called--")
    return {"name": state.name , "game": "pickeball"}

def soccer_dc(state: DataClassState):
    print("-- soccer node has been called--")
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
result_dc1 = graph_dc.invoke(DataClassState(name="Peter", game="pickeball"))
print(result_dc1)

result_dc2 = graph_dc.invoke(DataClassState(name="Jeniffer", game="soccer"))
print(result_dc2)

result_dc3 = graph_dc.invoke(DataClassState(name=123, game="soccer"))
print(result_dc3)