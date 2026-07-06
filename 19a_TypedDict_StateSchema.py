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

class TypedDictState(TypedDict):
    name: str
    game: Literal["pickeball", "soccer"]

def play_game_typed(state: TypedDictState):
    print("---Play Game node has been called--")
    return {"name": state['name'] }

def pickeball_typed(state: TypedDictState):
    print("-- pickeball node has been called--")
    return {"name": state["name"] , "game": "pickeball"}

def soccer_typed(state: TypedDictState):
    print("-- soccer node has been called--")
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

print("========Using TypedDict =========\n")
# Invoke calls with string 
result1 = graph_typed.invoke({"name": "Peter"})
print(result1)

result2 = graph_typed.invoke({"name": "Peter"})
print(result2)

# Invoke calls with int
result3 = graph_typed.invoke({"name": 123})
print(result3)

