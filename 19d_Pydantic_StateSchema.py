import os
import logging
import warnings
import random
from typing import Literal
from dotenv import load_dotenv

# Pydantic Import
from pydantic import BaseModel

# LangChain and LangGraph Imports
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END

# Suppress warnings and logs
warnings.filterwarnings("ignore")

# ### ========= Pydantic BaseModel State ========= ###
# Pydantic's BaseModel provides structured data with strict runtime type enforcement.
# It automatically coerces and validates data entering or moving through the graph.

class State(BaseModel):
    name: str
    game: Literal["soccer", "pickeball"] = "soccer"  # Default fallback values allowed

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
result_pd1 = graph_pydantic.invoke(State(name="Peter", game="pickeball"))
print(f"Result 1:\n{result_pd1}\n")

result_pd2 = graph_pydantic.invoke(State(name="Jeniffer", game="soccer"))
print(f"Result 2:\n{result_pd2}\n")

# CRITICAL PRO-TIP DEMONSTRATION: Runtime Coercion vs Error
# NOTE Try to change "123" to 123 . Also try str(123)
# Pydantic catches 'name=123' and safely casts it to a string ("123") before execution!
result_pd3 = graph_pydantic.invoke(State(name="123", game="soccer"))
print(f"Result 3 (Auto-Coerced Integer):")
print(result_pd3)
print(f"Verified Type of 'name' parameter: {type(result_pd3['name'])}")