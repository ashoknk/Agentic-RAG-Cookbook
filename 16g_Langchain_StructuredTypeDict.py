# ### Structured output
# Models can be requested to provide their response in a format matching a given schema. 
# This is useful for ensuring the output can be easily parsed and used in subsequent processing. 
# LangChain supports multiple schema types and methods for enforcing structured output.

# ### TypedDict
# TypedDict is best when you want lightweight data structures native to Python 
#  without the overhead of Pydantic class instances.

import os
import json
from dotenv import load_dotenv
from typing_extensions import TypedDict
from langchain.chat_models import init_chat_model

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

## Initialize the Model
GROQ_MODEL = "groq:qwen/qwen3-32b"
model = init_chat_model(GROQ_MODEL)

## 1. Define the Lightweight Schema
class ActorDict(TypedDict):
    name: str
    role: str

class MovieDetailsDict(TypedDict):
    title: str
    year: int
    cast: list[ActorDict]
    genres: list[str]

## 2. Bind the structure to the model
structured_dict_llm = model.with_structured_output(MovieDetailsDict)

## 3. Invoke the query
movie_query = "Provide details about the movie The Shawshank Redemption (1994)"
print(f"Sending Query: '{movie_query}'...\n")
movie_dict = structured_dict_llm.invoke(movie_query)

## ==========================================
## SHOWCASING THE IMPORTANCE OF STRUCTURED DATA
## ==========================================
print("=" * 50)
print("⚡ VISUALIZING STRUCTURED DATA (Standard Dictionary)")
print("=" * 50)

# Accessing directly via standard dictionary keys
print(f"🎬 MOVIE: {movie_dict['title']} ({movie_dict['year']})")
print(f"🎭 GENRES: {', '.join(movie_dict['genres'])}")

# Generating a cleanly aligned terminal text table
print("\n📊 CAST LIST TABLE:")
print(f"{'Actor Name':<25} | {'Character Role':<25}")
print("-" * 55)
for actor in movie_dict["cast"]:
    print(f"{actor['name']:<25} | {actor['role']:<25}")
print("-" * 55)

# Showing that it is instantly a serializable python dict
print("\n📋 DIRECT DICTIONARY PRINT:")
print(movie_dict)