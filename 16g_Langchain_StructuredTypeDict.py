"""
================================================================================
This script focuses on enforcing reliable data formats from unstructured text generation, 
leveraging standard Python **`TypedDict` schemas** combined with model-level constraints. 
This approach guarantees that the LLM's response can be reliably converted into 
predictable data structures, avoiding the parsing errors common with raw string manipulation.

# ### TypedDict###
TypedDict is best when you want lightweight data structures native to Python 
without the overhead of Pydantic class instances.
Developers often prefer TypedDict here because it returns a standard, native Python dictionary ({...}) 
instead of a custom Pydantic object instance, 
making it incredibly easy to serialize or pass directly into other functions.

1. TYPED DATA DEFINITION: Declares nested structured layouts using `TypedDict` models 
   to specify nested string arrays, integers, and objects.
2. ENFORCEMENT BINDING (`with_structured_output`):  Bind the structure to the model. Modifies the model's output routing. 
   This forces the model to leverage native tool-calling features to guarantee compliance with the requested schema.
3. PARSING DATA: Submits a movie metadata question and receives an instant, native Python dictionary object.
4. PRODUCTION UTILIZATION: Iterates over the structured return value to render 
   a text-aligned terminal table, demonstrating database readiness.
================================================================================
"""

import os
import json
from dotenv import load_dotenv
from typing_extensions import TypedDict
from langchain.chat_models import init_chat_model

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

## Initialize the Model
# GROQ_MODEL = "groq:qwen/qwen3.6-27b"
GROQ_MODEL = "groq:openai/gpt-oss-20b"
# NOTE "groq:llama-3.1-8b-instant" does not work well for this code
model = init_chat_model(GROQ_MODEL)

## 1. Define the Lightweight Schema
# https://typing.python.org/en/latest/spec/typeddict.html
class ActorDict(TypedDict):
    name: str
    role: str

class MovieDetailsDict(TypedDict):
    title: str
    year: int
    cast: list[ActorDict]
    genres: list[str]

## 2. Bind the structure to the model
# https://reference.langchain.com/python/langchain-groq/chat_models/ChatGroq/with_structured_output
structured_dict_llm = model.with_structured_output(MovieDetailsDict)


## 3. Invoke the query
movie_query = "Provide details about the movie The Shawshank Redemption (1994)"
print(f"Sending Query: '{movie_query}'...\n")
movie_dict = structured_dict_llm.invoke(movie_query)

## ==========================================
## SHOWCASING THE IMPORTANCE OF STRUCTURED DATA
## ==========================================
# Showing that it is instantly a serializable python dict. Demonstrating database readiness.
print("\n📋 DIRECT DICTIONARY PRINT:")
print(movie_dict)
# exit()

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

