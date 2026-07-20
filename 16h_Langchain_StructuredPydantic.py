"""
================================================================================
This script introduces structured data extraction using **Pydantic (v2)** schemas 
integrated directly into a LangChain execution flow. It demonstrates how to coerce 
unstructured text generations from an LLM into reliable, nested data structures.

# ### Pydantic ### 
Pydantic models provide the richest feature set with field validation, descriptions, and nested structures.
Pydantic is ideal when you need deep nested schemas, default values, and strict runtime type checking.

WHY USE PYDANTIC IN PRODUCTION?
-------------------------------
While basic Python dictionaries act as simple key-value bags, Pydantic objects 
enforce strict, runtime type validation, default fallback variables, and comprehensive 
attribute tracking, making them ideal for enterprise API downstream consumption.

1. OBJECT Blueprinting: Defines structured schema layers (`Actor`, `MovieDetails`) 
   using Pydantic fields equipped with clear documentation descriptions.
2. SCHEMA COMPLIANCE ENFORCEMENT: Uses `.with_structured_output()` to bind the 
   target model , forcing the model to generate responses matching 
   the precise schema dimensions.
3. EXECUTION & PARSING: Invokes a request, revealing how attributes can be accessed 
   via object dot-notation or cleanly exported to standard JSON strings with a 
   single method call (`.model_dump()`).
================================================================================
"""

import os
import json
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

## Initialize the Model
# NOTE "groq:llama-3.1-8b-instant" does not work well for this code
# # GROQ_MODEL = "groq:qwen/qwen3.6-27b"
GROQ_MODEL = "groq:openai/gpt-oss-20b"
model = init_chat_model(GROQ_MODEL)

## 1. Define the Structured Schema (with nested structure)
class Actor(BaseModel):
    name: str
    role: str

class MovieDetails(BaseModel):
    title: str = Field(description="The formal title of the movie")
    year: int = Field(description="The release year")
    cast: list[Actor] = Field(description="List of core actors and their roles")
    genres: list[str]
    budget: float | None = Field(None, description="Budget in millions USD")

## 2. Bind the structure to the model
structured_llm = model.with_structured_output(MovieDetails)

## 3. Invoke the query
movie_query = "Provide details about the movie The Shawshank Redemption (1994)"
print(f"Sending Query: '{movie_query}'...\n")
movie_data = structured_llm.invoke(movie_query)

## ==========================================
## SHOWCASING THE IMPORTANCE OF STRUCTURED DATA
## ==========================================
# Exporting cleanly to valid JSON with one method call. 
# `model_dump`  Generate a dictionary representation of the model
# https://reference.langchain.com/python/langsmith/_openapi_client/_models/BaseModel/model_dump
# https://pydantic.dev/docs/validation/dev/concepts/serialization/
print("\n📋 CLEAN EXPORT TO JSON STRING:")
print(json.dumps(movie_data.model_dump(), indent=4))


print("=" * 50)
print("🏆 VISUALIZING STRUCTURED DATA (Pydantic Object)")
print("=" * 50)

# Accessing fields natively as Python objects
print(f"🎬 MOVIE: {movie_data.title.upper()} ({movie_data.year})")
print(f"💰 BUDGET: ${movie_data.budget} Million" if movie_data.budget else "💰 BUDGET: Unknown")
print(f"🎭 GENRES: {', '.join(movie_data.genres)}")

# Generating a cleanly aligned terminal text table using Pydantic dot-notation
print("\n📊 CAST LIST TABLE:")
print(f"{'Actor Name':<25} | {'Character Role':<25}")
print("-" * 55)
for actor in movie_data.cast:
    # Notice we use actor.name and actor.role here instead of actor['name']
    print(f"{actor.name:<25} | {actor.role:<25}")
print("-" * 55)