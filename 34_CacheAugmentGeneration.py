"""
================================================================================
This script introduces **Cache-Augmented Generation (CAG)**, a highly performant 
alternative to traditional RAG architectures. Instead of spending extra computation time 
searching, indexing, and reranking chunks from a database for every query, CAG 
takes advantage of ultra-long context window models. It pre-loads entire reference text 
bodies directly into the system context upfront, letting the model instantly query 
the information using internal attention cache frameworks.

[Image comparing classic RAG (vector database lookups) with Cache-Augmented Generation (direct preloaded prompt context)]

WHEN IS CAG RELEVANT?
CAG shines brightest when dealing with fixed, slow-moving document sets (such as 
product guides, company policies, or specific rulebooks). By omitting vector stores, 
it eliminates retrieval latency, bypasses bad indexing strategies, and ensures the 
entire document pool is fully considered by the model.

1. STATE DEFINITION: Outlines a simplified workflow structure tracking the question, 
   cached text buffer, and final target answer.
2. PREFIX INGESTION (`preload_context`): Simulates importing a full reference 
   knowledge base into the persistent graph memory structure.
3. GENERATION RUNTIME (`generate`): Binds the question directly to the preloaded 
   cache block, instructing an optimized LLM (`gpt-4o-mini`) to extract precise 
   insights without querying an external vector database.
================================================================================
"""

import os
from typing import  TypedDict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

# # Cache-Augmented Generation (CAG)
# 
# ### Concept
# Instead of searching a vector store for every query, CAG pre-loads the entire 
# relevant context into the LLM's prompt prefix or a specialized cache. 
# This is highly effective for long-context models when working with 
# a fixed set of documents.

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# ### 1. Define the State
class State(TypedDict):
    question: str
    cached_context: str
    answer: str

# ### 2. Define the Nodes
#NOTE - use gpt-4o-mini if below does not work 
llm = ChatOpenAI(model="gpt-4o-mini")

# In CAG, we usually have a "Pre-load" step
def preload_context(state: State):
    print("---LOADING CACHE---")
    # This simulates loading a large knowledge base into state
    knowledge_base = (
        "The 2024 Olympics were held in Paris. "
        "France won 16 gold medals. "
        "The closing ceremony took place at Stade de France."
    )
    return {"cached_context": knowledge_base}

# #### Node: Generate
def generate(state: State):
    print("---GENERATING---")
    prompt = ChatPromptTemplate.from_template(
        "You have access to a cached knowledge base. Answer the question accurately.\n\n"
        "Cache: {cache}\n\n"
        "Question: {question}"
    )
    chain = prompt | llm
    response = chain.invoke({
        "cache": state["cached_context"],
        "question": state["question"]
    })
    return {"answer": response.content}

# ### 3. Build the Graph
workflow = StateGraph(State)

workflow.add_node("preload", preload_context)
workflow.add_node("generate", generate)

workflow.add_edge(START, "preload")
workflow.add_edge("preload", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()

# ### 4. Run
inputs = {"question": "How many gold medals did France win in the 2024 Olympics?"}
result = app.invoke(inputs)
print(f"\nFinal Answer: {result['answer']}")