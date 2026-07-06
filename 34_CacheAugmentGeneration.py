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