"""
#####  Chain Of Thought RAG (COT RAG) ##### 

### Problem with standard RAG
Standard RAG takes a query and retrieves documents, then generates an answer. However, if the query is complex (multi-step), 
standard RAG might fail because it tries to answer everything in one go without a plan.

CoT reasoning breaks down a complex question into intermediate steps, and allows retrieval + reflection at each step before answering.

User Query
   ↓
- Step 1: Decompose question → sub-steps (Reason)
- Step 2: Retrieve docs per step (Act)
- Step 3: Combine context (Observe)
- Step 4: Final answer generation (Reflect)
"""

import os
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from typing import List, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
# TODO from langchain_tavily import TavilySearch
#tavily_tool = TavilySearch(k=2)

from langgraph.graph import StateGraph, START, END

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

# ### 1. Define the State
# The state will store the original question, the generated plan, retrieved contexts, and the final answer.
class State(TypedDict):
    question: str
    plan: str
    context: List[str]
    answer: str

# ### 2. Define the Nodes
# We need three main nodes:
# 1.  **Planner**: To break down the question into steps.
# 2.  **Retriever**: To fetch information for those steps.
# 3.  **Generator**: To produce the final answer based on the plan and context.

#NOTE - use gpt-4o-mini if below does not work 
llm = ChatOpenAI(model="gpt-4o-mini")
search_tool = TavilySearchResults(k=2)

# #### Node 1: Planner ( Output is a clean, single-line search string)
def planner(state: State):
    print("---PLANNER---")
    prompt = ChatPromptTemplate.from_template(
        "You are an expert planner. Break down the following complex question into 2-3 search concepts, "
        "but output ONLY a single optimized keyword search string containing those key terms combined. "
        "Do not include headers, numbering, or introductory text. "
        "Question: {question}"
    )
    chain = prompt | llm
    result = chain.invoke({"question": state["question"]})
    return {"plan": result.content.strip()}

# #### Node 2: Retriever 
def retriever(state: State):
    print("---RETRIEVER---")
    search_query = state["plan"]
    print(f"Sending formatted query to Tavily: '{search_query}'")
    
    try:
        search_results = search_tool.invoke(search_query)
        print(f"Search results type: {type(search_results)}") # Debugging check
        
        # Defensive Check: If Tavily fails or returns a string error body instead of a list of dictionaries
        if isinstance(search_results, str) or not isinstance(search_results, list):
            print(f"⚠️ Tavily did not return a valid list. Response: {search_results}")
            return {"context": [f"Could not retrieve live search data. Reason: {search_results}"]}
            
        contexts = [res["content"] for res in search_results if isinstance(res, dict) and "content" in res]
        return {"context": contexts}
        
    except Exception as e:
        print(f"💥 Critical error during search API invocation: {e}")
        return {"context": [f"Search failure due to exception: {str(e)}"]}

# #### Node 3: Generator
def generator(state: State):
    print("---GENERATOR---")
    prompt = ChatPromptTemplate.from_template(
        "Based on the following plan and retrieved context, provide a comprehensive answer to the question.\n"
        "Question: {question}\n"
        "Plan: {plan}\n"
        "Context: {context}"
    )
    chain = prompt | llm
    result = chain.invoke({
        "question": state["question"],
        "plan": state["plan"],
        "context": "\n".join(state["context"])
    })
    return {"answer": result.content}

# ### 3. Build the Graph



workflow = StateGraph(State)

# Add nodes
workflow.add_node("planner", planner)
workflow.add_node("retriever", retriever)
workflow.add_node("generator", generator)

# Define edges
workflow.add_edge(START, "planner")
workflow.add_edge("planner", "retriever")
workflow.add_edge("retriever", "generator")
workflow.add_edge("generator", END)

app = workflow.compile()

# Save the file as a PNG
OUTPUT_IMAGE_PATH = "Image_PNGs/COTRag.png"
app.get_graph().draw_mermaid_png(output_file_path=OUTPUT_IMAGE_PATH)    
# Automatically display/open the image on macOS NOTE _Just for testing purposes
# os.system(f"open {OUTPUT_IMAGE_PATH}")

# ### 4. Run the Graph

inputs = {"question": "Who is the current CEO of the company that acquired Figma, and what was their revenue in the last fiscal year?"}
# NOTE: The Adobe/Figma deal was terminated, but this test query explores multi-step reasoning.

for output in app.stream(inputs):
    for key, value in output.items():
        print(f"Finished executing: {key}")

print("\n---FINAL ANSWER---")
print(output["generator"]["answer"])