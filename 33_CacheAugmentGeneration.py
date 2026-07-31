"""
================================================================================
Cache-Augmented Generation (CAG) with API Prompt Caching
================================================================================
This script introduces Cache-Augmented Generation (CAG), a performant 
alternative to traditional RAG architectures. Instead of searching, indexing, 
and reranking chunks from a vector database for every query, CAG pre-loads entire 
reference text bodies directly into the system prompt context upfront.

AUTOMATIC PROVIDER PROMPT CACHING:
1. Minimum Threshold: OpenAI automatically caches prompt prefixes that are 
   1,024 tokens or longer.
2. Prefix Alignment: Static text (the knowledge base) is placed inside the 
   `SystemMessage` at the very top of the prompt. Variable user queries are 
   placed in the `HumanMessage` at the end.
3. Cost & Latency Savings: On second and subsequent queries, the provider API 
   reads the system prompt directly from GPU memory cache, offering ~50% cost 
   discount and up to 80% latency reduction.

GRAPH FLOW:
  START ---> [ preload_context ] ---> [ generate ] ---> END

OpenAI's automatic prompt caching (for prompts > 1,024 tokens) is enabled for all modern models (gpt-4o and newer), including:
✅ gpt-4o-mini (Supported)
✅ gpt-4o (Supported)
✅ o1-mini and o1-preview
================================================================================
"""

import os
from typing import TypedDict
from dotenv import load_dotenv
import time

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# ==============================================================================
# 1. STATE & MODEL SETUP
# ==============================================================================
class State(TypedDict):
    question: str
    cached_context: str
    answer: str

# Instantiate a single shared LLM instance
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ==============================================================================
# 2. GRAPH NODES
# ==============================================================================
def preload_context(state: State):
    """
    Pre-loads a large reference knowledge base.
    To trigger OpenAI's automatic prompt caching, the prompt prefix 
    must exceed 1,024 tokens.
    """
    print("---LOADING LARGE KNOWLEDGE BASE INTO PREFIX CACHE---")

    unique_run_id = f"[Run ID: {time.time()}]\n\n"
    
    # Large document body (repeated sentences ensure token length > 1,024)
    base_text = (
        "The 2024 Summer Olympics, officially the Games of the XXXIII(33) Olympiad, "
        "were held in Paris, France, from 26 July to 11 August 2024. "
        "Paris became the second city to host the Summer Olympics three times, "
        "having previously hosted in 1900 and 1924. "
        "France won 16 gold medals, 26 silver medals, and 22 bronze medals, "
        "totaling 64 medals. "
        "The closing ceremony took place at Stade de France on August 11, 2024. "

        "The United States and China tied for the most gold medals won with 40 gold medals each, "
        "while the United States topped the total medal tally with 126 total medals. "
        "A total of 329 medal events were held across 32 sports, featuring 10,500 athletes from 206 National Olympic Committees. "
        "Breaking (breakdancing) made its official debut as an optional Olympic sport, "
        "while skateboarding, sport climbing, and surfing returned to the programme after debuting in Tokyo 2020. "
        "Surfing events took place in Teahupo'o, Tahiti, setting a record for the farthest venue from a host city. "
    )
    
    # Expand context size to satisfy OpenAI's >1,024 token requirement
    # large_knowledge_base = unique_run_id + (base_text * 40)
    large_knowledge_base =  (base_text * 40)
    return {"cached_context": large_knowledge_base}


def generate(state: State):
    """
    Places the static context in the System Message. 
    This allows OpenAI to hash and cache the exact prompt prefix automatically.
    """
    print("---GENERATING RESPONSE VIA CACHED PROMPT PREFIX---")
    
    # IMPORTANT: Static system prompt MUST come first for prefix matching
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a specialized assistant. Use the pre-loaded knowledge base to answer questions accurately:\n\n{cache}"),
        ("human", "{question}")
    ])
    
    chain = prompt | llm
    
    # Call the model
    response = chain.invoke({
        "cache": state["cached_context"],
        "question": state["question"]
    })
    
    # Extract token metrics to prove Prompt Caching execution to learners
    # https://docs.python.org/3/library/functions.html#getattr
    # Safely extracts token metrics from the LLM's raw response objects (`response.usage_metadata` and `response.response_metadata['token_usage']`).
    # It navigates nested dictionaries to retrieve `cached_tokens` (tokens served from API prompt cache) and `total_input_tokens` (total prompt tokens sent).
    # Using `.get()` and `getattr()` with fallback defaults ensures the code handles missing keys cleanly across different provider formats.

    usage = getattr(response, "usage_metadata", {})
    response_metadata = getattr(response, "response_metadata", {})
    token_usage = response_metadata.get("token_usage", {})
    prompt_tokens_details = token_usage.get("prompt_tokens_details", {})
    
    cached_tokens = prompt_tokens_details.get("cached_tokens", 0)
    total_input_tokens = usage.get("input_tokens", token_usage.get("prompt_tokens", 0))
    
    print(f"📊 [Token Metrics] Total Input: {total_input_tokens} | Cached Tokens: {cached_tokens}")
    if cached_tokens > 0:
        print("⚡ PROMPT CACHE HIT! The LLM re-used GPU memory prefix.")
    else:
        print("💾 PROMPT CACHE MISS/WRITE! The provider cached this prefix for future queries.")
        
    return {"answer": response.content}

# ==============================================================================
# 3. BUILD GRAPH
# ==============================================================================
workflow = StateGraph(State)

workflow.add_node("preload", preload_context)
workflow.add_node("generate", generate)

workflow.add_edge(START, "preload")
workflow.add_edge("preload", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()

# Save the file as a PNG
OUTPUT_IMAGE_PATH = "Image_PNGs/33_CacheAugmentGeneration.png"
app.get_graph().draw_mermaid_png(output_file_path=OUTPUT_IMAGE_PATH)    
# Automatically display/open the image on macOS NOTE _Just for testing purposes
os.system(f"open {OUTPUT_IMAGE_PATH}")


# ==============================================================================
# 4. RUN DEMONSTRATION
# ==============================================================================
if __name__ == "__main__":
    # On Query 1, the LLM provider (OpenAI) has never seen this exact system prompt prefix before.
    # Provider stored the X tokens in memory. (Cache Miss / Write)
    print("\n======================================================================")
    print("🔥 QUERY 1: Initial Invocation (Populates Provider Cache)")
    print("======================================================================")
    res1 = app.invoke({"question": "Which country won the most overall medals in 2024?"})
    print(f"Bot Answer: {res1['answer']}\n")

    # Provider read the X tokens directly from memory. (Cache Hit)
    print("======================================================================")
    print("🔥 QUERY 2: Follow-up Query (Re-uses Cached Prefix)")
    print("======================================================================")
    res2 = app.invoke({"question": "Where and when was the closing ceremony held?"})
    print(f"Bot Answer: {res2['answer']}")