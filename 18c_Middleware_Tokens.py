"""
================================================================================
CONCEPT 1: CONTEXT MEMORY & TOKEN MANAGEMENT
================================================================================
This script demonstrates how to use `SummarizationMiddleware` to manage an agent's 
memory during long-lived chat cycles. As multi-turn dialogues grow, they risk 
exceeding the model's maximum context window, causing memory loss or high costs.

This code showcases three automated compression strategies to optimize context size:
1. MESSAGE COUNT BOUNDARY: Triggers summarization when a specific number of 
   messages is accumulated, keeping a designated few active.
2. RAW TOKEN BOUNDARY: Computes character length limits and forces compression 
   once a precise token threshold is reached.
3. CONTEXT WINDOW FRACTION BOUNDARY: Monitors memory utilization as a percentage of the 
   model's total context limit, optimizing storage proportionally.

   InMemorySaver - 
        This checkpoint saver stores checkpoints in memory using a defaultdict.
   SummarizationMiddleware -
        Summarizes conversation history when token limits are approached.
        This middleware monitors message token counts and automatically summarizes older messages when a threshold is reached, 
        preserving recent messages and maintaining context continuity by ensuring AI/Tool message pairs remain together.

Significance: This prevents agent "amnesia" and keeps operating overhead low by 
automatically compressing historical logs while retaining vital conversation context.
================================================================================
"""

import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Choose Model
GPT_MODEL_SUMMARIZER = "groq:openai/gpt-oss-20b"

# ------------------------------------------------------------------------------
# 1. MESSAGE-BASED SUMMARIZATION STRATEGY
# ------------------------------------------------------------------------------
print("\n--- Running Strategy 1: Message-Based Threshold ---")

# https://reference.langchain.com/python/langgraph.checkpoint/memory/InMemorySaver
# https://reference.langchain.com/python/langchain/agents/middleware/summarization/SummarizationMiddleware
agent_summarizer = create_agent(
    model=GPT_MODEL_SUMMARIZER,
    checkpointer=InMemorySaver(),
    middleware=[
        SummarizationMiddleware(
            model=GPT_MODEL_SUMMARIZER,
            trigger=("messages", 10),  # Compress when history reaches 10 messages
            keep=("messages", 4)       # Keep the 4 most recent messages uncompressed
        )
    ]
)

config = {"configurable": {"thread_id": "message-limit"}}
questions = [
    "What is 2+2?",
    "What is 10*5?",
    "What is 100/4?",
    "What is 15-7?",
    "What is 3*3?",
    "What is 4*4?",
]

for q in questions:
    response = agent_summarizer.invoke({"messages": [HumanMessage(content=q)]}, config)
    messages = response["messages"]
    
    print(f"\nTotal Messages Count: {len(messages)}")
    
    # NOTE Just for debugging 
    for i, msg in enumerate(messages):
        # Identify message type
        msg_type = type(msg).__name__
        
        # Check if the content is a summary or regular message
        content_preview = msg.content
        if len(content_preview) > 80:
            content_preview = content_preview[:80] + "..."
            
        print(f"  [{i}] {msg_type}: {content_preview}")



# ------------------------------------------------------------------------------
# 2. TOKEN-BASED SUMMARIZATION STRATEGY
# ------------------------------------------------------------------------------
print("\n--- Running Strategy 2: Raw Token Limits ---")

@tool
def search_hotels(city: str) -> str:
    """Search hotels - returns long response to use more tokens."""
    return f"""Hotels in {city}:
    1. Grand Hotel - 5 star, $350/night, spa, pool, gym
    2. City Inn - 4 star, $180/night, business center
    3. Budget Stay - 3 star, $75/night, free wifi"""

agent_token_limit = create_agent(
    model=GPT_MODEL_SUMMARIZER,
    tools=[search_hotels],
    checkpointer=InMemorySaver(),
    middleware=[
        SummarizationMiddleware(
            model=GPT_MODEL_SUMMARIZER,
            trigger=("tokens", 550),   # Compress when exceeding 550 tokens
            keep=("tokens", 200),      # Compress down to 200 tokens
        ),
    ]
)

config_tokens = {"configurable": {"thread_id": "token-limit"}}

def count_tokens(messages):
    total_chars = sum(len(str(m.content)) for m in messages)
    return total_chars // 4  # 4 chars ≈ 1 token

cities = ["Paris", "London", "Tokyo", "New York", "San Francisco", "Seoul"]
for city in cities:
    response_token = agent_token_limit.invoke(
        {"messages": [HumanMessage(content=f"Find hotels in {city}")]},
        config=config_tokens
    )
    tokens = count_tokens(response_token["messages"])
    print(f"{city}: ~{tokens} tokens, {len(response_token['messages'])} messages")



# ------------------------------------------------------------------------------
# 3. FRACTION-BASED SUMMARIZATION STRATEGY
# ------------------------------------------------------------------------------
print("\n--- Running Strategy 3: Context Window Fractions ---")

@tool
def search_hotels_short(city: str) -> str:
    """Search hotels."""
    return f"Hotels in {city}: Grand Hotel $350, City Inn $180, Budget Stay $75"

agent_fraction = create_agent(
    model=GPT_MODEL_SUMMARIZER,
    tools=[search_hotels_short],
    checkpointer=InMemorySaver(),
    middleware=[
        SummarizationMiddleware(
            model=GPT_MODEL_SUMMARIZER,
            trigger=("fraction", 0.005),  # 0.5% of total context window capacity (640-token is 0.5%)
            keep=("fraction", 0.003),     # Compress down to 0.3% context usage
        ),
    ],
)

config_fraction = {"configurable": {"thread_id": "fraction-limit"}}

for city in cities:
    response_frac = agent_fraction.invoke(
        {"messages": [HumanMessage(content=f"Hotels in {city}")]},
        config=config_fraction
    )
    tokens_frac = count_tokens(response_frac["messages"])
    fraction_calc = tokens_frac / 128000  # relative fraction of a 128k window
    # Math: 128,000 tokens x 0.005 = 640 tokens
    # Math: 128,000 tokens x 0.003 = 384 tokens
    print(f"{city}: ~{tokens_frac} tokens ({fraction_calc:.4%}), {len(response_frac['messages'])} msgs")