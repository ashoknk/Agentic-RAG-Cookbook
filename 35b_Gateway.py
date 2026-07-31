"""
===LLM Gateway using LiteLLM==

This file builds directly on last file by scaling up to 
LiteLLM's advanced Router component. It introduces beginner learners to 
abstract alias mapping, 
load balancing rotation strategies: 
    - simple shuffle, 
    - least-busy, 
    - latency-based routing     
and custom observability callbacks  for production auditing.
https://docs.litellm.ai/docs/#litellm-python-sdk
"""


import os
import time
import json
import warnings
import logging
import asyncio
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)

from litellm import completion, completion_cost, Router
import litellm
from collections import Counter

litellm.suppress_debug_info = True
load_dotenv()

# Define standard model mapping variables
MODEL_GPT = "gpt-4o"
MODEL_GPT_MINI = "gpt-4o-mini"
MODEL_CLAUDE = "anthropic/claude-3-5-haiku-20241022"
MODEL_GEMINI = "gemini/gemini-2.5-flash"
MODEL_GROQ = "groq/llama-3.3-70b-versatile"

# ========================================
# Step 1 Alias Mapping : Implement Smart Routing via abstract alias names - model_list_routing, router
model_list_routing = [
    {
        "model_name": "fast-cheap",
        "litellm_params": {
            "model": MODEL_GROQ,
            "api_key": os.getenv("GROQ_API_KEY")
        }
    },
    {
        "model_name": "smart-coding",
        "litellm_params": {
            "model": MODEL_GPT,
            "api_key": os.getenv("OPENAI_API_KEY")
        }
    },
    {
        "model_name": "balanced",
        "litellm_params": {
            "model": MODEL_GPT_MINI,
            "api_key": os.getenv("OPENAI_API_KEY")
        }
    }
]

router = Router(model_list=model_list_routing)
fast_response = router.completion(
    model="fast-cheap",
    messages=[{"role": "user", "content": "Summarize: AI is changing software."}]
)
code_response = router.completion(
    model="smart-coding",
    messages=[{"role": "user", "content": "Write a Python function to reverse a string."}]
)

print("⚡ Fast/cheap (Groq): ", fast_response.choices[0].message.content[:150])
print("\n🧠 Smart/coding (GPT-4o):\n", code_response.choices[0].message.content[:300])
print("-" * 50)


# ========================================
# Step 2 Simple Shuffle:Configure Load Balancing pools across multiple provider keys - model_pool_list,shuffle_router
model_pool_list = [
    {
        "model_name": "gpt-pool",
        "litellm_params": {
            "model": MODEL_GPT,
            "api_key": os.getenv("OPENAI_API_KEY"),
        },
        "model_info": {"id": "openai-gpt4o"}
    },
    {
        "model_name": "gpt-pool",
        "litellm_params": {
            "model": MODEL_GROQ,
            "api_key": os.getenv("GROQ_API_KEY"),
        },
        "model_info": {"id": "groq-llama-70b"}
    },
]

# load balancing rotation strategies - simple shuffle
shuffle_router = Router(
    model_list=model_pool_list,
    routing_strategy="simple-shuffle"
)

print(f"\n{'Request':<10}{'Deployment Picked':<22}{'Latency':<12}{'Response':<40}")
print("-" * 84)

for i in range(6):
    r = shuffle_router.completion(
        model="gpt-pool",
        messages=[{"role": "user", "content": f"Say hello, request {i+1}"}]
    )
    deployment_id = r._hidden_params.get("model_id", "unknown")
    latency = r._response_ms
    answer = r.choices[0].message.content[:35]
    print(f"#{i+1:<9}{deployment_id:<22}{latency:>6.0f} ms   {answer}")

print("-" * 50)


# ========================================
# Step 3 Least Busy : Execute Least-Busy Routing strategy - least_busy_list, lb_router
least_busy_list = [
    {"model_name": "chat", "litellm_params": {"model": MODEL_GPT_MINI, "api_key": os.getenv("OPENAI_API_KEY")}, "model_info": {"id": "🔵 OpenAI"}},
    {"model_name": "chat", "litellm_params": {"model": MODEL_GROQ, "api_key": os.getenv("GROQ_API_KEY")}, "model_info": {"id": "🟢 Groq"}},
]

# load balancing rotation strategies - least-busy
lb_router = Router(model_list=least_busy_list, routing_strategy="least-busy")
# acts like a dictionary that defaults missing key values to 0
hits = Counter()

# LiteLLM attaches internal metadata to the response ex.model_id
for i in range(8):
    r = lb_router.completion(
        model="chat",
        messages=[{"role": "user", "content": f"Say 'OK' #{i}"}],
        max_tokens=5
    )
    # Extract the model_id once into a variable
    model_id = r._hidden_params.get("model_id", "?")
    hits[model_id] += 1
    print(f"Request {i+1} → {model_id}")

print("\n🎯 Distribution:")
for k, v in hits.most_common():
    print(f"   {k}: {'█' * v} ({v})")
print("-" * 50)

# ========================================
# Step 4 Latency Based Routing: Implement Observability callbacks to capture audit logs for every call
# load balancing rotation strategies - latency-based routing, and cost-based patterns

call_logs = []

def log_success(kwargs, completion_response, start_time, end_time):
    call_logs.append({
        "model": kwargs.get("model"),
        "prompt": kwargs["messages"][-1]["content"][:60],
        "input_tokens": completion_response.usage.prompt_tokens,
        "output_tokens": completion_response.usage.completion_tokens,
        "latency_sec": round((end_time - start_time).total_seconds(), 2),
        "cost_usd": kwargs.get("response_cost", 0),
        "user": kwargs.get("user", "anonymous")
    })

def log_failure(kwargs, completion_response, start_time, end_time):
    print("❌ Call failed:", kwargs.get("exception"))

litellm.success_callback = [log_success]
litellm.failure_callback = [log_failure]

for q, user in [
    ("What is RAG?", "Peter Pan"),
    ("Explain transformers.", "Captain Hook"),
    ("What is fine-tuning?", "Tinker Bell"),
]:
    completion(
        model=MODEL_GPT_MINI,
        messages=[{"role": "user", "content": q}],
        user=user
    )

print("Audit Log Captured:")
print(json.dumps(call_logs, indent=2, default=str))
print("-" * 50)