"""
This file serves as the foundational introduction to an LLM Gateway using LiteLLM. 
https://docs.litellm.ai/docs/#litellm-python-sdk

It builds upon the core concept of unified APIs by showing beginners how to make 
standard requests across different providers, handle unexpected provider outages using 
1.automated fallbacks, 
2.track per-call token costs, and 
3.implement local caching.
"""

import os
from dotenv import load_dotenv
import warnings
import logging
import time

# Step 2: Configure warning filters and logging suppression for clean output
warnings.filterwarnings("ignore")
logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)
logging.getLogger("LiteLLM Router").setLevel(logging.CRITICAL)
logging.getLogger("LiteLLM Proxy").setLevel(logging.CRITICAL)

from litellm import completion, completion_cost
import litellm
from litellm.caching import Cache
https://docs.litellm.ai/docs/caching/local_caching

litellm.suppress_debug_info = True
litellm.set_verbose = False
litellm.turn_off_message_logging = True
litellm.telemetry = False
litellm.callbacks = []

# Load environment variables and define standard model variables
load_dotenv()
MODEL_GPT = "gpt-4o"
MODEL_GPT_MINI = "gpt-4o-mini"
MODEL_CLAUDE = "anthropic/claude-haiku-4-5-20251001"
MODEL_GEMINI = "gemini/gemini-2.5-flash"
MODEL_GROQ = "groq/llama-3.3-70b-versatile"

# Step 1: Perform a quick check on loaded API keys
print("OpenAI key loaded:    ", "✅" if os.getenv("OPENAI_API_KEY") else "❌")
print("Anthropic key loaded: ", "✅" if os.getenv("ANTHROPIC_API_KEY") else "❌")
print("Gemini key loaded:    ", "✅" if os.getenv("GOOGLE_API_KEY") else "❌")
print("Groq key loaded:      ", "✅" if os.getenv("GROQ_API_KEY") else "❌")
print("-" * 50)

# Step 2: Demonstrate the unified API structure across multiple providers
prompt = "Explain RAG in one sentence."
providers = [
    ("🔵 OpenAI",     MODEL_GPT),
    ("🔵 OpenAI",     MODEL_GPT_MINI),
    ("🟣 Anthropic",  MODEL_CLAUDE),
    ("🟡 Gemini",     MODEL_GEMINI),
    ("🟢 Groq",       MODEL_GROQ),
]

# Step 3: Execute calls across all defined model providers in ONE unified loop
# The model (completions) can generate multiple candidate responses. choices[0] is first option
# choices[0] is a structured JSON payload
for label, model in providers:
    try:
        r = completion(model=model, messages=[{"role": "user", "content": prompt}])
        print(f"{label:<15}: {r.choices[0].message.content[:80]}")
    except Exception as e:
        print(f"{label:<15}: ❌ {type(e).__name__}")
print("-" * 50)



# Step 4: Configure automated fallbacks to handle primary model provider outages
response = completion(
    model=MODEL_GEMINI,
    messages=[{"role": "user", "content": "What is an LLM Gateway?"}],
    fallbacks=[
        MODEL_GPT_MINI,
        MODEL_GROQ
    ]
)
print("\nResponse:", response.choices[0].message.content[:200], "...")
print("\nWhich model actually answered?", response.model)
print("-" * 50)


# Step 5: Force an intentional primary failure to watch fallback rescue execution
response = completion(
    model="openai/fake-nonexistent-model-9999",     # Will fail intentionally
    messages=[{"role": "user", "content": "What is an LLM Gateway?"}],
    fallbacks=[
        MODEL_GPT_MINI,                              # Backup 1
        MODEL_GROQ                                   # Backup 2
    ]
)
print("✅ App still got a response, even though the primary failed!")
print(f"\n🤖 Model that actually answered: {response.model}")
print(f"\n📝 Response: {response.choices[0].message.content[:200]}...")
print("-" * 50)


# Step 6: Track and compute precise USD costs via built-in pricing tables
response = completion(
    model=MODEL_GPT_MINI,
    messages=[{"role": "user", "content": "Write a haiku about AI."}]
)
cost = completion_cost(completion_response=response)

print("Response:    ", response.choices[0].message.content)
print("\nInput tokens: ", response.usage.prompt_tokens)
print("Output tokens:", response.usage.completion_tokens)
print(f"Cost:         ${cost:.8f}")
print("-" * 50)


# Step 7: Enable in-memory local caching to optimize duplicate queries
litellm.callbacks = []
litellm.success_callback = []
litellm.failure_callback = []
litellm._async_success_callback = []
litellm._async_failure_callback = []
litellm.cache = None

print("✅ LiteLLM state reset — ready for clean caching demo")
litellm.cache = Cache(type="local")

cache_prompt = "What does LLM stand for? Answer in one line."

start = time.time()
r1 = completion(
    model=MODEL_GPT_MINI,
    messages=[{"role": "user", "content": cache_prompt}],
    caching=True
)
t1 = time.time() - start
print(f"❄️  First call (API):   {t1:.2f}s — {r1.choices[0].message.content}")

start = time.time()
r2 = completion(
    model=MODEL_GPT_MINI,
    messages=[{"role": "user", "content": cache_prompt}],
    caching=True
)
t2 = time.time() - start
print(f"⚡ Second call (cache): {t2:.4f}s — {r2.choices[0].message.content}")
print(f"\n🚀 Speedup: {t1/t2:.1f}x faster, and ZERO cost on the second call!")
print("-" * 50)