"""
================================================================================
This script serves as a technical showcase of optimization strategies within LangChain, 
specifically focusing on the performance and design tradeoffs between real-time token 
**Streaming** and parallelized asynchronous **Batch Processing**.

THE RUNTIME PATTERNS:
---------------------
- **Streaming (`.stream()`)**: Sets up an iterative generator that yields individual text 
  tokens piece-by-piece as they are processed by the LLM, reducing perceived latency.
- **Batching (`.batch()`)**: Aggregates a compilation of independent text requests 
  and submits them concurrently to the provider API, maximizing throughput.


1. SYNCHRONOUS BASELINE: Measures the traditional block time of a standard invoke call.
2. TOKEN GENERATION LOOP: Uses a streaming block to unpack and stream real-time responses.
3. PARALLELIZED POOL CONCURRENCY: Dispatches multiple questions simultaneously via 
   batch configurations, demonstrating concurrency-bounded throttling (`max_concurrency`).
================================================================================
"""

import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# ==============================================================================
# 1. INITIAL PREPARATION: Environment & Model Setup
# ==============================================================================
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Initialize the Gemini model for streaming and batch execution
model_gemini = init_chat_model("google_genai:gemini-2.5-flash")

question2 = "Why do mirrors reflect our image?"
question3 = "How does a touch screen on a phone work?"
question4 = "What makes a rainbow appear in the sky?"
question5 = "Write me a 200 words paragraph on GPS Navigation App"
append = " Please explain in 3 sentences."

# ==============================================================================
# 2. STREAMING CAPABILITIES (Progressive Output)
# ==============================================================================
print("======================================================================")
print("🔥 DEMONSTRATING STREAMING UTILITIES")
print("======================================================================")

print("1. Standard Invoke (Waiting for entire payload):")
print(model_gemini.invoke(question5).content)
print("\n" + "-"*40)

print("2. Progressive Stream Processing (Token by Token Chunks):")
# Calling stream() returns an iterator that yields output chunks as they are produced.
for chunk in model_gemini.stream(question5):
    # Depending on the model provider, text chunks are stored in .text or .content
    text_chunk = chunk.text if hasattr(chunk, 'text') else chunk.content
    print(text_chunk, end="|", flush=True)
print("\n")

print("-" * 60)
print("3. Short Query Streaming Example:")
for chunk in model_gemini.stream(question4 + append):
    text_chunk = chunk.text if hasattr(chunk, 'text') else chunk.content
    print(text_chunk, end="|", flush=True)
print("\n")

# ==============================================================================
# 3. BATCH CAPABILITIES (Parallel Async Processing)
# ==============================================================================
print("======================================================================")
print("🔥 DEMONSTRATING BATCH PROCESSING")
print("======================================================================")

# Batching a collection of independent requests can significantly 
# improve performance as the processing is handled concurrently
print("1. Standard Batch Execution Array:")
batch_responses = model_gemini.batch([
    question2,
    question3,
    question4
])
for idx, res in enumerate(batch_responses, 1):
    print(f"\nResponse {idx}:\n{res.content}")

print("\n" + "-"*40)

# Batching execution bounded by explicit concurrent execution window constraints
print("2. Concurrency-Bounded Batch Execution:")
concurrency_responses = model_gemini.batch(
    [
        question2,
        question3,
        question4
    ],
    config={
        'max_concurrency': 5,  # Limit to 5 parallel calls max
    }
)
for idx, res in enumerate(concurrency_responses, 1):
    print(f"\nBounded Response {idx}:\n{res.content}")