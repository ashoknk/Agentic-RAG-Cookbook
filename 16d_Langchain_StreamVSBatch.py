"""
================================================================================
This script serves as a technical showcase of optimization strategies within LangChain, 
specifically focusing on the == performance and design tradeoffs == 
between real-time token **Streaming** and parallelized asynchronous **Batch Processing**.

THE RUNTIME PATTERNS:
---------------------
- **Streaming (`.stream()`)**: 
    Sets up an iterative generator that yields individual text 
  tokens piece-by-piece as they are processed by the LLM, reducing perceived latency.
- **Batching (`.batch()`)**: 
    Aggregates a compilation of independent text requests 
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
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Initialize the model for streaming and batch execution
OPENAI_MODEL = "gpt-4o-mini"
model = init_chat_model(OPENAI_MODEL)

question1 = "Why do mirrors reflect our image?"
question2 = "How does a touch screen on a phone work?"
question3 = "What makes a rainbow appear in the sky?"
question4 = "Write me a 200 words paragraph on GPS Navigation App"
append = " Please explain in 3 sentences."

# ==============================================================================
# 2. STREAMING CAPABILITIES (Progressive Output)
# ==============================================================================
print("======================================================================")
print("🔥 DEMONSTRATING STREAMING UTILITIES")
print("======================================================================")

print("1. Standard Invoke (Waiting for entire payload):")
print(model.invoke(question1 + append).content)
print("\n" + "-"*40)

print("2. Progressive Stream Processing (Token by Token Chunks):")
# Calling stream() returns an iterator that yields output chunks as they are produced.
for chunk in model.stream(question1):
    # Depending on the model provider, text chunks are stored in .text or .content
    # Evaluates whether an object (chunk) possesses an attribute or method named text
    # adding end="|" inside print() changes the character printed at the end of the output from a default \n to a pipe symbol (|
    # flush=True forces the print() function to immediately clear its output buffer
    text_chunk = chunk.text if hasattr(chunk, 'text') else chunk.content
    print(text_chunk, end="|", flush=True)
print("\n")


print("-" * 60)
print("3. Short Query Streaming Example:")
for chunk in model.stream(question2 + append):
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
# invoke() processes exactly one prompt at a time, batch() is designed to process multiple independent prompts at once
# Because batch() fires the requests off simultaneously, the total time to get all answers back is roughly equal to the time of the single longest-running response.
print("1. Standard Batch Execution Array:")
batch_responses = model.batch([
    question2,
    question3,
    question4
])
for idx, res in enumerate(batch_responses, 1):
    print(f"\nResponse {idx}:\n{res.content}")

print("\n" + "-"*40)

# Batching execution bounded by explicit concurrent execution window constraints
# max_concurrency' setting acts as a throttle on the parallelism 
# By default, when you pass a list to model.batch(), LangChain will try to fire off all of those requests simultaneously using as many threads as it needs
# No matter how big my input list is, never have more than 2 or 5 API requests actively running at the exact same moment."
print("2. Concurrency-Bounded Batch Execution:")
concurrency_responses = model.batch(
    [
        question1,
        question2,
        question3,
        question4
    ],
    config={
        'max_concurrency': 2,  # Limit to 2 parallel calls max
    }
)
for idx, res in enumerate(concurrency_responses, 1):
    print(f"\nBounded Response {idx}:\n{res.content}")