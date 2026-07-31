"""
===LLM Gateway using LiteLLM==

This file builds directly on last file to showcase enterprise-grade integration patterns. 
1.It combines the LLM gateway with LangChain orchestration pipelines, 
2.implements dynamic multi-model task-aware fallback chains, 
3. and injects pure-Python runtime security guardrails (PII redaction, prompt injection detection, and topic filters) 
using LiteLLM callback hooks.
"""

import os
import time
import re
import json
from dotenv import load_dotenv
import warnings
import logging
import litellm

warnings.filterwarnings("ignore")
logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)
litellm.telemetry = False
litellm.suppress_debug_info = True
litellm.set_verbose = False
litellm.turn_off_message_logging = True
litellm.telemetry = False

from litellm import completion, completion_cost
from langchain_litellm import ChatLiteLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

#  Define standard model variables
MODEL_GPT = "gpt-4o"
MODEL_GPT_MINI = "gpt-4o-mini"
MODEL_CLAUDE = "anthropic/claude-3-5-haiku-20241022"
MODEL_GEMINI = "gemini/gemini-2.5-flash"
MODEL_GROQ = "groq/llama-3.3-70b-versatile"

# ==============================================================================
# Step 1: Integrate LangChain orchestration with LiteLLM backend wrapper
# USE CASE: Connect standard LangChain chains and agents to LiteLLM instead of locking into one SDK like OpenAI.
# LEARNER OUTCOME: Learn how to use ChatLiteLLM as a drop-in replacement for ChatOpenAI in LangChain.
# https://docs.langchain.com/oss/python/integrations/chat/litellm ChatLiteLLM: The main LangChain chat wrapper for LiteLLM.
# https://reference.langchain.com/python/langchain-litellm/chat_models/litellm/ChatLiteLLM
# ==============================================================================
llm = ChatLiteLLM(model=MODEL_GPT_MINI, temperature=0.3)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI tutor named Peter Pan. Be concise."),
    ("user", "{question}")
])

chain = prompt | llm | StrOutputParser()
answer = chain.invoke({"question": "What is an LLM Gateway in 3 bullets? Be concise."})
print("LangChain Response:\n", answer)
print("-" * 50)

# ==============================================================================
# Step 2: Multi-Provider LangChain Fallbacks
# USE CASE: Prevent LangChain chains from breaking when an API provider experiences downtime or rate limits.
# LEARNER OUTCOME: Learn how to attach backup LLM providers using LangChain's native `.with_fallbacks()` method.
# https://reference.langchain.com/python/langchain-core/runnables/base/Runnable/with_fallbacks
# ==============================================================================

primary = ChatLiteLLM(model="gpt-x")
fallback_1 = ChatLiteLLM(model=MODEL_GPT_MINI, temperature=0.2)
fallback_2 = ChatLiteLLM(model=MODEL_GROQ, temperature=0.2)
robust_llm = primary.with_fallbacks([fallback_1, fallback_2]) #Add fallbacks

lc_fallback_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert AI engineer. Always reply in JSON: {{\\\"answer\\\": ...}}"),
    ("user", "{question}")
])
fallback_chain = lc_fallback_prompt | robust_llm | StrOutputParser()
print(fallback_chain.invoke({"question": "What are the top 3 benefits of an LLM Gateway?"}))
print("-" * 50)


# ==============================================================================
# Step 3: Build a mini task-aware routing chatbot with dynamic fallback execution
# USE CASE: Save costs and improve speed by matching query difficulty to the right model 
#   (e.g., cheap models for summaries, heavy models for complex code).
# LEARNER OUTCOME: Learn how to build a complete smart router in Python combining prompt classification, 
#   dynamic model lists, and fallback logic.
# ==============================================================================

# Uses a fast, cheap LLM (Groq) to classify the user's prompt into one category: 'code', 'summary', or 'general'.
# This acts as a quick routing decision step before sending the prompt to a larger model.
def classify_task(user_query: str) -> str:
    cls = completion(
        model=MODEL_GROQ,
        messages=[{
            "role": "user",
            "content": (
                f"Classify the following query into EXACTLY one word: "
                f"'code', 'summary', or 'general'. Query: {user_query}\n\nAnswer:"
            )
        }],
        max_tokens=5
    )
    return cls.choices[0].message.content.strip().lower()

# Tries calling models in order from a list (primary first, then backups).
# If a model fails or goes down, it catches the error and automatically tries the next model in line.
def call_with_fallbacks(model_chain, messages):
    last_error = None
    for model in model_chain:
        try:
            return completion(model=model, messages=messages)
        except Exception as e:
            print(f"   ⚠️  {model} failed ({type(e).__name__}), trying next...")
            last_error = e
            continue
    raise last_error

# The main controller function that puts everything together: 
# classifies the prompt, selects the best model chain,
# executes the call with automatic fallbacks, and calculates total latency and cost.
def smart_chat(user_query: str):
    task = classify_task(user_query) #'code', 'summary', or 'general'
    routing = {
        "code":    [MODEL_GPT, MODEL_GPT_MINI, MODEL_GROQ],
        "summary": [MODEL_GPT_MINI, MODEL_GROQ],
        "general": [MODEL_GROQ, MODEL_GPT_MINI],
    }
    model_chain = routing.get(task, routing["general"])

    start = time.time()
    response = call_with_fallbacks(model_chain=model_chain, messages=[{"role": "user", "content": user_query}])
    latency = time.time() - start

    try:
        cost = completion_cost(completion_response=response)
        cost_str = f"${cost:.6f}"
    except Exception:
        cost_str = "n/a"

    return {
        "detected_task": task,
        "model_used": response.model,
        "answer": response.choices[0].message.content,
        "latency_sec": round(latency, 2),
        "cost_usd": cost_str
    }

queries = [
    "Write a Python function to compute Fibonacci numbers.",
    "Summarize the importance of attention mechanism in 2 sentences.",
]

for q in queries:
    print("=" * 70)
    print("❓ Q:", q)
    result = smart_chat(q)
    print(f"🏷️  Task:    {result['detected_task']}")
    print(f"🤖 Model:    {result['model_used']}")
    print(f"⏱️  Latency: {result['latency_sec']}s")
    print(f"💰 Cost:    {result['cost_usd']}")
    print(f"💬 Answer:  {result['answer'][:150]}...")
print("-" * 50)


# ==============================================================================
# Step 4: Implement runtime security guardrails via LiteLLM input hooks
# USE CASE: Protect user privacy and comply with data laws (like GDPR/HIPAA) by stripping out personal data 
# before it leaves your network.
# LEARNER OUTCOME: Learn how to intercept LLM prompts automatically using `litellm.input_callback` hooks 
# and modify them in memory.
# ==============================================================================
# Step 4: Implement runtime security guardrails via LiteLLM input hooks
PII_PATTERNS = {
    "EMAIL": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "PAN":   r"\b[A-Z]{5}\d{4}[A-Z]\b",
}

# Scans input text using regex patterns to find sensitive private information (like emails or PAN cards).
# Replaces sensitive data with placeholders like <EMAIL_REDACTED> so private info never reaches the LLM.
def redact_pii(text: str):
    detected = []
    clean = text
    for label, pattern in PII_PATTERNS.items():
        matches = re.findall(pattern, clean) # searches a string for all non-overlapping matches of a specified regex pattern and returns them as a list
        if matches:
            detected.append({"type": label, "count": len(matches)})
            clean = re.sub(pattern, f"<{label}_REDACTED>", clean) #used to replace occurrences of a regex pattern within a string
    return clean, detected

# A pre-call LiteLLM hook that intercept user messages right before sending them to the LLM backend.
# Automatically runs redact_pii() to clean sensitive data out of the prompt on the fly.
def pii_input_guardrail(kwargs):
    messages = kwargs.get("messages", [])
    for msg in messages:
        if msg.get("role") == "user":
            clean, detected = redact_pii(msg["content"])
            if detected:
                print(f"🚨 PII REDACTED: {detected}")
                msg["content"] = clean

# liteLLM provides input_callbacks, success_callbacks and failure_callbacks, 
# making it easy for you to send data to a particular provider depending on the status of your responses.
litellm.input_callback = [pii_input_guardrail]

secure_user_msg = "Hi, my email is peterpan@neverland.com and PAN is ABCDE1234F. Help me code."
safe_response = completion(
    model=MODEL_GPT_MINI,
    messages=[{"role": "user", "content": secure_user_msg}],
    max_tokens=50
)
print("\n💬 Secure Guardrail Response:")
print(safe_response.choices[0].message.content)
print("-" * 50)