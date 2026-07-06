"""
# 🛡️ Guardrails with LangChain — Crash Course (Part 1)

## 🧠 ==== What are Guardrails?====

Guardrails help you build **safe, compliant AI applications** by validating and filtering content at key points in your agent's execution.

### 📚 Topics Covered
1. What are Guardrails & Why do they matter?
2. Two approaches: Deterministic vs Model-based
3. Built-in: PII Detection Middleware
4. Built-in: Human-in-the-Loop Middleware
"""

# ## 📦 Initialization & Setup
import os
import re
import warnings
from getpass import getpass
from dotenv import load_dotenv

from langchain_core._api.deprecation import LangChainPendingDeprecationWarning
warnings.filterwarnings("ignore", category=LangChainPendingDeprecationWarning)

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

MODEL_GPT = "gpt-4o"
MODEL_GPT_MINI = "gpt-4o-mini"


"""
## ⚖️ Section 1: Deterministic Guardrails
### 
- Rule-based: regex, keyword matching, explicit checks
- ✅ Fast, predictable, cost-effective
- ❌ May miss nuanced violations
"""

import re

# --- Deterministic approach ---
def deterministic_guardrail(text: str) -> bool:
    """Returns True if content is blocked."""
    banned_keywords = ["hack", "exploit", "malware", "bomb"]
    return any(kw in text.lower() for kw in banned_keywords)

print("=== Deterministic Guardrail Demo ===")
test_inputs = [
    "How do I hack into a web server?",
    "Which country is Osaka located in?",
    "Explain how to check if attachment to the email is malware?",
]

for single_input in test_inputs:
    blocked = deterministic_guardrail(single_input)
    status = "🚫 BLOCKED" if blocked else "✅ ALLOWED"
    print(f"{status}: {single_input}")


"""
### ⚖️ Section 2: Model-Based Guardrails
- Uses LLMs/classifiers for semantic understanding
- ✅ Catches subtle/nuanced issues
- ❌ Slower and more expensive
"""

from langchain_openai import ChatOpenAI

# --- Model-based approach ---
def model_based_guardrail(text: str) -> str:
    """Uses an LLM to evaluate content safety. Returns SAFE or UNSAFE."""
    model = ChatOpenAI(model=MODEL_GPT_MINI, temperature=0)
    prompt = f"""Is the following user input safe to process? 
Reply with only 'SAFE' or 'UNSAFE'.

Input: {text}"""
    result = model.invoke([{"role": "user", "content": prompt}])
    return result.content.strip()

print("=== Model-Based Guardrail Demo ===")
for single_input in test_inputs:
    verdict = model_based_guardrail(single_input)
    status = "🚫 UNSAFE" if "UNSAFE" in verdict else "✅ SAFE"
    print(f"{status}: {single_input}")

"""
## 🔒 Section 3: Built-in Guardrail — PII Detection Middleware

LangChain provides built-in `PIIMiddleware` for detecting and handling Personally Identifiable Information (PII).

### Supported PII Types:
| Type | Example |
|---|---|
| `email` | user@example.com |
| `credit_card` | 5105-1051-0510-5100 |
| `ip` | 192.168.1.1 |
| `mac_address` | 43:3d:05:59:f1:1a |
| `url` | https://secret-site.com |

### Strategies:
| Strategy | Result |
|---|---|
| `redact` | `[REDACTED_EMAIL]` |
| `mask` | `****-****-****-1234` |
| `hash` | `a8f5f167...` |
| `block` | Raises an exception |
"""

from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

# Define a simple dummy tool
@tool
def customer_lookup(query: str) -> str:
    """Look up customer information."""
    return f"Customer record found for query: {query}"

# Create agent with PII Middleware

agent = create_agent(
    model=MODEL_GPT,
    tools=[customer_lookup],
    middleware=[
        # Redact emails in user input before sending to model
        PIIMiddleware(
            "email",
            strategy="redact",
            apply_to_input=True,
        ),
        # Mask credit cards in user input
        PIIMiddleware(
            "credit_card",
            strategy="mask",
            apply_to_input=True,
        ),
        # Block API keys - raise error if detected
        PIIMiddleware(
            "openAI langchain api_key",
            detector=r"sk-(proj-)?[a-zA-Z0-9]{32,}",
            strategy="block",
            apply_to_input=True,
        ),
        PIIMiddleware(
            "google_api_key",
            detector=r"AIzaSy[a-zA-Z0-9\-_]{33}",
            strategy="block",
            apply_to_input=True,
        ),
    ],
)


print("Agent with PII middleware created successfully!")

# Test PII Redaction
query = "My email is Jane.Smith@example.com and my card is 5105-1051-0510-5100. Can you help me with my late payment?"
# NOTE Below are more samples to test 
# query = "My email is Alex.jones@example.com and my card is 4111-1111-1111-1111. What is my credit limit?"
# query = "My email is Sarah.Williams@example.com and my card is 4000-0000-0000-0002. How do I update my billing address?"

result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": query
    }]
})


print("=== Agent Response ===")
print(result["messages"][-1].content)

# View raw result dictionary
# print(result) NOTE: This print lot of details 
from langchain_core.messages import HumanMessage, ToolMessage

print("=== Filtered HumanMessage and ToolMessage ===")
# Loop through the list of messages inside the result dictionary
for msg in result["messages"]:
    # Check if the message is either a HumanMessage or a ToolMessage
    if isinstance(msg, (HumanMessage, ToolMessage)):
        print(msg)
        print()  # Adds an empty line for clean formatting


# Test API Key Blocking
result2 =""
try:
    result2 = agent.invoke({
        "messages": [{
            "role": "user",
            "content": "Here are my keys: sk-abcdefghijklmnopqrstuvwxyz123456"
        }]
    })
    
except Exception as e:
    print(f"🚫 Blocked as expected: {e}")

# Check latest state of result
print(result2)

result3 =""
try:
    result3 = agent.invoke({
        "messages": [{
            "role": "user",
            "content": "Here are my keys: AIzaSyabcdefghijklmnopqrstuvwxyz1234567890"
        }]
    })
    
except Exception as e:
    print(f"🚫 Blocked as expected: {e}")

print(result3)


"""
## 👤 Section 4: Built-in Guardrail — Human-in-the-Loop Middleware

Pauses agent execution before sensitive operations and waits for human approval.

**Best for:**
- Financial transactions
- Sending emails to external parties
- Deleting production data
- Any operation with significant business impact

**Key requirement:** A `checkpointer` for state persistence across interrupts.
"""

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from langchain_core.tools import tool

@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Search results for: {query}"

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to a recipient."""
    return f"Email sent to {to} with subject: {subject}"

@tool
def delete_records(table: str, condition: str) -> str:
    """Delete records from the database."""
    return f"Deleted records from {table} where {condition}"

# Create agent with HITL middleware
hitl_agent = create_agent(
    model=MODEL_GPT,
    tools=[search_web, send_email, delete_records],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email": True,       # Require approval
                "delete_records": True,   # Require approval
                "search_web": False,      # Auto-approve
            }
        ),
    ],
    checkpointer=InMemorySaver(),  # Required for state persistence
)

print("Human-in-the-Loop agent created!")

# Step 1: Invoke — agent will pause before send_email
config = {"configurable": {"thread_id": "session_001"}}

result = hitl_agent.invoke(
    {"messages": [{"role": "user", "content": "Send an email to investors@company.com about the Q1 results"}]},
    config=config
)

print("=== Agent paused — awaiting human approval ===")
# print(result)

# =======
# =====================================================================
# ⏸️ REAL-TIME TERMINAL INTERRUPT LOOP
# =====================================================================
while True:
    user_choice = input("Do you want to approve this email step? (yes/no): ").strip().lower()
    
    if user_choice == 'yes':
        print("\nSending approval command to the agent...")
        # Resume with an 'approve' decision
        final_result = hitl_agent.invoke(
            Command(resume={"decisions": [{"type": "approve"}]}),
            config=config   # Uses the exact same thread_id to resume
        )
        print("=== Approved! Final response ===")
        print(final_result["messages"][-1].content)
        break
        
    elif user_choice == 'no':
        # Prompt for a reason to feed back into the agentic flow
        reason_string = input("Provide a reason for rejection: ")
        final_result = hitl_agent.invoke(
            Command(resume={"decisions": [{"type": "reject", "reason": reason_string}]}),
            config=config
        )
        print("=== Rejected! Final response ===")
        # print(final_result["messages"][-1].content)
        print("❌ The email draft was rejected. Conversation with agent suspended.")
        break
        
    else:
        print("Invalid input. Please type 'yes' or 'no'.")
# ======


"""
---
## 📝 Summary

| Guardrail Type | Hook | When it Runs | Best For |
|---|---|---|---|
| PII Middleware | Input/Output | Around model calls | Data privacy, compliance |
| Human-in-the-Loop | Tool level | Before sensitive tools | High-stakes decisions |

### 🔑 Key Takeaways
1. **Guardrails = Middleware** — implement them via the `middleware=[]` parameter in `create_agent()`
2. **Layer your guardrails** — defense in depth is best practice
3. **Deterministic first, model-based second** — use cheap rule-based checks early to avoid expensive LLM calls
4. **Human-in-the-Loop requires a checkpointer** — use `InMemorySaver` for dev, persistent store for production
5. **Custom middleware** gives you full control via `before_agent()` and `after_agent()` hooks

### 📚 Additional Resources
- [LangChain Guardrails Docs](https://docs.langchain.com/oss/python/langchain/guardrails)
- [Middleware Docs](https://docs.langchain.com/oss/python/langchain/middleware/overview)
- [Human-in-the-Loop Docs](https://docs.langchain.com/oss/python/langchain/human-in-the-loop)
- [LangSmith for Observability](https://docs.langchain.com/oss/python/langchain/observability)

"""