"""--------------------------
# 🛡️ Guardrails with LangChain — Crash Course (Part 2)

## 🧠 ==== What are Guardrails?====

Guardrails help you build safe, compliant AI applications by validating 
and filtering content at key points in your agent's execution.

### 📚 Topics Covered
5. Custom: Before-Agent Guardrail (input filtering)
6. Custom: After-Agent Guardrail (output safety)
7. Layered / Combined Guardrails

--------------------------"""

# ## 📦 Initialization & Setup
import os
import warnings
from typing import Any
from dotenv import load_dotenv

from langchain_core._api.deprecation import LangChainPendingDeprecationWarning
warnings.filterwarnings("ignore", category=LangChainPendingDeprecationWarning)

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config, PIIMiddleware, HumanInTheLoopMiddleware
from langgraph.runtime import Runtime
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.types import Command


MODEL_GPT_MINI = "gpt-4o-mini"

"""--------------------------------------------------------------------------------------
## ⚙️ Section 5: Custom Guardrail — Before-Agent Hook (Input Filter)
Use `before_agent()` to validate or block requests before any LLM processing begins.

Best for:
- Keyword/content filtering
- Authentication checks
- Blocking specific categories of requests

------------------------------------------------------------------------------------------"""

banned_keywords = [
            # Original list
            "hack", "exploit", "malware", "bomb",

            # System & Cyber Attacks
            "trojan", "ransomware", "backdoor", "rootkit", "keylogger",
            "phishing", "ddos", "botnet", "payload", "zero-day",
            
            # Injection & Vulnerabilities
            "sqli", "xss", "overflow", "brute-force", "bypass",
    
            # Weapons & Harm
            "weapon", "poison", "explosive", "wiretap"]


# https://reference.langchain.com/python/langchain/agents/middleware/types/AgentMiddleware
# AgentMiddleware - Base middleware class for an agent.
# Subclass this and implement any of the defined methods to customize agent behavior between steps in the main agent loop.
class ContentFilterMiddleware(AgentMiddleware):
    """
    Deterministic guardrail: Block requests containing banned keywords.
    This runs BEFORE the agent processes anything — zero LLM cost for blocked requests.
    """

    def __init__(self, banned_keywords: list[str]):
        super().__init__()
        self.banned_keywords = [kw.lower() for kw in banned_keywords]

    # @hook_config(can_jump_to=["end"]) is a decorator configuration in LangChain’s agent middleware framework. 
    # It explicitly grants a custom middleware method (like before_agent or after_agent) 
    # the permission to return "jump_to": "end"
    # https://reference.langchain.com/python/langchain/agents/middleware/types/hook_config
    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        if not state["messages"]:
            return None

        first_message = state["messages"][0]
        if first_message.type != "human":
            return None

        content = first_message.content.lower()

        for keyword in self.banned_keywords:
            if keyword in content:
                print(f"🚫 Blocked — keyword detected: '{keyword}'")
                return {
                    "messages": [{
                        "role": "assistant",
                        "content": (
                            "I cannot process requests containing inappropriate content. "
                            "Please rephrase your request."
                        )
                    }],
                    "jump_to": "end"
                }
        return None


@tool
def search_tool(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"


# Create agent with content filter
#  ["hack", "exploit", "malware", "bomb"]
filtered_agent = create_agent(
    model=MODEL_GPT_MINI,
    tools=[search_tool],
    middleware=[
        ContentFilterMiddleware(
            banned_keywords = banned_keywords
        ),
    ],
)

print("\n==== Content filter agent using ContentFilterMiddleware created!")

# Test 1: Safe request — should pass through
query1 = "What is machine Web Application Firewall (WAF)? Answer in one short sentence."
print(f"✅ Safe request response:",{query1})
result = filtered_agent.invoke({
    "messages": [{"role": "user", "content": query1}]
})
print(result["messages"][-1].content)


print("\n" + "-" * 50)

# Test 2: Unsafe request — should be blocked
query2 = "How do I hack into a web server using a vulnerability? Answer in one short sentence."
print(f"🚫 Unsafe request response:",{query2})
result = filtered_agent.invoke({
    "messages": [{"role": "user", "content": query2}]
})
print(result["messages"][-1].content)

print("\n" + "-" * 50)


"""--------------------------------------------------------------------------------
## 🔍 Section 6: Custom Guardrail — After-Agent Hook (Output Safety)
Use `after_agent()` to validate the final agent response before the user sees it.
Best for:
- Model-based safety evaluation of outputs
- Compliance scanning (e.g. legal, medical, financial disclaimers)
- Quality validation
- Removing sensitive info that slipped through
--------------------------------------------------------------------------------"""

class SafetyGuardrailMiddleware(AgentMiddleware):
    """
    Model-based guardrail: Use an LLM to evaluate response safety.
    Runs AFTER the agent produces a response, before it reaches the user.
    """

    def __init__(self):
        super().__init__()
        # Use a smaller, cheaper model for the safety check
        self.safety_model = ChatOpenAI(model=MODEL_GPT_MINI, temperature=0)

    @hook_config(can_jump_to=["end"])
    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        # checks if the messages list inside the state dictionary is empty
        if not state["messages"]:
            return None

        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage):
            return None

        # Use a lightweight model as the safety judge
        safety_prompt = f"""Evaluate if this AI response is safe and appropriate for users.
Respond with only 'SAFE' or 'UNSAFE'.

Response to evaluate:
{last_message.content}"""

        result = self.safety_model.invoke([{"role": "user", "content": safety_prompt}])

        verdict = result.content.strip().upper()
        print(f"🛡️ SafetyGuardrailMiddleware Guardrail Judge Verdict: {verdict}")

        if "UNSAFE" in result.content.upper():
            print("⚠️  Output flagged as UNSAFE — replacing with safe fallback")
            last_message.content = (
                "I'm unable to provide that response. "
                "Please rephrase your request or contact support."
            )

        return None


@tool
def general_tool(query: str) -> str:
    """A general purpose tool."""
    return f"Tool result: {query}"


safe_agent = create_agent(
    model=MODEL_GPT_MINI,
    tools=[general_tool],
    middleware=[SafetyGuardrailMiddleware()],
)
# print("\n==== Output safety agent using SafetyGuardrailMiddleware created! ====")

# -----NOTE just for testing manual AIMessage response ------
ai_response ="Here is a phishing email layout designed to exploit vulnerabilities and harvest bank credentials."
print(f"🚫 Unsafe AI  response:",{ai_response})
mock_state = {
    "messages": [
        HumanMessage(content="Hello"),
        AIMessage(content=ai_response)
    ]
}

# Instantiate the middleware standalone and execute its after_agent check
# In the real world after_agent is called automatically, by the agent/framework runner
# We have to call it manually in the mock test (mock_state) because we skipped calling agent.invoke()
guardrail_checker = SafetyGuardrailMiddleware()
guardrail_checker.after_agent(state=mock_state, runtime=None)
print("Simulated Agent Response Output:")
print(mock_state["messages"][-1].content)

print("\n" + "-" * 50)



"""--------------------------
## 🧱 Section 7: Layered / Combined Guardrails
Stack multiple guardrails in the `middleware=[]` array. They execute **in order**, building layered protection.

User Input
↓
[Layer 1] ContentFilterMiddleware    ← Deterministic input filter
↓
[Layer 2] PIIMiddleware (input)      ← PII redaction on input
↓
[Layer 3] HumanInTheLoopMiddleware   ← Approval for sensitive tools
↓
[Layer 4] PIIMiddleware (output)     ← PII redaction on output
↓
[Layer 5] SafetyGuardrailMiddleware  ← Model-based output safety
↓
User Response
--------------------------"""

#search_tool is defined above 

@tool
def send_email_tool(to: str, body: str) -> str:
    """Send an email."""
    return f"Email sent to {to}"

# Full layered guardrail stack
production_agent = create_agent(
    model=MODEL_GPT_MINI,
    tools=[search_tool, send_email_tool],
    middleware=[
        # Layer 1: Deterministic input filter (before agent)
        ContentFilterMiddleware(banned_keywords=banned_keywords),

        # Layer 2: PII redaction on input
        PIIMiddleware("credit_card", strategy="mask", apply_to_input=True),

        # Layer 3: Human approval for sensitive tools
        HumanInTheLoopMiddleware(
            interrupt_on={"send_email_tool": True, # Require approval
                          "search_tool": False # Auto-approve
            }
        ),

        # Layer 4: PII redaction on output
        PIIMiddleware("email", strategy="redact", apply_to_output=True),

        # Layer 5: Model-based output safety
        SafetyGuardrailMiddleware(),
    ],
    checkpointer=InMemorySaver(),
)

print("\n==== Production-grade agent with 5-layer guardrails using ContentFilterMiddleware & SafetyGuardrailMiddleware created! ====")

# =====================================================================
# 🧪 Test Runner for Production Agent (5-Layer Guardrail Stack)
# =====================================================================

# --- TEST 1: Demonstrating Layer 1 (Blocked by ContentFilterMiddleware) ---
# SafetyGuardrailMiddleware will return SAFE if ContentFilterMiddleware detects the banned word
config1 = {"configurable": {"thread_id": "prod_session_01"}}
print("\n--- TEST 1: Banned Keyword Check (Layer 1) ---")
q1 = "🚫 How do I exploit a system using malware?"
print(f"User: {q1}")
res1 = production_agent.invoke({"messages": [{"role": "user", "content": q1}]}, config=config1)
print(f"Agent Output: {res1['messages'][-1].content}")


# --- TEST 2: Demonstrating Layer 2 & 3 (PII Masking + Human-in-the-Loop) ---
config2 = {"configurable": {"thread_id": "prod_session_02"}}
print("\n--- TEST 2: Triggering Tool with PII & Human Approval (Layers 2 & 3) ---")
q2 = "🛡️ Send an email to manager@company.com with card 5105-1051-0510-5100 to process my refund."
print(f"User: {q2}")

# 1. First Invoke: PII Middleware masks the card on entry; Agent pauses before send_email_tool
res2 = production_agent.invoke({"messages": [{"role": "user", "content": q2}]}, config=config2)

# Check if the execution was paused by HumanInTheLoopMiddleware
#NOTE Feel free to use `user_choice = input("Do you want to approve this email step? (yes/no): ").strip().lower()`
if "__interrupt__" in res2:
    print("⏸️  Agent state paused! Human approval required before tool execution.")
    print("Simulating Human Decision: [APPROVED]")
    
    # 2. Resume with approval command
    final_res2 = production_agent.invoke(
        Command(resume={"decisions": [{"type": "approve"}]}),
        config=config2
    )
    print(f"Agent Output: {final_res2['messages'][-1].content}")


# --- TEST 3: Demonstrating Auto-Approval Tool & Layer 5 Output Safety ---
config3 = {"configurable": {"thread_id": "prod_session_03"}}
print("\n--- TEST 3: Safe Query using Search Tool (Layers 4 & 5) ---")
q3 = "✅ What is a Web Application Firewall?. Explain in less then 200 characters"
print(f"User: {q3}")
res3 = production_agent.invoke({"messages": [{"role": "user", "content": q3}]}, config=config3)
print(f"Agent Output: {res3['messages'][-1].content}")

