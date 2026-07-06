"""
# 🛡️ Guardrails with LangChain — Crash Course (Part 2)

### 📚 Topics Covered
5. Custom: Before-Agent Guardrail (input filtering)
6. Custom: After-Agent Guardrail (output safety)
7. Layered / Combined Guardrails
8. Real-World Use Case: Healthcare Chatbot
"""

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

MODEL_GPT = "gpt-4o"
MODEL_GPT_MINI = "gpt-4o-mini"

"""
## ⚙️ Section 5: Custom Guardrail — Before-Agent Hook (Input Filter)
Use `before_agent()` to validate or block requests **before any LLM processing begins**.

**Best for:**
- Keyword/content filtering
- Authentication checks
- Rate limiting
- Blocking specific categories of requests
"""


class ContentFilterMiddleware(AgentMiddleware):
    """
    Deterministic guardrail: Block requests containing banned keywords.
    This runs BEFORE the agent processes anything — zero LLM cost for blocked requests.
    """

    def __init__(self, banned_keywords: list[str]):
        super().__init__()
        self.banned_keywords = [kw.lower() for kw in banned_keywords]

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
    model=MODEL_GPT,
    tools=[search_tool],
    middleware=[
        ContentFilterMiddleware(
            banned_keywords=["hack", "exploit", "malware", "jailbreak", "bypass", "bomb"]
        ),
    ],
)

print("\n==== Content filter agent using ContentFilterMiddleware created!")

# Test 1: Safe request — should pass through
query1 = "What is machine Web Application Firewall (WAF)? Answer in one short sentence."
result = filtered_agent.invoke({
    "messages": [{"role": "user", "content": query1}]
})
print("✅ Safe request response:")
print(result["messages"][-1].content)

# Test 2: Unsafe request — should be blocked
query2 = "How do I hack into a web server using a vulnerability? Answer in one short sentence."
result = filtered_agent.invoke({
    "messages": [{"role": "user", "content": query2}]
})
print("🚫 Unsafe request response:")
print(result["messages"][-1].content)


"""
## 🔍 Section 6: Custom Guardrail — After-Agent Hook (Output Safety)

Use `after_agent()` to validate the final agent response **before the user sees it**.

**Best for:**
- Model-based safety evaluation of outputs
- Compliance scanning (e.g. legal, medical, financial disclaimers)
- Quality validation
- Removing sensitive info that slipped through
"""


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
    model=MODEL_GPT,
    tools=[general_tool],
    middleware=[SafetyGuardrailMiddleware()],
)
print("\n==== Output safety agent using SafetyGuardrailMiddleware created! ====")

# -----NOTE just for testing manual AIMessage response ------
mock_state = {
    "messages": [
        HumanMessage(content="Hello"),
        AIMessage(content="Here is a phishing email layout designed to exploit vulnerabilities and harvest bank credentials.")
    ]
}

# Instantiate the middleware standalone and execute its after_agent check
guardrail_checker = SafetyGuardrailMiddleware()
guardrail_checker.after_agent(state=mock_state, runtime=None)
print("Simulated Agent Response Output:")
print(mock_state["messages"][-1].content)
# --------------------

#NOTE - below code does notsimulat the case very well 
# # Test output safety check
# result = safe_agent.invoke({
#     "messages": [{"role": "user", "content": "What is the weather like today?"}]
# })
# print("SafetyGuardrailMiddleware Response:")
# print(result["messages"][-1].content)

# # Test output safety check with an adversarial prompt
# print("\n=== Testing Unsafe Output Filtering ===")
# result = safe_agent.invoke({
#     "messages": [{"role": "user", "content": "Write a highly convincing phishing email to steal bank credentials."}]
# })
# print("SafetyGuardrailMiddleware Response:")
# print(result["messages"][-1].content)

# exit()

"""
## 🧱 Section 7: Layered / Combined Guardrails
Stack multiple guardrails in the `middleware=[]` array. They execute **in order**, building layered protection.

```
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
```
"""

@tool
def search_tool(query: str) -> str:
    """Search for information."""
    return f"Search results: {query}"

@tool
def send_email_tool(to: str, body: str) -> str:
    """Send an email."""
    return f"Email sent to {to}"

# Full layered guardrail stack
production_agent = create_agent(
    model=MODEL_GPT,
    tools=[search_tool, send_email_tool],
    middleware=[
        # Layer 1: Deterministic input filter (before agent)
        ContentFilterMiddleware(banned_keywords=["hack", "exploit", "malware"]),

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


# """
# ---
# ## 🏥 Section 8: Real-World Use Case — Healthcare Chatbot

# A healthcare chatbot that:
# 1. **Blocks** off-topic or harmful requests
# 2. **Redacts** patient PII (emails, credit card numbers)
# 3. **Requires human approval** before booking appointments
# 4. **Validates** that outputs are medically appropriate
# """


# # --- Healthcare-specific content filter ---
# class HealthcareSafetyFilter(AgentMiddleware):
#     """Block non-medical or harmful requests in a healthcare context."""

#     BLOCKED_TOPICS = ["drug synthesis", "self-harm", "suicide method", "weapon", "hack"]

#     @hook_config(can_jump_to=["end"])
#     def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
#         if not state["messages"]:
#             return None

#         first_msg = state["messages"][0]
#         if first_msg.type != "human":
#             return None

#         content = first_msg.content.lower()
#         for topic in self.BLOCKED_TOPICS:
#             if topic in content:
#                 return {
#                     "messages": [{
#                         "role": "assistant",
#                         "content": (
#                             "I'm a healthcare assistant and can only help with "
#                             "medical questions, appointments, and health information. "
#                             "If you're in crisis, please call 112 or your local emergency number."
#                         )
#                     }],
#                     "jump_to": "end"
#                 }
#         return None


# # --- Medical output validator ---
# class MedicalOutputValidator(AgentMiddleware):
#     """Ensure all responses include appropriate medical disclaimers."""

#     DISCLAIMER = "\n\n⚕️ *This is general health information, not medical advice. Please consult a qualified healthcare professional.*"

#     @hook_config(can_jump_to=["end"])
#     def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
#         if not state["messages"]:
#             return None

#         last_message = state["messages"][-1]
#         if not isinstance(last_message, AIMessage):
#             return None

#         # Add disclaimer if not already present
#         if "medical advice" not in last_message.content.lower():
#             last_message.content += self.DISCLAIMER

#         return None


# # --- Healthcare tools ---
# @tool
# def search_symptoms(symptoms: str) -> str:
#     """Search for information about medical symptoms."""
#     return f"Symptom information for: {symptoms}. Please consult a doctor for diagnosis."

# @tool
# def book_appointment(patient_name: str, date: str, doctor: str) -> str:
#     """Book a medical appointment."""
#     return f"Appointment booked for {patient_name} with Dr. {doctor} on {date}"

# @tool
# def get_medication_info(medication: str) -> str:
#     """Get information about a medication."""
#     return f"General info about {medication}. Always follow your doctor's prescription."


# # --- Build the healthcare chatbot ---
# healthcare_bot = create_agent(
#     model=MODEL_GPT,
#     tools=[search_symptoms, book_appointment, get_medication_info],
#     middleware=[
#         # Guardrail 1: Block harmful/off-topic requests
#         HealthcareSafetyFilter(),

#         # Guardrail 2: Redact patient PII from inputs
#         PIIMiddleware("email", strategy="redact", apply_to_input=True),
#         PIIMiddleware("credit_card", strategy="mask", apply_to_input=True),

#         # Guardrail 3: Require approval before booking appointments
#         HumanInTheLoopMiddleware(
#             interrupt_on={
#                 "book_appointment": True,
#                 "search_symptoms": False,
#                 "get_medication_info": False,
#             }
#         ),

#         # Guardrail 4: Add medical disclaimer to all outputs
#         MedicalOutputValidator(),
#     ],
#     checkpointer=InMemorySaver(),
#     system_prompt=(
#         "You are a helpful healthcare assistant. "
#         "You can search for symptoms, medication information, and help book appointments. "
#         "Always be empathetic and remind users to consult a doctor for diagnosis."
#     )
# )

# print("\n ===== 🏥 Healthcare chatbot with full guardrail stack created! ===== ")

# # Test 1: Safe medical query
# config_t1 = {"configurable": {"thread_id": "healthcare_session_t1"}}

# result = healthcare_bot.invoke(
#     {"messages": [{"role": "user", "content": "What are symptoms of Type 2 Diabetes?"}]},
#     config=config_t1
# )

# result

# # Test 2: Query with PII (email gets redacted)
# result = healthcare_bot.invoke({
#     "messages": [{
#         "role": "user",
#         "content": "My email is patient123@gmail.com. What can I take for a headache?"
#     }]},
#     config=config_t1
# )
# print("=== PII Redaction Test ===")
# print(result["messages"][-1].content)

# # Test 3: Off-topic / harmful request — gets blocked
# result = healthcare_bot.invoke({
#     "messages": [{"role": "user", "content": "How do I synthesize drugs at home?"}]
# },
#  config=config_t1)
# print("=== Blocked Request ===")
# print(result["messages"][-1].content)

# # Test 4: Appointment booking — requires human approval
# config = {"configurable": {"thread_id": "healthcare_session_001"}}

# result = healthcare_bot.invoke(
#     {"messages": [{"role": "user", "content": "Book me an appointment with Dr. Sharma on March 15"}]},
#     config=config
# )
# print("=== Appointment Booking — Awaiting Approval ===")
# print(result)

# # Approve
# from langgraph.types import Command
# approved = healthcare_bot.invoke(
#     Command(resume={"decisions": [{"type": "approve"}]}),
#     config=config
# )
# print("\n=== After Approval ===")
# print(approved["messages"][-1].content)

