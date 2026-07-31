"""--------------------------
# 🛡️ Guardrails with LangChain — Crash Course (Part 3)
## 🧠 ==== What are Guardrails?====
Guardrails help you build safe, compliant AI applications by validating 
and filtering content at key points in your agent's execution.

### 📚 Topics Covered
8. Real-World Use Case: Healthcare Chatbot


1. User sends message
       │
       ▼
2. HealthcareSafetyFilter.before_agent()   <-- INVOCATION 1 (Input Check)
       │
       ├─► [ If Harmful ] ──► Return block message & Jump to End
       │                                         │
       ▼ [ If Safe ]                             ▼
3. PIIMiddleware (Input Redaction)         4. MedicalOutputValidator.after_agent() <-- INVOCATION 2 (Output Check)
       │                                         │
       ▼                                         ▼
4. LLM & Tools Execution                   User sees final response

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


"""--------------------------
## 🏥 Section 8: Real-World Use Case — Healthcare Chatbot

A healthcare chatbot that:
1. **Blocks** off-topic or harmful requests
2. **Redacts** patient PII (emails, credit card numbers)
3. **Requires human approval** before booking appointments
4. **Validates** that outputs are medically appropriate
--------------------------"""

# --- Healthcare-specific content filter (Input Guardrail) ---
# Invoked BEFORE any LLM model processing happens for EVERY SINGLE USER REQUEST

class HealthcareSafetyFilter(AgentMiddleware):
    """Block non-medical or harmful requests in a healthcare context."""

    BLOCKED_TOPICS = ["drug synthesis", "self-harm", "suicide method", "weapon", "hack"]

    # before_agent - Decorator used to dynamically create a middleware with the before_agent hook.
    # https://reference.langchain.com/python/langchain/agents/middleware/types/before_agent
    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        if not state["messages"]:
            return None

        first_msg = state["messages"][0]
        if first_msg.type != "human":
            return None

        content = first_msg.content.lower()
        for topic in self.BLOCKED_TOPICS:
            if topic in content:
                return {
                    "messages": [{
                        "role": "assistant",
                        "content": (
                            "I'm a healthcare assistant and can only help with "
                            "medical questions, appointments, and health information. "
                            "If you're in crisis, please call 911 or your local emergency number."
                        )
                    }],
                    "jump_to": "end"
                }
        return None


# --- Medical output validator (Output Guardrail) ---
# Invoked AFTER the agent finishes generating a response (output) for EVERY STEP (Test 1, 2, 3, 4 initial, & 4 after approval).
class MedicalOutputValidator(AgentMiddleware):
    """Ensure all responses include appropriate medical disclaimers."""

    DISCLAIMER = "\n\n⚕️ *DISCLAIMER This is general health information, not medical advice. Please consult a qualified healthcare professional.*"

    # `after_agent` - Decorator used to dynamically create a middleware with the after_agent hook.
    # https://reference.langchain.com/python/langchain/agents/middleware/types/after_agent
    @hook_config(can_jump_to=["end"])
    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        if not state["messages"]:
            return None

        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage):
            return None

        # Add disclaimer if not already present
        if "DISCLAIMER" not in last_message.content.lower():
            last_message.content += self.DISCLAIMER

        return None


# --- Healthcare tools ---
@tool
def search_symptoms(symptoms: str) -> str:
    """Search for information about medical symptoms."""
    return f"Symptom information for: {symptoms}. Please consult a doctor for diagnosis."

@tool
def book_appointment(patient_name: str, date: str, doctor: str) -> str:
    """Book a medical appointment."""
    return f"Appointment booked for {patient_name} with Dr. {doctor} on {date}"

@tool
def get_medication_info(medication: str) -> str:
    """Get information about a medication."""
    return f"General info about {medication}. Always follow your doctor's prescription."


# --- Build the healthcare chatbot ---
healthcare_bot = create_agent(
    model=MODEL_GPT_MINI,
    tools=[search_symptoms, book_appointment, get_medication_info],
    middleware=[
        # --- Guardrail 1: Block harmful/off-topic requests. --- 
        HealthcareSafetyFilter(),
        # --- Guardrail 2: Redact patient PII from inputs --- 
        PIIMiddleware("email", strategy="redact", apply_to_input=True),
        PIIMiddleware("credit_card", strategy="mask", apply_to_input=True),
        # --- Guardrail 3: Require approval before booking appointments --- 
        HumanInTheLoopMiddleware(
            interrupt_on={
                "search_symptoms": False,
                "book_appointment": True,
                "get_medication_info": False,
            }
        ),
        # ---  Guardrail 4: Add medical disclaimer to all outputs --- 
        MedicalOutputValidator(),
    ],
    checkpointer=InMemorySaver(),
    system_prompt=(
        "You are a helpful healthcare assistant. "
        "You can search for symptoms, medication information, and help book appointments. "
        "Always be empathetic and remind users to consult a doctor for diagnosis."
    )
)

print("\n ===== 🏥 Healthcare chatbot with full guardrail stack created! ===== ")

# ----------------- Test 1: Safe medical query -----------------
print("\n=== Test 1: Safe medical query ===")
query1 = "✅ What are symptoms of Type 2 Diabetes? Answer in one short sentence."
print(f"User: {query1}")
config_t1 = {"configurable": {"thread_id": "healthcare_session_t1"}}

result1 = healthcare_bot.invoke(
    {"messages": [{"role": "user", "content": query1}]},
    config=config_t1
)

# print(f"Assistant: {result1['messages'][-1].content}")


# --------------- Test 2: Query with PII (email gets redacted) -----------------
# print("\n=== Test 2: Query with PII (email gets redacted) ===")
query2 = "🛡️ My email is peterpan@neverland.com. What can I take for a headache? Answer in one short sentence."
print(f"User Input: {query2}")

config_t2 = {"configurable": {"thread_id": "healthcare_session_t2"}}
result2 = healthcare_bot.invoke(
    {"messages": [{"role": "user", "content": query2}]},
    config=config_t2
)

# Print the redacted user prompt processed by PIIMiddleware
print(f"Processed Input (Redacted): {result2['messages'][0].content}")
print(f"Assistant Output: {result2['messages'][-1].content}")


# ----------------- Test 3: Off-topic / harmful request — gets blocked -----------------
# print("\n=== Test 3: Off-topic / harmful request — gets blocked ===")
query3 = "🚫 How do I synthesize drugs at home? Answer in one short sentence."
print(f"User: {query3}")

config_t3 = {"configurable": {"thread_id": "healthcare_session_t3"}}
result3 = healthcare_bot.invoke(
    {"messages": [{"role": "user", "content": query3}]},
    config=config_t3
)
print(f"Assistant Output: {result3['messages'][-1].content}")


# ----------------- Test 4: Appointment booking — requires human approval -----------------
print("\n=== Test 4: Appointment booking — requires human approval ===")
query4 = "⏸️ Book me an appointment with Dr.Hannibal Lecter on Friday 13th . Answer in one short sentence."
print(f"User: {query4}")
config_t4 = {"configurable": {"thread_id": "healthcare_session_t4"}}

result4 = healthcare_bot.invoke(
    {"messages": [{"role": "user", "content": query4}]},
    config=config_t4
)

print("\n=== Appointment Booking — Awaiting Approval ===")
# Clean print without token metadata dump
print(f"Assistant: {result4['messages'][-1].content}")

# Approve execution
approved = healthcare_bot.invoke(
    Command(resume={"decisions": [{"type": "approve"}]}),
    config=config_t4
)
print("\n=== After Approval ===")
print(f"Assistant: {approved['messages'][-1].content}")