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
---
## 🏥 Section 8: Real-World Use Case — Healthcare Chatbot

A healthcare chatbot that:
1. **Blocks** off-topic or harmful requests
2. **Redacts** patient PII (emails, credit card numbers)
3. **Requires human approval** before booking appointments
4. **Validates** that outputs are medically appropriate
"""

# --- Healthcare-specific content filter ---
class HealthcareSafetyFilter(AgentMiddleware):
    """Block non-medical or harmful requests in a healthcare context."""

    BLOCKED_TOPICS = ["drug synthesis", "self-harm", "suicide method", "weapon", "hack"]

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
                            "If you're in crisis, please call 112 or your local emergency number."
                        )
                    }],
                    "jump_to": "end"
                }
        return None


# --- Medical output validator ---
class MedicalOutputValidator(AgentMiddleware):
    """Ensure all responses include appropriate medical disclaimers."""

    DISCLAIMER = "\n\n⚕️ *This is general health information, not medical advice. Please consult a qualified healthcare professional.*"

    @hook_config(can_jump_to=["end"])
    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        if not state["messages"]:
            return None

        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage):
            return None

        # Add disclaimer if not already present
        if "medical advice" not in last_message.content.lower():
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
    model=MODEL_GPT,
    tools=[search_symptoms, book_appointment, get_medication_info],
    middleware=[
        # Guardrail 1: Block harmful/off-topic requests
        HealthcareSafetyFilter(),

        # Guardrail 2: Redact patient PII from inputs
        PIIMiddleware("email", strategy="redact", apply_to_input=True),
        PIIMiddleware("credit_card", strategy="mask", apply_to_input=True),

        # Guardrail 3: Require approval before booking appointments
        HumanInTheLoopMiddleware(
            interrupt_on={
                "book_appointment": True,
                "search_symptoms": False,
                "get_medication_info": False,
            }
        ),

        # Guardrail 4: Add medical disclaimer to all outputs
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

# Test 1: Safe medical query
print("=== Test 1: Safe medical query ===")
query1 = "What are symptoms of Type 2 Diabetes? Answer in one short sentence."
config_t1 = {"configurable": {"thread_id": "healthcare_session_t1"}}

result = healthcare_bot.invoke(
    {"messages": [{"role": "user", "content": query1}]},
    config=config_t1
)

print(result["messages"][-1].content)

# Test 2: Query with PII (email gets redacted)
print("=== Test 2: Query with PII (email gets redacted) ===")
query2 = "My email is peterpan@neverland.com. What can I take for a headache? Answer in one short sentence."
result = healthcare_bot.invoke({
    "messages": [{
        "role": "user",
        "content": query2
    }]},
    config=config_t1
)
print("=== PII Redaction Test ===")
print(result["messages"][-1].content)

# Test 3: Off-topic / harmful request — gets blocked
query3 = "How do I synthesize drugs at home? Answer in one short sentence."
print("=== Test 3: Off-topic / harmful request — gets blocked ===")
result = healthcare_bot.invoke({
    "messages": [{"role": "user", "content": query3}]
},
 config=config_t1)
print("=== Blocked Request ===")
print(result["messages"][-1].content)

# Test 4: Appointment booking — requires human approval
print("=== Test 4: Appointment booking — requires human approval ===")
query4 = "Book me an appointment with Dr. Sharma on March 15. Answer in one short sentence."
config = {"configurable": {"thread_id": "healthcare_session_001"}}

result = healthcare_bot.invoke(
    {"messages": [{"role": "user", "content": query4}]},
    config=config
)
print("=== Appointment Booking — Awaiting Approval ===")
print(result)

# Approve
from langgraph.types import Command
approved = healthcare_bot.invoke(
    Command(resume={"decisions": [{"type": "approve"}]}),
    config=config
)
print("\n=== After Approval ===")
print(approved["messages"][-1].content)

