import os
import random
from pprint import pprint
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage

# ==============================================================================
# 1. INITIAL PREPARATION: Environment & Keys Setup
# ==============================================================================
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

GROQ_MODEL = "llama-3.1-8b-instant"
llm = ChatGroq(model=GROQ_MODEL)

# ==============================================================================
# 2. UNDERSTANDING CORE MESSAGE OBJECTS
# ==============================================================================
print("======================================================================")
print("📦 PART 1: CORE MESSAGE OBJECTS & CHAT INVOCATION")
print("======================================================================")

messages = [AIMessage(content="Please tell me how can I help", name="LLMModel")]
messages.append(HumanMessage(content="I want to learn coding", name="Ash"))
messages.append(AIMessage(content="Which programming language you want to learn", name="LLMModel"))
messages.append(HumanMessage(content="I want to learn python programming language", name="Ash"))

# NOTE: This is a  demonstration of how message objects are stored in Python List (not Graph state)
print("Current conversation history stack:")
for message in messages:
    message.pretty_print()

# Pinging the model with the message history array
result = llm.invoke(messages)
# NOTE: Response is not important here as we are demonstrating the message handling.
# print(f"\n🤖 LLM Raw Response:\n{result.content}\n")

# ==============================================================================
# 3. TOOL BINDING MECHANICS
# ==============================================================================
print("\n======================================================================")
print("🛠️ PART 2: TOOL BINDING WITH THE LLM")
print("======================================================================")

def add(a: int, b: int) -> int:
    """Add a and b
    Args:
        a (int): first int
        b (int): second int
    Returns:
        int
    """
    return a + b

# Bind the tool scheme to the model metadata definition
tool_message = HumanMessage(content="What is 2 plus 2", name="Ash")
llm_with_tools = llm.bind_tools([add])
tool_call = llm_with_tools.invoke([tool_message])

print("Tool Call content string returned:")
print(f"Content: '{tool_call.content}'")
print(f"Detected Tool Calls Array: {tool_call.tool_calls}\n")

print("Updated conversation history stack:")
tool_message.pretty_print()

# ==============================================================================
# 4. THE STATE OVERRIDE PROBLEM (Why we need Reducers)
# ==============================================================================
print("\n======================================================================")
print("❌ PART 3: THE STATE OVERRIDE PROBLEM (STANDARD PYTHON BEHAVIOR)")
print("======================================================================")

# Simulation: Standard state dictionary without a custom Reducer framework
mock_state = {
    "messages": [HumanMessage(content="Hello, graph!")]
}

print("Initial State Context:")
print(mock_state)

# A Node executes and returns a state update dictionary payload
node_update = {
    "messages": [AIMessage(content="Hello back!")]
}

print(f"\nNode finishes execution and returns update: {node_update}")

# Standard dict merge simulation (Default state update logic)
mock_state.update(node_update)

print("\nResulting State (Notice the initial 'Hello, graph!' was COMPLETELY OVERRIDDEN):")
print(mock_state)
print("⚠️ Conclusion: Without a Reducer, we lose our entire conversation history memory!")
print("======================================================================\n")

# NOTE: Please note I donot use StateGraph.compile() in this example. We use it in next example