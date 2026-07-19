"""
================================================================================
This script provides an educational exploration of the foundational unit of context 
in LangChain: the **Message Object Array**. It maps out the precise responsibilities 
of different message subclasses used to maintain conversation state.

THE FIVE CORE CONTEXT CLASSES:
- `SystemMessage`: Establishes behavior guidelines, personas, or strict output constraints.
- `HumanMessage`: Represents text or multimodal inputs provided by the user.
- `AIMessage`: Contains model responses, token metrics, or structured tool requests.
- `ToolMessage`: Holds execution data returned from external tool lookups.

1. PERSONA SETUP: Tests complex instructions using highly specialized `SystemMessage` prompts.
2. HISTORY SIMULATION: Injects manual message arrays into the payload to simulate 
   past interactions, observing how the model maintains context.
3. METADATA ANALYSIS: Examines tracking structures, capturing message identities 
   and provider-agnostic token consumption arrays (`usage_metadata`).
================================================================================
"""

import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

# #### Messages
# Messages are the fundamental unit of context for models in LangChain. 
# They represent the input and output of models, 
# carrying both the content and metadata needed 
# to represent the state of a conversation when interacting with an LLM.
# Messages are objects that contain:
#  - Role - Identifies the message type (e.g. system, user)
#  - Content - Represents the actual content of the message (like text, images, audio, documents, etc.)
#  - Metadata - Optional fields such as response information, message IDs, and token usage
# 
# LangChain provides a standard message type that works across all model providers, 
# ensuring consistent behavior regardless of the model being called.

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

GROQ_MODEL = "groq:llama-3.1-8b-instant"
model = init_chat_model(GROQ_MODEL)
# print(model.invoke("Please tell what is a web application firewall. Explain in 3 sentences."))

# ### Text Prompts
# Use text prompts when:
# - You have a single, standalone request
# - You don’t need conversation history
# - You want minimal code complexity


# print("\n=======  NO SQL database ========")
print(model.invoke("what is NO SQL database. Explain in 3 sentences."))


# ### Message Prompts
# Alternatively, you can pass in a list of messages to the model by providing a list of message objects.

# ### System Message
# A SystemMessage represent an initial set of instructions that primes the model’s behavior. 
# You can use a system message to set the tone, define the model’s role, and establish guidelines for responses.

# ### Human Message
# A HumanMessage represents user input and interactions. They can contain text, images, audio, files, and 
# any other amount of multimodal content.

messages = [
    SystemMessage("You are an Italian grandmother and an expert chef who loves traditional cooking."),
    HumanMessage("How do I make the perfect homemade lasagna from scratch? Please explain in 4 sentences.")
]

messages2 = [
    SystemMessage("You are an encouraging personal fitness trainer helping a complete beginner."),
    HumanMessage("Can you create a simple 10-minute morning workout routine for me? Please explain in 4 sentences.")
]


# print("\n======= Lasagna Recipe ========")
response = model.invoke(messages)
print(response.content)

# print("\n=======  Fitness trainer ========")
response2 = model.invoke(messages2)
print(response2.content)


# ==================================================
question = "How do I create a FastAPI application?"
system_msg_coding = SystemMessage("You are a helpful coding assistant.")
coding_messages = [
    system_msg_coding,
    HumanMessage(question)
]
response_coding = model.invoke(coding_messages)
print(response_coding.content)


# Detailed info to the LLM through system instructions setup
system_msg_senior = SystemMessage("""
You are a senior Python developer with expertise in web frameworks.
Always provide code examples and explain your reasoning.
Be concise but thorough in your explanations.
""")

senior_messages = [
    system_msg_senior,
    HumanMessage(question)
]
response_senior = model.invoke(senior_messages)
print(response_senior.content)


# ==================================================
# ## Message Metadata
human_msg_meta = HumanMessage(
    content="Hello!",
    name="alice",  # Optional: identify different users
    id="msg_123",  # Optional: unique identifier for tracing
)

response_meta = model.invoke([human_msg_meta])
print(response_meta)


# ==================================================
# ### AI Message
# An AIMessage represents the output of a model invocation. They can include multimodal data, tool calls, 
# and provider-specific metadata that you can later access.
# Injecting manual historic context simulation items
ai_msg_history = AIMessage("I'd be happy to help you with that question!")
conversation_history = [
    SystemMessage("You are a helpful assistant"),
    HumanMessage("Can you help me?"),
    ai_msg_history,  # Insert as if it came from the model
    HumanMessage("Great! What's 2+2?")
]

response_history = model.invoke(conversation_history)
# print(response_history.content)
# print(response_history.usage_metadata)

# ==================================================
# ### Tool Message
# An ToolMessage represents the output of a tool call. For models that support tool calling, AI messages can contain tool calls. 
# Tool messages are used to pass the results of a single tool execution back to the model.

# After a model makes a tool call execution tracking example setup
CALL_ID = "call_123"
ai_message_tool_call = AIMessage(
    content=[],
    tool_calls=[{
        "name": "get_weather",
        "args": {"location": "Tokyo"},
        "id": CALL_ID
    }]
)

weather_result = "Sunny, 72°F"
tool_message_receipt = ToolMessage(
    content=weather_result,
    tool_call_id= CALL_ID  # Must match the call ID
)

messages_with_tool_context = [
    HumanMessage("What's the weather in Tokyo?"),
    ai_message_tool_call,  # Model's tool call
    tool_message_receipt,  # Tool execution result
]
response_final_tool = model.invoke(messages_with_tool_context)
print(tool_message_receipt)
print(response_final_tool)