import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

# #### Messages
# 
# Messages are the fundamental unit of context for models in LangChain. 
# They represent the input and output of models, carrying both the content and 
# metadata needed to represent the state of a conversation when interacting with an LLM.
# Messages are objects that contain:
#  - Role - Identifies the message type (e.g. system, user)
#  - Content - Represents the actual content of the message (like text, images, audio, documents, etc.)
#  - Metadata - Optional fields such as response information, message IDs, and token usage
# 
# LangChain provides a standard message type that works across all model providers, 
# ensuring consistent behavior regardless of the model being called.

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

GROQ_MODEL = "groq:qwen/qwen3-32b"
model = init_chat_model(GROQ_MODEL)
print(model.invoke("Please tell what is a web application firewall. Explain in 3 sentences."))

# ### Text Prompts
# Text prompts are strings - ideal for straightforward generation tasks where 
# you don’t need to retain conversation history.
print(model.invoke("what is NO SQL database. Explain in 3 sentences."))

# Use text prompts when:
# - You have a single, standalone request
# - You don’t need conversation history
# - You want minimal code complexity

# ### Message Prompts
# Alternatively, you can pass in a list of messages to the model by providing a list of message objects.

# ### System Message
# A SystemMessage represent an initial set of instructions that primes the model’s behavior. 
# You can use a system message to set the tone, define the model’s role, and establish guidelines for responses.

# ### Human Message
# A HumanMessage represents user input and interactions. They can contain text, images, audio, files, and 
# any other amount of multimodal content.

# ### AI Message
# An AIMessage represents the output of a model invocation. They can include multimodal data, tool calls, 
# and provider-specific metadata that you can later access.

# ### Tool Message
# An ToolMessage represents the output of a tool call. For models that support tool calling, AI messages can contain tool calls. 
# Tool messages are used to pass the results of a single tool execution back to the model.

# messages = [
#     SystemMessage("You are a poetry expert"),
#     HumanMessage("Write a poem on artificial intelligence")
# ]

#TODO add beter print statements with ====
messages = [
    SystemMessage("You are an Italian grandmother and an expert chef who loves traditional cooking."),
    HumanMessage("How do I make the perfect homemade lasagna from scratch? Please explain in 10 sentences.")
]

messages2 = [
    SystemMessage("You are an encouraging personal fitness trainer helping a complete beginner."),
    HumanMessage("Can you create a simple 10-minute morning workout routine for me? Please explain in 10 sentences.")
]

response = model.invoke(messages)
print("\n======= Lasagna Recipe ========")
print(response.content)

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

# ## Message Metadata
human_msg_meta = HumanMessage(
    content="Hello!",
    name="alice",  # Optional: identify different users
    id="msg_123",  # Optional: unique identifier for tracing
)

response_meta = model.invoke([human_msg_meta])
print(response_meta)

# Injecting manual historic context simulation items
ai_msg_history = AIMessage("I'd be happy to help you with that question!")
conversation_history = [
    SystemMessage("You are a helpful assistant"),
    HumanMessage("Can you help me?"),
    ai_msg_history,  # Insert as if it came from the model
    HumanMessage("Great! What's 2+2?")
]

response_history = model.invoke(conversation_history)
print(response_history.content)
print(response_history.usage_metadata)

# After a model makes a tool call execution tracking example setup
ai_message_tool_call = AIMessage(
    content=[],
    tool_calls=[{
        "name": "get_weather",
        "args": {"location": "Tokyo"},
        "id": "call_123"
    }]
)

weather_result = "Sunny, 72°F"
tool_message_receipt = ToolMessage(
    content=weather_result,
    tool_call_id="call_123"  # Must match the call ID
)

messages_with_tool_context = [
    HumanMessage("What's the weather in Tokyo?"),
    ai_message_tool_call,  # Model's tool call
    tool_message_receipt,  # Tool execution result
]
response_final_tool = model.invoke(messages_with_tool_context)
print(tool_message_receipt)
print(response_final_tool)