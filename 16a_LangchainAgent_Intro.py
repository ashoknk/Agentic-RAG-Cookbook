import os
import random
from dotenv import load_dotenv
import langchain
from langchain.agents import create_agent


# View LangChain version
print(langchain.__version__)  # Output: '1.1.0'

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


def get_weather(city: str) -> str:
    """Get the weather for a city randomly."""
    weather_types = ["sunny", "rainy", "cloudy", "windy", "snowy", "stormy", "overcast"]
    selected_weather = random.choice(weather_types)
    return f"The weather in {city} is {selected_weather}."

# Initialize the agent
agent = create_agent(
    model="gpt-5",
    tools=[get_weather],
    system_prompt="You are a helpful assistant."
)
print("Agent initialized:", agent)

# ### run the agent
response = agent.invoke({
    "messages": [{"role": "user", "content": "What is the weather like in New York?"}]
})

# NOTE: Just for debugging purposes - Inspect message history output
# print(response["messages"])

# Test alternative string invocation syntax
agent_response = agent.invoke({"messages": "What is the weather in Tokyo?"})
# print(agent_response)

# --- Replace your standard print(agent_response) with this block ---

print("\n---  Agent Messages ---")
for msg in agent_response.get("messages", []):
    # Check if it's a HumanMessage or a ToolMessage
    if msg.__class__.__name__ in ["HumanMessage", "ToolMessage"]:
        
        # 1. Extract the class type name and the text content
        msg_type = msg.__class__.__name__
        content = msg.content
        
        # 2. Extract metadata safely (fall back to 'Unknown' if missing)
        metadata = getattr(msg, "response_metadata", {})
        provider = metadata.get("model_provider", "openai")
        model_name = metadata.get("model_name", "gpt-5-2025-08-07")
        #TODO getattr(msg, "usage_metadata", {})
        
        # 3. Build the conditional display based on the type
        if msg_type == "HumanMessage":
            print(f"{msg_type}(content='{content}')")
            print(f". model_provider': '{provider}', 'model_name': '{model_name}'")
            
        if msg_type == "AIMessage":    
            print(f"{msg_type}(content='{content}')")
            print(f". model_provider': '{provider}', 'model_name': '{model_name}'") 

        if msg_type == "ToolMessage":
            # Include the tool name identifier for ToolMessages
            tool_name = getattr(msg, "name", "get_weather")
            print(f"{msg_type}(content='{content}', name='{tool_name}'")
            print(f". model_provider': '{provider}', 'model_name': '{model_name}'")

# NOTE : too much detailed information. use what you can in above code.
# print("\n--- 📋 Detailed Agent Message History ---")
# for msg in agent_response.get("messages", []):
#     msg_type = msg.__class__.__name__
    
#     # 1. Base Core Fields
#     msg_id = getattr(msg, "id", "N/A")
#     content = msg.content if msg.content else "[No text content - Tool Call Init]"
    
#     # 2. Extract Metadata & Token Usage Safely
#     response_metadata = getattr(msg, "response_metadata", {})
#     provider = response_metadata.get("model_provider", "Unknown Provider")
#     model_name = response_metadata.get("model_name", "Unknown Model")
    
#     # Access usage metadata from LangChain unified attribute or fall back to response_metadata
#     usage_metadata = getattr(msg, "usage_metadata", {})
#     if not usage_metadata:
#         usage_metadata = response_metadata.get("token_usage", {})
        
#     input_tokens = usage_metadata.get("input_tokens", usage_metadata.get("prompt_tokens", 0))
#     output_tokens = usage_metadata.get("output_tokens", usage_metadata.get("completion_tokens", 0))
#     total_tokens = usage_metadata.get("total_tokens", usage_metadata.get("total_tokens", 0))

#     # --- Conditional Block formatting by Message Type ---
    
#     if msg_type == "HumanMessage":
#         print(f"👤 \033[1;34m{msg_type}\033[0m | ID: {msg_id}")
#         print(f"   ↳ Prompt: \"{content}\"")
#         print("-" * 50)

#     elif msg_type == "AIMessage":
#         print(f"🤖 \033[1;32m{msg_type}\033[0m | ID: {msg_id}")
#         print(f"   ↳ Engine: {model_name} ({provider})")
        
#         # Check if the AI wants to call a tool
#         tool_calls = getattr(msg, "tool_calls", [])
#         if tool_calls:
#             for call in tool_calls:
#                 print(f"   ⚙️  \033[1;33mAction Request:\033[0m Invoking tool `{call['name']}` with arguments: {call['args']}")
#         else:
#             print(f"   ↳ Text Response: \"{content}\"")
            
#         print(f"   📊 Token Usage -> Input: {input_tokens} | Output: {output_tokens} | Total: {total_tokens}")
#         print("-" * 50)

#     elif msg_type == "ToolMessage":
#         tool_name = getattr(msg, "name", "Unknown Tool")
#         tool_call_id = getattr(msg, "tool_call_id", "N/A")
#         print(f"🛠️  \033[1;35m{msg_type}\033[0m | ID: {msg_id}")
#         print(f"   ↳ Executed Tool: `{tool_name}` (Call ID Reference: {tool_call_id})")
#         print(f"   ↳ Raw Output Payload: \"{content}\"")
#         print("-" * 50)            