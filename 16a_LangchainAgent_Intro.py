"""
================================================================================
This script acts as an entry-level laboratory introduction to the core design patterns 
of **AI Agents** within modern LangChain. It showcases how an LLM can move 
beyond passive text generation to orchestrate execution flows, evaluating user prompts 
and autonomously choosing when to invoke pre-registered external python tools.

1. TOOL REGISTRATION: Declares a simple python native mock weather lookup function.
2. RUNTIME COMPILATION: Uses `create_agent` to wrap an LLM alongside the tool array.
3. AGENT INTERACTION: Dispatches a prompt requiring real-world external insight.
4. MESSAGE CHAIN ANALYSIS: Iterates through and unpacks the execution state array, 
   tracing how the agent creates `HumanMessage`, `AIMessage` (containing tool call requests), 
   and final grounded `ToolMessage` payloads.
================================================================================
"""

import os
import random
from dotenv import load_dotenv
import langchain
from langchain.agents import create_agent

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

def get_weather(city: str) -> str:
    """Get the weather for a city randomly."""
    weather_types = ["sunny", "rainy", "cloudy", "windy", "snowy", "stormy", "overcast"]
    selected_weather = random.choice(weather_types)
    return f"The weather in {city} is {selected_weather}."

# Initialize the agent
LLM_MODEL = "gpt-4o-mini"
# https://reference.langchain.com/python/langchain/agents/factory/create_agent
# `create_agent` Creates an agent graph that calls tools in a loop until a stopping condition is met.
agent = create_agent(
    model=LLM_MODEL,
    tools=[get_weather],
    system_prompt="You are a helpful assistant."
)
# print("Agent initialized:", agent) # NOTE: Just for debugging purposes


# ### ========= Run the Agent ========= ### 

# NOTE: Just for testing
# Option 1: Ideal for quick, single-turn testing scripts.
# agent_response1 = agent.invoke({"messages": "What is the weather in Tokyo?"})
# print("\nagent 1",agent_response1)
# exit()

# Option 2:  Modern LangChain agents manage conversation history using an explicit state array of message objects
agent_response2 = agent.invoke({
    "messages": [{"role": "user", "content": "What is the weather like in New York?"}]
})

# NOTE: Just for debugging purposes - Inspect message history output
# print("\n agent 2",agent_response2["messages"])


print("\n---  Agent Messages ---")
for msg in agent_response2.get("messages", []):
    # Check if it's a HumanMessage, ToolMessage or AIMessage
    if msg.__class__.__name__ in ["HumanMessage", "ToolMessage", "AIMessage"]:
        
        # 1. Extract the class type name and the text content
        # __class__.__name__ is used to retrieve the name of an object's class as a plain string.
        msg_type = msg.__class__.__name__
        content = msg.content
        
        # 2. Extract metadata safely (fall back to 'Unknown' if missing)
        metadata = getattr(msg, "response_metadata", {})
        provider = metadata.get("model_provider", "openai")
        model_name = metadata.get("model_name", LLM_MODEL)

        # 3. Build the conditional display based on the type
        # A HumanMessage is a message that is passed in from a user to the model.
        if msg_type == "HumanMessage":
            print(f"{msg_type}(content='{content}')")
            print(f". model_provider': '{provider}', 'model_name': '{model_name}'")
            
        # An AIMessage is returned from a chat model as a response to a prompt.
        # Represents the output of the model and consists of both the raw output 
        # as returned by the model and standardized fields (e.g., tool calls, usage metadata) added by LangChain .
        if msg_type == "AIMessage":    
            print(f"{msg_type}(content='{content}')")
            print(f". model_provider': '{provider}', 'model_name': '{model_name}'") 
        
        # ToolMessage objects contain the result of a tool invocation. Typically, the result is encoded inside the content field.
        if msg_type == "ToolMessage":
            # Include the tool name identifier for ToolMessages
            tool_name = getattr(msg, "name", "get_weather")
            print(f"{msg_type}(content='{content}', name='{tool_name}'")
            print(f". model_provider': '{provider}', 'model_name': '{model_name}'")
        
        # NOTE - For testing 
        # Extract the main Message ID
        message_id = getattr(msg, "id", "N/A")
        
        # Extract Tool Call ID (Only exists on ToolMessages)
        tool_call_id = getattr(msg, "tool_call_id", "N/A")
        
        # Extract System Fingerprint (Saves inside response_metadata for AIMessages)
        metadata = getattr(msg, "response_metadata", {})
        system_fingerprint = metadata.get("system_fingerprint", "N/A")
        
        # === Print them out to inspect ===
        print(f"   ↳ [ID]: {message_id}")
        if msg_type == "ToolMessage":
            print(f"   ↳ [Tool Call ID]: {tool_call_id}")
        if system_fingerprint != "N/A":
            print(f"   ↳ [System Fingerprint]: {system_fingerprint}")            

