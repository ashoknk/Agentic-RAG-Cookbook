"""
================================================================================
This script details the low-level architecture of LangChain tool binding, 
demonstrating what happens behind the scenes of an autonomous agent. 
Instead of relying on high-level wrappers like `create_agent`, it uncovers how 
to explicitly bind tool definitions to a raw chat model and manually coordinate 
the tool execution loop.

Without those two .append() statements, the LLM would lose all context of what tool it asked to run and 
what data was retrieved, causing the loop to break.
LLM - "I see the user asked for city weather. I see that I requested the weather tool, and
 I see that the tool responded saying it is sunny/rainy. Based on all of this context, I can now finally write back"

THE TOOL CALL TRILOGY:
----------------------
1. **Schema Binding (`bind_tools`)**: Attaches functional blueprints directly to the LLM. 
   This exposes the tool's signature (name, parameters, arguments) to the model.
   https://reference.langchain.com/python/langchain-core/language_models/chat_models/BaseChatModel/bind_tools
2. **Model Argument Generation**: The model processes a question and returns an 
   `AIMessage` containing a structured `tool_calls` dictionary instead of generic text.
   https://reference.langchain.com/python/langchain-core/messages/ai/AIMessage
3. **Manual Execution Loop**: Intercepts the generated argument dictionary, calls 
   the tool function programmatically, appends the result as a `ToolMessage`, 
   and returns the collection to the LLM for a final conversational answer.
   https://reference.langchain.com/python/langchain-core/messages/tool/ToolMessage
================================================================================
"""

import os
import random
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool

# ### Tools
# Models can request to call tools that perform tasks such as fetching data from a database, 
# searching the web, or running code. Tools are pairings of:
# 1. A schema, including the name of the tool, a description, and/or argument definitions (often a JSON schema)
# 2. A function or coroutine to execute.

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Initialize the model for streaming and batch execution
OPENAI_MODEL = "gpt-4o-mini"
model = init_chat_model(OPENAI_MODEL)

question1 ="How does a microwave cook food so fast? "
question2 ="Why do mirrors reflect our image?"
append = " Please explain in 3 sentences."

# Decorator - registers a function as an  tool that an agent or LLM can call.
@tool
def get_weather(city: str) -> str:
    """Get the weather for a city randomly."""
    weather_types = ["sunny", "rainy", "cloudy", "windy", "snowy", "stormy", "overcast"]
    selected_weather = random.choice(weather_types)
    return f"The weather in {city} is {selected_weather}."


# Binding the declared tool directly to the target chat model instance
model_with_tools = model.bind_tools([get_weather])

city1 = "London"  # (United Kingdom)
city2 = "Paris"   # (France)
city3 = "New York"  # (United States)

question1 = f"What's the weather like in {city1}?"

# ### Tool Execution Loops
# LLMs do not have "state" or memory on their own; they only know what you pass into them in the messages array

# Step 1: User Question -> LLM decides to call a tool
messages = [("user", "What's the weather like in London?")]
# Execute the tool with the generated arguments
# Run the local get_weather function using LangChain's tool.invoke(), and gets back a ToolMessage
ai_msg = model_with_tools.invoke(messages)

# By appending this ai_msg to the list, we are saving the LLM's intent to call a tool into the conversation history.
messages.append(ai_msg) 
for tool_call in ai_msg.tool_calls:
    # View tool calls made by the model
    print(f"Tool: {tool_call['name']}")
    print(f"Args: {tool_call['args']}")



# Step 2: Manually execute the requested tool and append the result
tool_call = ai_msg.tool_calls[0]
tool_result = get_weather.invoke(tool_call)
# By appending this tool_result to the list, we are saving the actual answer from the tool into the conversation history.
messages.append(tool_result)
print("Response:", tool_result)


# Step 3: Send the whole history back for the final conversational answer
# the LLM looks at the whole chain of events
# I see the user asked for London weather. I see that I requested the weather tool, and I see that the tool responded saying it is sunny. Based on all of this context, I can now finally write back
final_response = model_with_tools.invoke(messages)
print("\n========= Final Response What's the weather like in========")
print(final_response.content)

# Inspect the final assembled conversation chain structure state array
print("\n========= Final Messages ========")
print(messages) # NOTE : Just for testing 



