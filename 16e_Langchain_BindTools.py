"""
================================================================================
This script details the low-level architecture of LangChain tool binding, 
demonstrating what happens behind the scenes of an autonomous agent. 
Instead of relying on high-level wrappers like `create_agent`, it uncovers how 
to explicitly bind tool definitions to a raw chat model and manually coordinate 
the tool execution loop.

THE TOOL CALL TRILOGY:
----------------------
1. **Schema Binding (`bind_tools`)**: Attaches functional blueprints directly to the LLM. 
   This exposes the tool's signature (name, parameters, arguments) to the model.
2. **Model Argument Generation**: The model processes a question and returns an 
   `AIMessage` containing a structured `tool_calls` dictionary instead of generic text.
3. **Manual Execution Loop**: Intercepts the generated argument dictionary, calls 
   the tool function programmatically, appends the result as a `ToolMessage`, 
   and returns the collection to the LLM for a final conversational answer.
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

question1 ="How does a microwave cook food so fast? "
question2 ="Why do mirrors reflect our image?"
append = " Please explain in 3 sentences."


GROQ_MODEL = "groq:qwen/qwen3-32b"
model = init_chat_model(GROQ_MODEL)
# response = model.invoke(question1 + append)
# print("\n=======Question 1:========", question1)
# print("Response:", response)

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
city4 = "Sydney"  # (Australia)
city5 = "Cairo"   # (Egypt)

question3 = f"What's the weather like in {city1}?"
print("\n=======Question 3:========", question3)

response_tool = model_with_tools.invoke(question3)
print("\n========= What's the weather like in {city1}?")
#Look for tool_calls': [{'id': 'abc', 'function': {'arguments': '{"city":"London"}', 'name': 'get_weather'}
print("Response:", response_tool)


for tool_call in response_tool.tool_calls:
    # View tool calls made by the model
    print(f"Tool: {tool_call['name']}")
    print(f"Args: {tool_call['args']}")

# ### Tool Execution Loops

# Step 1: Model generates tool calls
# question4 = f"What's the weather like in {city2}?"
messages = [{"role": "user", "content": question3}]
ai_msg = model_with_tools.invoke(messages)
messages.append(ai_msg)

# Step 2: Execute tools and collect results
for tool_call in ai_msg.tool_calls:
    # Execute the tool with the generated arguments
    tool_result = get_weather.invoke(tool_call)
    messages.append(tool_result)

# Step 3: Pass results back to model for final response
# Look for ToolMessage(content='The weather in London is abc.', name='get_weather', tool_call_id='xyz')]
final_response = model_with_tools.invoke(messages)
print("\n========= Final Response What's the weather like in========", city1)
print(final_response.text)

# Inspect the final assembled conversation chain structure state array
print("\n========= Final Messages ========")
print(messages)