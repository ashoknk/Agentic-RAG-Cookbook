"""
================================================================================
This script upgrades the core agent pattern introduced in `16a` by shifting from a 
predictable mock function to a live, multi-step production weather lookup utility. 
It highlights the ability of AI agents to coordinate real-world web requests and 
process dynamic JSON responses over live endpoints.

1. LIVE API ARCHITECTURE (`get_weather`): Constructs a two-tiered network call utility. 
   First, it reaches out to a Geocoding API to resolve a city name string into geographic 
   coordinates (latitude and longitude). Second, it invokes the Open-Meteo 
   Forecast API using those coordinates to fetch live atmospheric metrics.
2. AGENT BINDING: Hooks the function into `create_agent`.
3. SYSTEM QUERY: Executes runtime user queries. The agent automatically handles parameter 
   extraction, launches the sequence, parses the live dataset, and returns a clean answer.
================================================================================
"""

import os
import requests
import langchain
from dotenv import load_dotenv
from langchain.agents import create_agent


load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


def get_weather(city: str) -> str:
    """Get the live current weather parameters for a specific city."""
    try:
        # Step 1: Use Open-Meteo's free geocoding API to resolve city to lat/long coordinates
        # https://open-meteo.com/en/docs/geocoding-api?name=Tokio&language=pt
        geocoding_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json"
        geo_response = requests.get(geocoding_url, timeout=10)
        # throw a HTTPError exception if an HTTP request fails with a client error (4xx) or server error (5xx)
        geo_response.raise_for_status()
        geo_data = geo_response.json()
        
        if not geo_data.get("results"):
            return f"Could not find coordinates for the city: {city}."

        #Get latitude, longitude, city_name, country
        location = geo_data["results"][0]
        lat = location["latitude"]
        lon = location["longitude"]
        city_name = location["name"]
        country = location.get("country", "")

        # Step 2: Fetch current weather metrics using coordinates
        # air temperature measured at 2 meters, amount of moisture in the air, "feels like" temperature,amount of water
        weather_url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation"
        }
        
        weather_response = requests.get(weather_url, params=params, timeout=10)
        # throw a HTTPError exception if an HTTP request fails with a client error (4xx) or server error (5xx)
        weather_response.raise_for_status()
        weather_data = weather_response.json()
        current = weather_data.get("current", {})
        
        temp = current.get("temperature_2m", "N/A")
        feels_like = current.get("apparent_temperature", "N/A")
        humidity = current.get("relative_humidity_2m", "N/A")
        precipitation = current.get("precipitation", "N/A")

        return (
            f"The current weather in {city_name}, {country} is {temp}°C. "
            f"It feels like {feels_like}°C with a relative humidity of {humidity}% "
            f"and {precipitation}mm of precipitation."
        )
    except Exception as e:
        return f"Error retrieving live weather data for {city}: {str(e)}"


# Initialize the agent
LLM_MODEL = "gpt-4o-mini"
agent = create_agent(
    model=LLM_MODEL,
    tools=[get_weather],
    system_prompt="You are a helpful assistant."
)

# Run the agent
# Modern LangChain agents manage conversation history using an explicit state array of message objects
agent_response = agent.invoke({
    "messages": [{"role": "user", "content": "What is the weather like in New York?"}]
})

# NOTE: Just for debugging purposes - Inspect message history output
# print("\n agent 2",agent_response["messages"])


print("\n---  Agent Messages ---")
for msg in agent_response.get("messages", []):
    if msg.__class__.__name__ in ["HumanMessage", "ToolMessage"]:
        msg_type = msg.__class__.__name__
        content = msg.content
        
        metadata = getattr(msg, "response_metadata", {})
        provider = metadata.get("model_provider", "openai")
        model_name = metadata.get("model_name", LLM_MODEL)
        
        if msg_type == "HumanMessage":
            print(f"{msg_type}(content='{content}')")
            print(f". model_provider': '{provider}', 'model_name': '{model_name}'")
            
        if msg_type == "ToolMessage":
            tool_name = getattr(msg, "name", "get_weather")
            print(f"{msg_type}(content='{content}', name='{tool_name}')")
            print(f". model_provider': '{provider}', 'model_name': '{model_name}'")

        # # NOTE - For testing 
        # # Extract the main Message ID
        # message_id = getattr(msg, "id", "N/A")
        
        # # Extract Tool Call ID (Only exists on ToolMessages)
        # tool_call_id = getattr(msg, "tool_call_id", "N/A")
        
        # # Extract System Fingerprint (Saves inside response_metadata for AIMessages)
        # metadata = getattr(msg, "response_metadata", {})
        # system_fingerprint = metadata.get("system_fingerprint", "N/A")
        
        # # === Print them out to inspect ===
        # print(f"   ↳ [ID]: {message_id}")
        # if msg_type == "ToolMessage":
        #     print(f"   ↳ [Tool Call ID]: {tool_call_id}")
        # if system_fingerprint != "N/A":
        #     print(f"   ↳ [System Fingerprint]: {system_fingerprint}")            
