import os
import requests
import langchain
from dotenv import load_dotenv
from langchain.agents import create_agent

#TODO put some basic langchain setup here from langchain docs
print(langchain.__version__)  # Output: '1.1.0'

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


def get_weather(city: str) -> str:
    """Get the live current weather parameters for a specific city."""
    try:
        # Step 1: Use Open-Meteo's free geocoding API to resolve city to lat/long coordinates
        geocoding_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json"
        geo_response = requests.get(geocoding_url, timeout=10)
        geo_response.raise_for_status()
        geo_data = geo_response.json()
        
        if not geo_data.get("results"):
            return f"Could not find coordinates for the city: {city}."
            
        location = geo_data["results"][0]
        lat = location["latitude"]
        lon = location["longitude"]
        city_name = location["name"]
        country = location.get("country", "")

        # Step 2: Fetch current weather metrics using coordinates
        weather_url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation"
        }
        
        weather_response = requests.get(weather_url, params=params, timeout=10)
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
agent = create_agent(
    model="gpt-5",
    tools=[get_weather],
    system_prompt="You are a helpful assistant."
)
print("Agent initialized:", agent)

# Run the agent
agent_response = agent.invoke({"messages": "What is the weather in New York?"})

print("\n---  Agent Messages ---")
for msg in agent_response.get("messages", []):
    if msg.__class__.__name__ in ["HumanMessage", "ToolMessage"]:
        msg_type = msg.__class__.__name__
        content = msg.content
        
        metadata = getattr(msg, "response_metadata", {})
        provider = metadata.get("model_provider", "openai")
        model_name = metadata.get("model_name", "gpt-5-2025-08-07")
        
        if msg_type == "HumanMessage":
            print(f"{msg_type}(content='{content}')")
            print(f". model_provider': '{provider}', 'model_name': '{model_name}'")
            
        if msg_type == "ToolMessage":
            tool_name = getattr(msg, "name", "get_weather")
            print(f"{msg_type}(content='{content}', name='{tool_name}')")
            print(f". model_provider': '{provider}', 'model_name': '{model_name}'")