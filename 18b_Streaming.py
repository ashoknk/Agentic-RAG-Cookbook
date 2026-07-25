"""
================================================================================
📌 SCRIPT PURPOSE: Implementing and Evaluating LangGraph Streaming Configurations
================================================================================
This script builds a stateful chatbot ("SuperBot") utilizing LangGraph  
The script demonstrates how to view and handle the information flowing out of a chatbot application in real time, 
a concept known as streaming.  When building advanced chatbots, you don't always want to wait for the artificial intelligence 
to finish its entire thought before showing a response. 

This code showcases how to intercept the chatbot's inner workings as they happen, using two different methods: 
    Synchronous (.stream()) and Asynchronous (.astream_events()).  
  
  1. Synchronous vs. Asynchronous execution engines (.stream() vs. .astream_events())
  2. State Streaming Modes:
     - stream_mode="updates" : It shows you only what changed during the last step
     - stream_mode="values"  : It shows you the entire conversation history up to that exact moment.

    An asynchronous pipeline works like an event-driven engine that runs without blocking your main application. 
    Instead of waiting for a node to finish entirely, it broadcasts real-time diagnostic alerts (called "events") 
    the millisecond they happen behind the scenes     

    .stream() (Part 1): 
        Waits for the AI to finish its complete sentence, then prints the whole sentence out at once.  
    .astream_events() (Part 2): Listens to the internal gears of the engine. 
        It tells you exactly when a task starts (on_chain_start), hooks into the live generation, and 
        lets you print the AI's response word-by-word (token-by-token) while the AI is still thinking 
================================================================================
"""

import os
import logging
import warnings
from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv

# LangChain & Model Provider Utilities
from langchain.chat_models import init_chat_model
from langchain_core.messages import AnyMessage, HumanMessage

# LangGraph Core Framework Utilities
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
# https://docs.langchain.com/oss/python/langgraph/persistence

import asyncio
# https://docs.python.org/3/library/asyncio.html

# Suppress heavy console warning logs
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# ==============================================================================
# 1. INITIALIZATION: Environment & Keys Setup
# ==============================================================================
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# ==============================================================================
# 2. DEFINING THE REDUCER STATE SCHEMA & BOT LOGIC
# ==============================================================================
class State(TypedDict):
    # Annotated + add_messages appends turns automatically instead of overriding state keys
    messages: Annotated[list[AnyMessage], add_messages]

# Hook up our high-performance open-weight target inference engine
GROQ_MODEL = "groq:openai/gpt-oss-20b"
llm_groq=init_chat_model(model=GROQ_MODEL)


def superbot(state: State):
    """The central inference node passing active conversational history to the LLM"""
    return {"messages": [llm_groq.invoke(state['messages'])]}

# ==============================================================================
# 3. COMPILING GRAPH STATE MACHINE WITH MEMORY
# ==============================================================================
# Instantiating the in-memory checkpointer provider for conversation multi-turn tracking
# By default, an AI graph or state machine is completely stateless.MemorySaver() acts as an in-memory database.
memory = MemorySaver()

# Build the Graph canvas structure
graph = StateGraph(State)
graph.add_node("SuperBot", superbot)
graph.add_edge(START, "SuperBot")
graph.add_edge("SuperBot", END)

# Compile into an executable application, registering our short-term persistent memory
graph_builder = graph.compile(checkpointer=memory)

# Render and save pipeline diagram locally on execution
OUTPUT_IMAGE_PATH = "Image_PNGs/Streaming.png"
graph_builder.get_graph().draw_mermaid_png(output_file_path=OUTPUT_IMAGE_PATH)    
os.system(f"open {OUTPUT_IMAGE_PATH}")

# ==============================================================================
# 4. SYNCHRONOUS STREAMING UTILITIES (.stream())
# ==============================================================================
def run_sync_streaming_demo():
    # ==============================================================================
    # 4. SYNCHRONOUS STREAMING UTILITIES (.stream())
    # ==============================================================================
    print("======================================================================")
    print("🔥 PART 1: EVALUATING SYNCHRONOUS (.stream()) CHANNELS")
    print("======================================================================")

    # Establish a continuous thread block reference configuration 
    # The checkpointer (MemorySaver()) uses these thread configurations to keep distinct conversations from tangling together
    # config_thread_1 (from Part 1): Belongs to Captain Hook, who is talking about hiking, soccer, and pickleball.
    config_thread_1 = {"configurable": {"thread_id": "111"}}

    # ------------------------------------------------------------------------------
    # 💡 KNOWLEDGE CORNER: stream_mode="updates"
    # What it does: This prints out ONLY what changes inside a node function. 
    # ------------------------------------------------------------------------------
    print("\n🔹 Action A: Executing stream_mode='updates'")
    print("----------------------------------------------------------------------")
    query_1 = "Hi, My name is Captain Hook and I like to hike.(Reply in under 10 words)"
    for chunk in graph_builder.stream({'messages': [HumanMessage(content=query_1)]}, config_thread_1, stream_mode="updates"):
        if "SuperBot" in chunk:
            for msg in chunk["SuperBot"]["messages"]:
                print(f"AI Content: {msg.content}")        

    # ------------------------------------------------------------------------------
    # 💡 KNOWLEDGE CORNER: stream_mode="values"
    # What it does: Instead of an isolated update dictionary, this yields the 
    # completely compiled state history array of the entire graph up to that millisecond. 
    # ------------------------------------------------------------------------------
    print("\n🔹 Action B: Executing stream_mode='values'")
    print("----------------------------------------------------------------------")
    query_2 = "I also like soccer."
    for chunk in graph_builder.stream({'messages': [HumanMessage(content=query_2)]}, config_thread_1, stream_mode="values"):
        if "messages" in chunk:
            for msg in chunk["messages"]:
                print(f"[{msg.__class__.__name__}]: {msg.content}")
                print(f"AI Content: {msg.content}")
        
    # {'messages': [
    #   HumanMessage(content='Hi, My name is Captain Hook and I like to hike...'), 
    #   AIMessage(content='Great to meet you, Captain Hook! Happy hiking!'), 
    #   HumanMessage(content='I also like soccer.'),
    #   AIMessage(content='Nice! Do you play or just watch?')]}    
    #  It gives you the global 'messages' state key directly, and it contains every single turn of the conversation    

    print("\n🔹 Action C: Executing another stream_mode='updates' sequence on the same thread")
    print("----------------------------------------------------------------------")
    query_3 = "I also like pickleball."
    for chunk in graph_builder.stream({'messages': [HumanMessage(content=query_3)]}, config_thread_1, stream_mode="updates"):
        # Extract and print from the 'updates' chunk structure
        if "SuperBot" in chunk:
            for msg in chunk["SuperBot"]["messages"]:
                print(f"AI Content: {msg.content}")        

    # {'SuperBot': {'messages': [AIMessage(content="Cool! Pickleball's fun. What’s your favorite court?", ...)]}}
    # Notice how the output is explicitly wrapped in the name of the node execution block ('SuperBot'). 
    # Inside it, there is only one single message—the brand-new response the AI just came up with. 
    # The previous questions about hiking or soccer are completely absent from this chunk because 'updates' mode only tells you what that specific node just changed.


# ==============================================================================
# 5. ASYNCHRONOUS EVENT-DRIVEN STREAMING (.astream_events())
# ==============================================================================
# Define an asynchronous orchestration function wrapper for executing event loops
async def run_async_streaming_demo():
    print("\n======================================================================")
    print("🔥 PART 2: EVALUATING ASYNCHRONOUS (.astream_events()) PIPELINES")
    print("======================================================================")
    # config_thread_2 (from Part 2): Belongs to a completely new user named Peter Pan, who wants to talk about playing pickleball.
    config_thread_2 = {"configurable": {"thread_id": "2222"}}
    
    async_query = "Hey I am Peter Pan and I like to play pickleball. (Reply in under less than 5 sentences)"
    
    print(f"Opening event hook connection for query: '{async_query}'\n")
    
    async for event in graph_builder.astream_events({"messages": [HumanMessage(content=async_query)]}, config_thread_2, version="v2"):
        # Extract metadata identifiers for target rendering filter
        kind = event["event"]
        name = event["name"]
        
        # Filter 1: Catch when the LangGraph orchestration engine runs individual nodes
        # Prints a message to the console saying, "Hey, the robot just started thinking!"
        if kind == "on_chain_start" and name == "SuperBot":
            print(f"🎬 on_chain_start -> Node Processing Initialized: '{name}'")
            
        # Filter 2: Intercepts individual words mid-flight as the LLM generates them and instantly displays them on the screen separated by pipes (word|by|word).
        elif kind == "on_chat_model_stream":
            # print(f"💬 on_chat_model_stream -> Node Processing Concluded: '{name}'")
            content = event["data"]["chunk"].content
            if content:
                print(f"💬" +content, end="|", flush=True)
                
        # Filter 3: Catch when nodes finalize data execution
        # Prints a message saying, "The robot is done thinking now."  
        elif kind == "on_chain_end" and name == "SuperBot":
            print(f"\n🛑 on_chain_end -> Node Processing Concluded: '{name}'")

# Execute our non-blocking asynchronous event evaluation loop
if __name__ == "__main__":
    
    run_sync_streaming_demo()
    asyncio.run(run_async_streaming_demo())