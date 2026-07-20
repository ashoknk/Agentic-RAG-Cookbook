"""
================================================================================
This script applies the LangGraph state machine workflow to construct an 
interactive conversational AI agent chatbot. It acts as a direct technical 
demonstration of how state frameworks are iteratively passed along to maintain 
contextual conversation threads.

THE VALUE OF ANNOTATED REDUCERS:
--------------------------------
By default, whenever a node finishes processing in LangGraph, it overrides key-value 
pairs inside the global state. To prevent losing conversational context, this script 
introduces an **Annotated Reducer (`add_messages`)**. This tells LangGraph to *append* 
incoming messages to a cumulative history list instead of replacing the entire key.

By wrapping the list with `Annotated[list, add_messages]`, you are defining a Reducer 
function (`add_messages`) for that specific state key:
   - **Annotated**: A Python typing mechanism that allows you to attach metadata to 
     a variable type. In LangGraph, it binds a specific update rule (the reducer) 
     to a state key.
   - **add_messages**: The built-in LangGraph reducer function. Instead of replacing 
     the existing array, it instructs LangGraph to append new messages to the 
     existing list, accumulating the conversation history over multiple turns.

CODE EXECUTION FLOW:
--------------------
1. STATE PREPARATION: Configures a `State` class where the `messages` attribute is 
   explicitly wrapped inside the `add_messages` reducer function to preserve context.
2. CHAT BOT NODE (`superbot`): Implements a single executor node that captures the 
   historical message array and submits it to a Groq LLM instance.
3. WORKFLOW WIRING: Connects the `START` node to the bot executor node and ends 
   at the terminal `END` state.
4. RUNTIME MEMORY DEMONSTRATION: Compiles the pipeline and issues an initial 
   introduction prompt constrained by brevity guidelines, followed immediately by a 
   dependent query confirming that historical state values have been accurately retained.
================================================================================
"""

import os
from dotenv import load_dotenv
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

## Reducers
from typing import Annotated
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

load_dotenv()
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")

class State(TypedDict):
    messages:Annotated[list,add_messages]


GROQ_MODEL = "groq:openai/gpt-oss-20b"
llm_groq=init_chat_model(model=GROQ_MODEL)
# Change your original QUERY string to this:
QUERY = "Hey I am Peter Pan and I like to play pickleball. (Reply in under 10 words)"

### We Will start With Creating Nodes
# The primary benefit of using Annotated and add_messages in this code is to control how LangGraph manages 
# the conversation history within the global State.
def superbot(state:State):
    return {"messages":[llm_groq.invoke(state['messages'])]}

graph=StateGraph(State)

## node
graph.add_node("SuperBot",superbot)

## Edges
graph.add_edge(START,"SuperBot")
graph.add_edge("SuperBot",END)

graph_builder=graph.compile()

OUTPUT_IMAGE_FOLDER = "Image_PNGs"
os.makedirs(OUTPUT_IMAGE_FOLDER, exist_ok=True)
OUTPUT_IMAGE_PATH = OUTPUT_IMAGE_FOLDER + "/Chatbot.png"
graph_builder.get_graph().draw_mermaid_png(output_file_path=OUTPUT_IMAGE_PATH)    
# 2. Automatically display/open the image on macOS
os.system(f"open {OUTPUT_IMAGE_PATH}")


print("=" * 50)
print("==== Demonstrating Memory ====")
print("=" * 50)

# 1. First turn: Tell the bot something specific
state = graph_builder.invoke({'messages': QUERY})
print(f"Bot Response 1: {state['messages'][-1].content}")

# 2. Second turn: Ask a follow-up question relying on memory
# We pass the full history (state['messages']) so the bot sees the accumulated state
state['messages'].append(HumanMessage(content="What was my name again and what sport do I like?"))
final_state = graph_builder.invoke({'messages': state['messages']})

print(f"Bot Response 2: {final_state['messages'][-1].content}")    