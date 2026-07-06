### Implementing simple Chatbot Using LangGraph

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END

## Reducers
from typing import Annotated
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages:Annotated[list,add_messages]


import os
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")

from langchain_openai import ChatOpenAI
# OPENAI_MODEL_NAME="gpt-4o"
OPENAI_MODEL = "gpt-5.4-mini"
llm=ChatOpenAI(model=OPENAI_MODEL)
llm.invoke("Hello")

from langchain_groq import ChatGroq

GROQ_MODEL="qwen/qwen3-32b"
# qwen/qwen3-32b
llm_groq=ChatGroq(model=GROQ_MODEL)
llm_groq.invoke("Hey I am Ash and i like to play pickeball")

### We Will start With Creating Nodes

def superbot(state:State):
    return {"messages":[llm_groq.invoke(state['messages'])]}

graph=StateGraph(State)

## node
graph.add_node("SuperBot",superbot)
## Edges

graph.add_edge(START,"SuperBot")
graph.add_edge("SuperBot",END)


graph_builder=graph.compile()


## NOTE: View for Jupyter notebook
# from IPython.display import Image, display
# display(Image(graph_builder.get_graph().draw_mermaid_png()))

OUTPUT_IMAGE_PATH = "Image_PNGs/Chatbot.png"
graph_builder.get_graph().draw_mermaid_png(output_file_path=OUTPUT_IMAGE_PATH)    
# 2. Automatically display/open the image on macOS
os.system(f"open {OUTPUT_IMAGE_PATH}")

## Invocation
graph_builder.invoke({'messages':"Hi,My name is Ash And I like to hike"})

#### Streaming The responses
for event in graph_builder.stream({"messages":"Hello My name is Ash"}):
    print(event)