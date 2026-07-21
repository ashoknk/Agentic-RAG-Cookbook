"""
================================================================================
SCRIPT PURPOSE & OVERVIEW:
--------------------------
This script serves as a foundational hello-world lab for **LangGraph**, 
introducing how to create a non-linear stateful execution workflow. 
It breaks down the core structural units of any graph pipeline: 
States, Nodes, Edges, and Conditional Routers.


THE CORE MECHANICS:
-------------------
- State: A unified dictionary that acts as the single source of 
  truth passing through every stage of the lifecycle.
- Nodes: Standard Python functions that process the state and return incremental updates to it.
- Edges: Paths connecting nodes together to establish execution sequence.
- Conditional Edges: Decision-making routers that dynamically compute the next 
  destination based on the state payload at runtime.
  https://docs.langchain.com/oss/python/langgraph/graph-api

THE LOGICAL FLOW:
-----------------
1. SCHEMA DEFINITION: Configures a primitive `State` tracking a `graph_info` string.
2. NODE COMPILATION: Implements execution nodes (`start_play`, `soccer`, `pickelball`) 
   and a random conditional routing decision function (`random_play`).
3. GRAPH CONSTRUCTION: Instantiates `StateGraph`, registers the functional nodes, 
   sets up hard edges, and maps out conditional routing paths.
4. EXECUTION ENGINES: Compiles the pipeline, exports a structural graph layout image 
   via Mermaid, and invokes the workflow to track mutations over the string data.
================================================================================
"""

import os
import random
from typing import Literal
from typing_extensions import TypedDict
from IPython.display import Image,display
from langgraph.graph import StateGraph,START,END


class State(TypedDict):
    graph_info:str

#### ======= Nodes ======= ####
# Nodes are just python functions.
# Because the state is a TypedDict with schema as defined above, each node can access the key, with state['graph_state'].
# Each node returns a new value of the state key graph_state.
# By default, the new value returned by each node will override the prior state value.

GAME1 = "soccer"
GAME2 = "pickelball"

def start_play(state: State):
    print("Start_Play node has been called")
    return {"graph_info": state['graph_info']}

def soccer(state: State):
    print(f"My {GAME1} node has been called")
    return {"graph_info": state['graph_info'] + f" {GAME1}"}

def pickelball(state: State):
    print(f"My {GAME2} node has been called")
    return {"graph_info": state['graph_info'] + f" {GAME2}"}

def random_play(state:State)-> Literal['soccer','pickelball']:
    graph_info=state['graph_info']

    if random.random()>0.5:
        return "soccer"
    else:
        return "pickelball"
    

# #### ==== Graph Construction ==== ####
# Now, we build the graph from our components defined above.
# The StateGraph class is the graph class that we can use.
# First, we initialize a StateGraph with the State class we defined above.
# Then, we add our nodes and edges.

# We use the START Node, a special node that sends user input to the graph, 
# to indicate where to start our graph.
# The END Node is a special node that represents a terminal node.
# Finally, we compile our graph to perform a few basic checks on the graph structure.
# We can visualize the graph as a Mermaid diagram.    

## Build Graph
graph=StateGraph(State)

## Adding the nodes
# https://reference.langchain.com/python/langgraph/graph/state/StateGraph/add_node
graph.add_node("start_play",start_play)
graph.add_node("soccer",soccer)
graph.add_node("pickelball",pickelball)

## Schedule the flow of the graph
# https://reference.langchain.com/python/langgraph/graph/state/StateGraph/add_edge
# https://reference.langchain.com/python/langgraph/graph/state/StateGraph/add_conditional_edges
graph.add_edge(START,"start_play")
graph.add_conditional_edges("start_play",random_play)
graph.add_edge("soccer",END)
graph.add_edge("pickelball",END)

## Compile the graph
# https://reference.langchain.com/python/langgraph/graph/state/StateGraph/compile
graph_builder=graph.compile()

# Save the file as a PNG
OUTPUT_IMAGE_FOLDER = "Image_PNGs"
os.makedirs(OUTPUT_IMAGE_FOLDER, exist_ok=True)

OUTPUT_IMAGE_PATH = OUTPUT_IMAGE_FOLDER + "/SimpleGraph.png"
graph_builder.get_graph().draw_mermaid_png(output_file_path=OUTPUT_IMAGE_PATH)    
# 2. Automatically display/open the image on macOS
os.system(f"open {OUTPUT_IMAGE_PATH}")

### GRAPH Invocation
# Capture the final state dictionary returned by the workflow
final_state = graph_builder.invoke({"graph_info": "Hey! My name is Peter Pan and I like to play"})

print("\n--- Final Output ---")
# Print the updated string from the 'graph_info' key
print(final_state["graph_info"])