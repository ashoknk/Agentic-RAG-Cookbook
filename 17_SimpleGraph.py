# ### Build a Simple Workflow or Graph Using LangGraph
# #### State
# First, define the State of the graph.
# The State schema serves as the input schema for all Nodes and Edges in the graph.
# Let's use the TypedDict class from python's typing module as our schema, 
# which provides type hints for the keys.
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
# The first positional argument is the state, as defined above.
# Because the state is a TypedDict with schema as defined above, each node can access the key, 
# graph_state, with state['graph_state'].
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
graph.add_node("start_play",start_play)
graph.add_node("soccer",soccer)
graph.add_node("pickelball",pickelball)

## Schedule the flow of the graph
graph.add_edge(START,"start_play")
graph.add_conditional_edges("start_play",random_play)
graph.add_edge("soccer",END)
graph.add_edge("pickelball",END)

## Compile the graph
graph_builder=graph.compile()

## NOTE: View for Jupyter notebook
# display(Image(graph_builder.get_graph().draw_mermaid_png()))


# 1. Save the file as a PNG
OUTPUT_IMAGE_PATH = "Image_PNGs/SimpleGraph.png"
graph_builder.get_graph().draw_mermaid_png(output_file_path=OUTPUT_IMAGE_PATH)    
# 2. Automatically display/open the image on macOS
os.system(f"open {OUTPUT_IMAGE_PATH}")

### GRAPH Invocation
# Capture the final state dictionary returned by the workflow
final_state = graph_builder.invoke({"graph_info": "Hey! My name is Peter Pan and I like to play"})

print("\n--- Final Output ---")
# Print the updated string from the 'graph_info' key
print(final_state["graph_info"])