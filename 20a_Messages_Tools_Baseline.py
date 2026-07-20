"""
Demonstrate the default "State Override Problem" in LangGraph.
When a Reducer is NOT used, returning new state updates overwrites 
the existing state key instead of appending to it.
"""
import os
from IPython.display import Image,display
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END

# 1. Define State WITHOUT a Reducer
class PlainState(TypedDict):
    messages: list  # Pure list type hint, no reducer annotation!

# 2. Node Functions
def step_1(state: PlainState):
    print("--- STEP 1 ---")
    print(f"Current State: {state['messages']}")
    # Returns a new message
    return {"messages": [AIMessage(content="Hello! How can I help?")]}

def step_2(state: PlainState):
    print("\n--- STEP 2 ---")
    print(f"Current State: {state['messages']}")
    # Returns another message
    return {"messages": [HumanMessage(content="I want to learn Python.")]}

# 3. Build & Compile Graph
builder = StateGraph(PlainState)
builder.add_node("step_1", step_1)
builder.add_node("step_2", step_2)

builder.add_edge(START, "step_1")
builder.add_edge("step_1", "step_2")
builder.add_edge("step_2", END)

graph = builder.compile()

# Save the file as a PNG
OUTPUT_IMAGE_FOLDER = "Image_PNGs"
os.makedirs(OUTPUT_IMAGE_FOLDER, exist_ok=True)

OUTPUT_IMAGE_PATH = OUTPUT_IMAGE_FOLDER + "/20a_Messages_Tools_Baseline.png"
graph.get_graph().draw_mermaid_png(output_file_path=OUTPUT_IMAGE_PATH)    
# 2. Automatically display/open the image on macOS
os.system(f"open {OUTPUT_IMAGE_PATH}")


# 4. Run Execution
print("========== WITHOUT REDUCER (OVERRIDE BEHAVIOR) ==========\n")
initial_input = {"messages": [HumanMessage(content="Hi, This is Harry Potter and I like magic")]}
final_state = graph.invoke(initial_input)

print("\n========== FINAL RESULT ==========")
print(f"Final State Messages: {final_state['messages']}")
print("\n❌ PROBLEM: We lost 'Hi...' and 'Hello! How can I help?' because each node OVERWROTE the list!")