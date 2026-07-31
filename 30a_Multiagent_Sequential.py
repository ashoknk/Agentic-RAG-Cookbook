"""
================================================================================
This script implements a sequential, state-passing Multi-Agent RAG system using LangGraph 
It orchestrates two specialized agents 
    (a Researcher and a Blog Writer)  that work in a linear pipeline. 
    1.The Researcher gathers information using web search and internal tools, 
    2.and then passes the collected state directly to the Blog Writer to generate a detailed blog post.

🔄 GRAPH EXECUTION FLOW & NODE INTERACTIONS:
1. START ──> `research_node`:
   - Invokes `research_agent` (ReAct agent with `CyberSecurityResearchNotes` 
     & `TavilySearch`).
   - Gathers data and appends findings to `MessagesState`.
   - Uses `get_next_node()` to evaluate the agent's output:
       * If response contains "FINAL ANSWER" ──> Route to END.
       * Otherwise ──> Route to `blog_generator` node via `Command(goto=...)`.

    Edge Case: If the user query is simple or straightforward (or if the LLM feels it has completely answered 
    the prompt during the research step), research_agent may prefix its final output string with FINAL ANSWER       

2. `research_node` ──> `blog_node`:
   - Receives state containing the accumulated research history.
   - Invokes `blog_agent` to synthesize the research into a detailed blog post.
   - Uses `get_next_node()` to evaluate the writer's output:
       * If blog contains "FINAL ANSWER" ──> Route to END.
       * Otherwise ──> Route back to `researcher` node for additional context.

3. Termination (END):
   - Workflow finishes when either node explicitly emits a "FINAL ANSWER".

================================================================================
"""

import os
from typing import Literal
from dotenv import load_dotenv

os.environ["USER_AGENT"] = "Agentic-RAG-Cookbook/1.0 (contact: ash@codeaiwashnaiku.com)"

from langchain.chat_models import init_chat_model
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import Tool
from langchain_tavily import TavilySearch
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import BaseMessage, HumanMessage

from langchain.agents import create_agent
from langgraph.graph import MessagesState, END, StateGraph, START
from langgraph.types import Command

# https://reference.langchain.com/python/langgraph/graph/message/MessagesState
#  A shared data structure that represents the current snapshot of your application.

# Load environment variables (e.g., OPENAI_API_KEY, TAVILY_API_KEY)
load_dotenv()
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY", "your-tavily-key")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "your-openai-key")

# Initialize Chat Model
llm = init_chat_model("openai:gpt-4o-mini")

# Configure Web Search Tool
tavily_tool = TavilySearch(max_results=5)


# Generic function to create a retrieval tool
def make_retriever_tool_from_text(file, name, desc):
    embedding = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=512)
    docs = TextLoader(file, encoding="utf-8").load()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    ).split_documents(docs)
    vs = FAISS.from_documents(chunks, embedding)
    retriever = vs.as_retriever()

    def tool_func(query: str) -> str:
        print(f"📚 Using tool: {name}")
        results = retriever.invoke(query)
        return "\n\n".join(doc.page_content for doc in results)

    return Tool(name=name, description=desc, func=tool_func)


# Initialize Internal Retrieval Tool
FILE_NAME = "cybersecurity_data/internal_docs.txt"
internal_tool_1 = make_retriever_tool_from_text(
    FILE_NAME,
    "CyberSecurityResearchNotes",
    "Search internal research notes for experimental results",
)


# Graph orchestration helper functions
def get_next_node(last_message: BaseMessage, goto: str):
    print("🏁 [Workflow Engine] 'FINAL ANSWER' detected. Terminating execution loop.")
    if "FINAL ANSWER" in last_message.content:
        # Stop work if any agent signals they are finished
        return END
    return goto


# Dynamically constructs a base prompt specifying collaborative multi-agent rules, tool usage instructions, and stopping criteria.
# Appends a custom role-specific suffix to define the unique responsibilities of each individual agent.
def make_system_prompt(suffix: str) -> str:
    return (
        "You are a helpful AI assistant, collaborating with other assistants."
        " Use the provided tools to progress towards answering the question."
        " If you are unable to fully answer, that's OK, another assistant with different tools "
        " will help where you left off. Execute what you can to make progress."
        " If you or any of the other assistants have the final answer or deliverable,"
        " prefix your response with FINAL ANSWER so the team knows to stop."
        f"\n{suffix}"
    )


# --- RESEARCH TEAM AGENT ---
# Initializes a prebuilt ReAct agent initialized with a specific LLM, designated tools, and persona prompts.
# Combines tool execution and reasoning loops into a single runnable agent interface.
research_agent = create_agent(
    model=llm,
    tools=[internal_tool_1, tavily_tool],
    system_prompt=make_system_prompt(
        "You can only do research. Use the tool that you are binded with, you can use both of them."
        " You are working with a content writer colleague."
    ),
)

# `Command` - One or more commands to update the graph's state and send messages to nodes.
# https://reference.langchain.com/python/langgraph/types/Command

# Executes the research agent to gather data using assigned tools and formats the output as a HumanMessage.
# Dynamically routes execution either downstream to the blog generator or terminates at END if finished.
def research_node(state: MessagesState) -> Command[Literal["blog_generator", END]]:
    print("\n🔍 [Researcher Node] Analyzing query and gathering technical insights...")

    result = research_agent.invoke(state)
    goto = get_next_node(result["messages"][-1], "blog_generator")

    # Wrap in a human message to ensure downstream LLMs can parse it properly
    result["messages"][-1] = HumanMessage(content=result["messages"][-1].content, name="researcher")
    return Command(
        update={"messages": result["messages"],},
        goto=goto,)


# --- BLOG WRITE AGENT ---
# Initializes a prebuilt ReAct agent initialized with a specific LLM, designated tools, and persona prompts.
# Combines tool execution and reasoning loops into a single runnable agent interface.
blog_agent = create_agent(
    model=llm,
    tools=[],
    system_prompt=make_system_prompt(
        "You can only write a detailed blog. You are working with a researcher colleague."
    ),
)

# Invokes the blog writing agent to synthesize the researcher's context into a structured, detailed blog post.
# Wraps the resulting text into a message object and routes the state to either END or back to the researcher for more data.
def blog_node(state: MessagesState) -> Command[Literal["researcher", END]]:
    print("\n✍️ [Blog Generator Node] Synthesizing research data into a detailed blog post...")
    result = blog_agent.invoke(state)
    goto = get_next_node(result["messages"][-1], "researcher")

    result["messages"][-1] = HumanMessage(content=result["messages"][-1].content, name="blog_generator")
    return Command(
        update={"messages": result["messages"],},
        goto=goto,)


# --- COMPILE SEQUENTIAL WORKFLOW ---
workflow = StateGraph(MessagesState)
workflow.add_node("researcher", research_node)
workflow.add_node("blog_generator", blog_node)

workflow.add_edge(START, "researcher")
# Explicit edges showing possible hand-offs
# workflow.add_edge("researcher", "blog_generator")  # Researcher hands off to Blog Writer
# workflow.add_edge("blog_generator", "researcher")  # Blog Writer asks for more research
# workflow.add_edge("blog_generator", END)           # Blog Writer finishes and outputs FINAL ANSWER

graph = workflow.compile()

# Save the file as a PNG
OUTPUT_IMAGE_PATH = "Image_PNGs/30a_Multiagent_Sequential.png"
graph.get_graph().draw_mermaid_png(output_file_path=OUTPUT_IMAGE_PATH)    
# Automatically display/open the image on macOS NOTE _Just for testing purposes
os.system(f"open {OUTPUT_IMAGE_PATH}")


if __name__ == "__main__":
    query = (
        "Write a detailed blog post on how modern Next-Gen WAFs move beyond legacy regex filters "
        "to stop User-Agent spoofing. Explain how SmartParse tokenization, behavioral signals, "
        "and TLS/JA3 fingerprinting protect enterprise origin servers from resource exhaustion."
    )
    print(f"🚀 Executing Multi-Agent Pipeline for Query: '{query}'\n")
    response = graph.invoke({"messages": query})
    print("\n======================= FINAL GENERATED BLOG =======================")
    print(response["messages"][-1].content)
    print("====================================================================")