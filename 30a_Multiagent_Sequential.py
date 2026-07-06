"""
================================================================================
MAIN GOAL:
This script implements a sequential, state-passing Multi-Agent RAG system using LangGraph v0.3.
It orchestrates two specialized agents (a Researcher and a Blog Writer) that work in a 
linear pipeline. The Researcher gathers information using web search and internal tools, 
and then passes the collected state directly to the Blog Writer to generate a detailed blog post.
================================================================================
"""

import os
from typing import Literal
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain.agents import Tool
from langchain_tavily import TavilySearch
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import BaseMessage, HumanMessage

from langgraph.prebuilt import create_react_agent
from langgraph.graph import MessagesState, END, StateGraph, START
from langgraph.types import Command

# Load environment variables (e.g., OPENAI_API_KEY, TAVILY_API_KEY)
load_dotenv()

# Secure credential assignments
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY", "your-tavily-key")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "your-openai-key")

# Setup dummy file for retriever if it does not exist
if not os.path.exists("internal_docs.txt"):
    with open("internal_docs.txt", "w", encoding="utf-8") as f:
        f.write(
            "Title: Transformer Variants for Production\n"
            "We have used the following transformer variants in production deployments:\n"
            "1. EfficientFormer: Optimized for mobile inference.\n"
            "2. Reformer: Tested for memory efficiency on embedded devices.\n"
        )

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
internal_tool_1 = make_retriever_tool_from_text(
    "internal_docs.txt",
    "InternalResearchNotes",
    "Search internal research notes for experimental results",
)


# Graph orchestration helper functions
def get_next_node(last_message: BaseMessage, goto: str):
    if "FINAL ANSWER" in last_message.content:
        # Stop work if any agent signals they are finished
        return END
    return goto


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
research_agent = create_react_agent(
    llm,
    tools=[internal_tool_1, tavily_tool],
    prompt=make_system_prompt(
        "You can only do research. Use the tool that you are binded with, you can use both of them."
        " You are working with a content writer colleague."
    ),
)


def research_node(state: MessagesState) -> Command[Literal["blog_generator", END]]:
    result = research_agent.invoke(state)
    goto = get_next_node(result["messages"][-1], "blog_generator")

    # Wrap in a human message to ensure downstream LLMs can parse it properly
    result["messages"][-1] = HumanMessage(
        content=result["messages"][-1].content, name="researcher"
    )
    return Command(
        update={
            "messages": result["messages"],
        },
        goto=goto,
    )


# --- BLOG WRITE AGENT ---
blog_agent = create_react_agent(
    llm,
    tools=[],
    prompt=make_system_prompt(
        "You can only write a detailed blog. You are working with a researcher colleague."
    ),
)


def blog_node(state: MessagesState) -> Command[Literal["researcher", END]]:
    result = blog_agent.invoke(state)
    goto = get_next_node(result["messages"][-1], "researcher")

    result["messages"][-1] = HumanMessage(
        content=result["messages"][-1].content, name="blog_generator"
    )
    return Command(
        update={
            "messages": result["messages"],
        },
        goto=goto,
    )


# --- COMPILE SEQUENTIAL WORKFLOW ---
workflow = StateGraph(MessagesState)
workflow.add_node("researcher", research_node)
workflow.add_node("blog_generator", blog_node)

workflow.add_edge(START, "researcher")
graph = workflow.compile()

# Example Run Execution (Uncomment below lines to run)
# if __name__ == "__main__":
#     response = graph.invoke({"messages": "Write a detailed blog on transformer variants in production deployments"})
#     print(response["messages"][-1].content)