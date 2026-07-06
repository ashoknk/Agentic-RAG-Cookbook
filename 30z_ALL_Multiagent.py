"""
🤖 What are Multi-Agent RAG Systems?

A Multi-Agent RAG System splits the RAG pipeline into multiple specialized agents 
— each responsible for a specific role — and enables them to collaborate on a single query or task.

1. 📋 Multi-Agent Network RAG System with LangGraph
Project Overview:
A beginner-friendly Retrieval-Augmented Generation (RAG) system that uses a multi-agent 
architecture to intelligently answer questions from your documents. Built with LangGraph v0.3 
for workflow orchestration and OpenAI for language understanding.

What It Does:
Transforms your documents (PDFs, text files) into a searchable knowledge base that can answer 
questions intelligently using AI. Simply upload documents and ask questions in natural language 
- the system finds relevant information and generates comprehensive answers.

Key Features:
- 📚 Multi-Format Support: Handles PDF and text documents
- 🤖 3-Agent Architecture: Specialized agents for document processing, retrieval, and answer generation
- 🔍 Smart Search: Vector-based semantic search finds relevant information
- 💬 Natural Language Q&A: Ask questions in plain English
"""

import os
from typing import Annotated, Dict, List, Literal, Optional
from pathlib import Path
from tempfile import TemporaryDirectory
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain.agents import Tool
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from langchain.tools import WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import BaseMessage, HumanMessage, trim_messages
from langchain_community.document_loaders import WebBaseLoader
from langchain_experimental.utilities import PythonREPL

from langgraph.prebuilt import create_react_agent
from langgraph.graph import MessagesState, END, StateGraph, START
from langgraph.types import Command

# Load environment variables
load_dotenv()

# NOTE: Avoid hardcoding sensitive API credentials in plaintext strings
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY", "your-key-here")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

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
        # Any agent decided the work is done
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
        "You can only do research. Use the tool that you are binded with, you can use both of them"
        " You are working with a content writer colleague."
    ),
)


def research_node(state: MessagesState) -> Command[Literal["blog_generator", END]]:
    result = research_agent.invoke(state)
    goto = get_next_node(result["messages"][-1], "blog_generator")

    # Wrap in a human message, as not all providers allow
    # AI message at the last position of the input messages list
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

workflow.add_edge(START, \"researcher\")
graph = workflow.compile()

# Execute Sequential Workflow Example
# response = graph.invoke({"messages": "Write a detailed blog on transformer variants in production deployments"})
# print(response["messages"][-1].content)


"""
### Multi Agent Supervisor With RAG
Supervisor is a multi-agent architecture where specialized agents are coordinated by a central supervisor agent. 
The supervisor agent controls all communication flow and task delegation, making decisions about which agent 
to invoke based on the current context and task requirements.

In this section, you will build a supervisor system with two agents — a research and a math expert.
1. Build specialized research and math agents
2. Build a supervisor for orchestrating them with the prebuilt langgraph-supervisor
3. Implement advanced task delegation
"""

from langgraph_supervisor import create_supervisor

web_search = TavilySearch(max_results=3)

# Build Specialized Research Agent
supervisor_research_agent = create_react_agent(
    model=llm,
    tools=[web_search, internal_tool_1],
    prompt=(
        "You are a research agent.\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with research-related tasks, DO NOT do any math\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name="research_agent",
)


# Build Math Tools and Agent
def add(a: float, b: float):
    """Add two numbers."""
    return a + b


def multiply(a: float, b: float):
    """Multiply two numbers."""
    return a * b


def divide(a: float, b: float):
    """Divide two numbers."""
    return a / b


supervisor_math_agent = create_react_agent(
    model=llm,
    tools=[add, multiply, divide],
    prompt=(
        "You are a math agent.\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with math-related tasks\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name="math_agent",
)

# Compile Supervisor Architecture
# NOTE: Requires `pip install langgraph-supervisor`
supervisor_graph = create_supervisor(
    model=llm,
    agents=[supervisor_research_agent, supervisor_math_agent],
    prompt=(
        "You are a supervisor managing two agents:\n"
        "- a research agent. Assign research-related tasks to this agent\n"
        "- a math agent. Assign math-related tasks to this agent\n"
        "Assign work to one agent at a time, do not call agents in parallel.\n"
        "Do not do any work yourself."
    ),
    add_handoff_back_messages=True,
    output_mode="full_history",
).compile()

# Execute Supervisor Graph Example
# supervisor_response = supervisor_graph.invoke({"messages": "list all the transformer variants in production deployments from the retriever and then tell me what is 5 plus 10"})
# print(supervisor_response["messages"][-1].content)


"""
### Hierarchical Agent Teams With RAG
For some applications, the system may be more effective if work is distributed hierarchically.
You can do this by composing different subgraphs and creating a top-level supervisor, 
along with mid-level supervisors.
"""


@tool
def scrape_webpages(urls: List[str]) -> str:
    """Use requests and bs4 to scrape the provided web pages for detailed information."""
    loader = WebBaseLoader(urls)
    docs = loader.load()
    return "\n\n".join(
        [
            f'<Document name="{doc.metadata.get("title", "")}">\n{doc.page_content}\n</Document>'
            for doc in docs
        ]
    )


# Temporary environment file management setup
_TEMP_DIRECTORY = TemporaryDirectory()
WORKING_DIRECTORY = Path(_TEMP_DIRECTORY.name)


@tool
def create_outline(
    points: Annotated[List[str], "List of main points or sections."],
    file_name: Annotated[str, "File path to save the outline."],
) -> Annotated[str, "Path of the saved outline file."]:
    """Create and save an outline."""
    with (WORKING_DIRECTORY / file_name).open("w") as file:
        for i, point in enumerate(points):
            file.write(f"{i + 1}. {point}\n")
    return f"Outline saved to {file_name}"


@tool
def write_document(
    content: Annotated[str, "Text content to be written into the document."],
    file_name: Annotated[str, "File path to save the document."],
) -> Annotated[str, "Path of the saved document file."]:
    """Create and save a text document."""
    with (WORKING_DIRECTORY / file_name).open("w") as file:
        file.write(content)
    return f"Document saved to {file_name}"


@tool
def edit_document(
    file_name: Annotated[str, "Path of the document to be edited."],
    inserts: Annotated[
        Dict[int, str],
        "Dictionary where key is the line number (1-indexed) and value is the text to be inserted at that line.",
    ],
) -> Annotated[str, "Path of the edited document file."]:
    """Edit a document by inserting text at specific line numbers."""
    with (WORKING_DIRECTORY / file_name).open("r") as file:
        lines = file.readlines()

    sorted_inserts = sorted(inserts.items())
    for line_number, text in sorted_inserts:
        if 1 <= line_number <= len(lines) + 1:
            lines.insert(line_number - 1, text + "\n")
        else:
            return f"Error: Line number {line_number} is out of range."

    with (WORKING_DIRECTORY / file_name).open("w") as file:
        file.writelines(lines)

    return f"Document edited and saved to {file_name}"


# Warning: This executes code locally, which can be unsafe when not sandboxed
repl = PythonREPL()


@tool
def python_repl_tool(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    return f"Successfully executed:\n{result}"

### Suggested Code Changes & Enhancements
# 1. **API Key Management Security**: The hardcoded Tavily key string (`"tvly-VYpLu..."`) has been wrapped inside `os.getenv("TAVILY_API_KEY", "your-key-here")`. Always manage sensitive runtime configurations securely inside your target environment or local `.env` setup.
# 2. **Consolidated Imports**: Imports scattered across the original `.ipynb` cells have been cleanly structured at the top of the script for enhanced visibility and clean execution patterns.
# 3. **Sandbox Python Code Execution**: The `python_repl_tool` processes dynamic string payloads contextually. Consider executing this function inside isolated Docker infrastructure or sandboxed computational boundaries if you extend this setup beyond internal developer runtimes.