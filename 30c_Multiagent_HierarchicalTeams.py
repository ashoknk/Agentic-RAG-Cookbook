"""
================================================================================
MAIN GOAL:
This script implements an advanced Hierarchical Agent Teams pattern. It orchestrates nested 
subgraphs (a Research Team subgraph and a Writing Team subgraph) coordinated by a top-level Root Supervisor.
This architecture is highly scalable and handles complex multi-stage tasks (researching, outline creation, 
document writing, python chart generation).

HOW THIS CODE DIFFERS FROM FLAT SUPERVISOR (`supervisor_rag.py`):
1. Hierarchical Nesting vs. Flat Delegation: File 2 uses a single flat supervisor coordinating individual 
   agents. This file uses a supervisor that coordinates entire *subgraphs* (teams), each of which 
   has its own internal supervisor managing sub-agents.
2. Local Tool Integration and File Sharing: It integrates local filesystem state management (reading, writing, 
   and editing documents) and dynamic code execution (Python REPL) shared among sub-agents.
3. Manual Subgraph Invocation: Shows how subgraphs are wrapped as standard LangGraph nodes 
   (`research_graph.invoke` and `paper_writing_graph.invoke`) to pass state cleanly across team boundaries.
================================================================================
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
from langchain_community.document_loaders import WebBaseLoader
from langchain_experimental.utilities import PythonREPL
from langchain_openai import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import tool

from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command

# Load environment variables
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

# Configure Web Search Tools
tavily_tool = TavilySearch(max_results=5)


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


internal_tool_1 = make_retriever_tool_from_text(
    "internal_docs.txt",
    "InternalResearchNotes",
    "Search internal research notes for experimental results",
)


# --- GENERAL UTILITIES & SHARED TOOLS ---
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


# Temporary shared environment directory
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


@tool
def read_document(
    file_name: Annotated[str, "File path to read the document from."],
    start: Annotated[Optional[int], "The start line. Default is 0"] = None,
    end: Annotated[Optional[int], "The end line. Default is None"] = None,
) -> str:
    """Read the specified document."""
    with (WORKING_DIRECTORY / file_name).open("r") as file:
        lines = file.readlines()
    if start is None:
        start = 0
    return "\n".join(lines[start:end])


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


# --- HIERARCHICAL STATE & SUPERVISOR BUILDER ---
class State(MessagesState):
    next: str


def make_supervisor_node(llm: BaseChatModel, members: list[str]) -> str:
    options = ["FINISH"] + members
    system_prompt = (
        "You are a supervisor tasked with managing a conversation between the"
        f" following workers: {members}. Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status. When finished,"
        " respond with FINISH."
    )

    class Router(TypedDict):
        """Worker to route to next. If no workers needed, route to FINISH."""

        next: Literal[*options]

    def supervisor_node(state: State) -> Command[Literal[*members, "__end__"]]:
        messages = [
            {"role": "system", "content": system_prompt},
        ] + state["messages"]
        response = llm.with_structured_output(Router).invoke(messages)
        goto = response["next"]
        if goto == "FINISH":
            goto = END

        return Command(goto=goto, update={"next": goto})

    return supervisor_node


# --- RESEARCH TEAM SUBGRAPH ---
team_search_agent = create_react_agent(llm, tools=[tavily_tool, internal_tool_1])


def search_node(state: State) -> Command[Literal["supervisor"]]:
    result = team_search_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="search")
            ]
        },
        goto="supervisor",
    )


web_scraper_agent = create_react_agent(llm, tools=[scrape_webpages])


def web_scraper_node(state: State) -> Command[Literal["supervisor"]]:
    result = web_scraper_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=result["messages"][-1].content, name="web_scraper"
                )
            ]
        },
        goto="supervisor",
    )


research_supervisor_node = make_supervisor_node(llm, ["search", "web_scraper"])

research_builder = StateGraph(State)
research_builder.add_node("supervisor", research_supervisor_node)
research_builder.add_node("search", search_node)
research_builder.add_node("web_scraper", web_scraper_node)
research_builder.add_edge(START, "supervisor")
research_graph = research_builder.compile()


# --- WRITING TEAM SUBGRAPH ---
doc_writer_agent = create_react_agent(
    llm,
    tools=[write_document, edit_document, read_document],
    prompt=(
        "You can read, write and edit documents based on note-taker's outlines. "
        "Don't ask follow-up questions."
    ),
)


def doc_writing_node(state: State) -> Command[Literal["supervisor"]]:
    result = doc_writer_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=result["messages"][-1].content, name="doc_writer"
                )
            ]
        },
        goto="supervisor",
    )


note_taking_agent = create_react_agent(
    llm,
    tools=[create_outline, read_document],
    prompt=(
        "You can read documents and create outlines for the document writer. "
        "Don't ask follow-up questions."
    ),
)


def note_taking_node(state: State) -> Command[Literal["supervisor"]]:
    result = note_taking_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=result["messages"][-1].content, name="note_taker"
                )
            ]
        },
        goto="supervisor",
    )


chart_generating_agent = create_react_agent(
    llm, tools=[read_document, python_repl_tool]
)


def chart_generating_node(state: State) -> Command[Literal["supervisor"]]:
    result = chart_generating_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=result["messages"][-1].content, name="chart_generator"
                )
            ]
        },
        goto="supervisor",
    )


doc_writing_supervisor_node = make_supervisor_node(
    llm, ["doc_writer", "note_taker", "chart_generator"]
)

paper_writing_builder = StateGraph(State)
paper_writing_builder.add_node("supervisor", doc_writing_supervisor_node)
paper_writing_builder.add_node("doc_writer", doc_writing_node)
paper_writing_builder.add_node("note_taker", note_taking_node)
paper_writing_builder.add_node("chart_generator", chart_generating_node)
paper_writing_builder.add_edge(START, "supervisor")
paper_writing_graph = paper_writing_builder.compile()


# --- TOP LEVEL ROOT SUPERVISOR ---
teams_supervisor_node = make_supervisor_node(llm, ["research_team", "writing_team"])


def call_research_team(state: State) -> Command[Literal["supervisor"]]:
    response = research_graph.invoke({"messages": state["messages"][-1]})
    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=response["messages"][-1].content, name="research_team"
                )
            ]
        },
        goto="supervisor",
    )


def call_paper_writing_team(state: State) -> Command[Literal["supervisor"]]:
    response = paper_writing_graph.invoke({"messages": state["messages"][-1]})
    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=response["messages"][-1].content, name="writing_team"
                )
            ]
        },
        goto="supervisor",
    )


super_builder = StateGraph(State)
super_builder.add_node("supervisor", teams_supervisor_node)
super_builder.add_node("research_team", call_research_team)
super_builder.add_node("writing_team", call_paper_writing_team)
super_builder.add_edge(START, "supervisor")
super_graph = super_builder.compile()

# Example Run Execution (Uncomment below lines to run)
# if __name__ == "__main__":
#     response = super_graph.invoke(
#         {
#             "messages": [
#                 ("user", "Write about transformer variants in production deployments.")
#             ],
#         }
#     )
#     print(response["messages"][-1].content)