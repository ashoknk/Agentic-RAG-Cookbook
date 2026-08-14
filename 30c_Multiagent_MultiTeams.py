"""
================================================================================

This script implements an advanced Hierarchical Agent Teams pattern. It orchestrates nested 
subgraphs (a Research Team subgraph and a Writing Team subgraph) coordinated by a top-level Root Supervisor.
This architecture is highly scalable and handles complex multi-stage tasks 
(researching, outline creation, document writing, python chart generation).

HOW THIS CODE DIFFERS FROM FLAT SUPERVISOR (`30b_Multiagent_Supervisor.py`):
1. Hierarchical Nesting vs. Flat Delegation: 
    a. 30b_Multiagent_Supervisor.py a single flat supervisor coordinating individual agents. 
    b. This file uses a supervisor that coordinates entire *subgraphs* (teams), 
    each of which has its own internal supervisor managing sub-agents.
2. Local Tool Integration and File Sharing: 
    It integrates local filesystem state management (reading, writing, and editing documents) and 
    dynamic code execution (Python REPL) shared among sub-agents.
3. Manual Subgraph Invocation: 
    Shows how subgraphs are wrapped as standard LangGraph nodes 
   (`research_graph.invoke` and `paper_writing_graph.invoke`) to pass state cleanly across team boundaries.


┌────────────────────────────────────────────────────────────────────────┐
│                             super_builder                              │
│                                                                        │
│                       ┌──────────────────────┐                         │
│                       │   teams_supervisor   │                         │
│                       └──────────┬───────────┘                         │
│                                  │                                     │
│               ┌──────────────────┴──────────────────┐                  │
│               ▼                                     ▼                  │
│   ┌───────────────────────┐             ┌───────────────────────┐      │
│   │  call_research_team   │             │call_paper_writing_team│      │
│   │        (Node)         │             │        (Node)         │      │
│   └───────────┬───────────┘             └───────────┬───────────┘      │
│               │                                     │                  │
└───────────────┼─────────────────────────────────────┼──────────────────┘
                │ (Invokes)                           │ (Invokes)
                ▼                                     ▼
     ┌─────────────────────┐               ┌─────────────────────┐
     │   research_graph    │               │ paper_writing_graph │
     │     (Subgraph)      │               │     (Subgraph)      │
     └─────────────────────┘               └─────────────────────┘   
================================================================================

team_search_agent ---> tools=[tavily_tool, internal_tool_1]
web_scraper_agent --->  tools=[scrape_webpages]
doc_writer_agent --->  tools=[write_document, edit_document, read_document]
note_taking_agent ---> tools=[create_outline, read_document]
chart_generating_agent --->  tools=[read_document, python_repl_tool]

"""

import os
import io
import sys
from dotenv import load_dotenv

os.environ["USER_AGENT"] = "Agentic-RAG-Cookbook/1.0 (contact: ash@codeaiwashnaiku.com)"

from typing import Annotated, Dict, List, Literal, Optional
from pathlib import Path
from tempfile import TemporaryDirectory
from typing_extensions import TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.tools import Tool
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.language_models.chat_models import BaseChatModel

from langchain.agents import create_agent
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command

# Load environment variables
load_dotenv()
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY", "your-tavily-key")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "your-openai-key")

# Initialize Chat Model
llm = init_chat_model("openai:gpt-4o-mini")

# Configure Web Search Tools
tavily_tool = TavilySearch(max_results=5)

# Loads a text file, splits it into vector chunks via FAISS, & 
# builds a custom retriever tool for internal document search.
def make_retriever_tool_from_text(file, name, desc):
    embedding = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=512)
    docs = TextLoader(file, encoding="utf-8").load()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    ).split_documents(docs)
    vs = FAISS.from_documents(chunks, embedding)
    retriever = vs.as_retriever()

    
    # Executes a vector search against the loaded document retriever and returns matching context chunks (Inner function)  
    def tool_func(query: str) -> str:
        print(f"📚 Using tool: {name}")
        results = retriever.invoke(query)
        return "\n\n".join(doc.page_content for doc in results)

    return Tool(name=name, description=desc, func=tool_func)

FILE_NAME = "cybersecurity_data/internal_docs.txt"
internal_tool_1 = make_retriever_tool_from_text(
    FILE_NAME,
    "CyberSecurityResearchNotes",
    "Search internal research notes for experimental results",
)


# --- GENERAL UTILITIES & SHARED TOOLS ---
# Fetches web pages using WebBaseLoader and formats their extracted raw contents 
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
print(f"📁 Temporary working directory created at: {WORKING_DIRECTORY}") #NOTE Just for testing


# Formats a list of strings into a numbered outline and writes it to a file inside the temporary working directory.
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


# Creates a new text document file in the temporary directory with the provided string contents
@tool
def write_document(
    content: Annotated[str, "Text content to be written into the document."],
    file_name: Annotated[str, "File path to save the document."],
) -> Annotated[str, "Path of the saved document file."]:
    """Create and save a text document."""
    with (WORKING_DIRECTORY / file_name).open("w") as file:
        file.write(content)
    return f"Document saved to {file_name}"


# Reads an existing document and inserts specified lines at designated line numbers before re-saving the file
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

    # inserts.items() extracts each entry as a (line_number, text) tuple.  
    # sorted(...) orders those tuples by line_number in ascending order.
    sorted_inserts = sorted(inserts.items())
    # Validate line numbers and insert text sequentially at specified 1-indexed positions.
    for line_number, text in sorted_inserts:
        if 1 <= line_number <= len(lines) + 1:
            lines.insert(line_number - 1, text + "\n")
            print(f"Inserted text at line {line_number}: {text}") #NOTE Just for testing
        else:
            return f"Error: Line number {line_number} is out of range."

    with (WORKING_DIRECTORY / file_name).open("w") as file:
        file.writelines(lines)

    return f"Document edited and saved to {file_name}"


# Reads lines from a target document in the working directory, 
# returning all text or a specific slice based on start/end indices
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


# Dynamically executes Python code snippets while redirecting standard output 
# to return generated print outputs or error details.
@tool
def python_repl_tool(
    code: Annotated[str, "The python code to execute to generate your chart."],
) -> str:
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    
    # Capture standard output (what print() statements write to)
    old_stdout = sys.stdout
    redirected_output = sys.stdout = io.StringIO()
    
    try:
        # Execute the code in a global/local dictionary context
        exec(code, {}, {})
        sys.stdout = old_stdout  # Restore standard output
        result = redirected_output.getvalue()
        return f"Successfully executed:\n{result}"
        
    except BaseException as e:
        sys.stdout = old_stdout  # Restore standard output even if it fails
        return f"Failed to execute. Error: {repr(e)}"


# --- HIERARCHICAL STATE & SUPERVISOR BUILDER ---
class State(MessagesState):
    next: str

# Factory function that constructs a custom supervisor node function tailored to 
# direct routing between specified worker nodes.
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

    # Prompts the LLM with structured output to select the next worker node or signal task completion (Inner function)
    def supervisor_node(state: State) -> Command[Literal[*members, "__end__"]]:
        messages = [
            {"role": "system", "content": system_prompt},
        ] + state["messages"]
        response = llm.with_structured_output(Router).invoke(messages)
        # Reads the destination selected by the supervisor LLM (which will be a worker agent's name or "FINISH"
        goto = response["next"]
        # Checks if the LLM decided the task is complete. Convert the string "FINISH" into LangGraph's END constant.
        if goto == "FINISH":
            goto = END

        return Command(goto=goto, update={"next": goto})

    return supervisor_node

    # Command -https://reference.langchain.com/python/langgraph/types/Command


# --- RESEARCH TEAM SUBGRAPH ---
team_search_agent = create_agent(llm, tools=[tavily_tool, internal_tool_1])

# Invokes the search team agent on current state messages and routes results back to the research supervisor node
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


web_scraper_agent = create_agent(llm, tools=[scrape_webpages])

# Runs the web scraper agent to extract webpage contents and routes its response back to the research supervisor
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

# Save the file as a PNG
OUTPUT_IMAGE_PATH = "Image_PNGs/30a_research_team.png"
research_graph.get_graph().draw_mermaid_png(output_file_path=OUTPUT_IMAGE_PATH)    
os.system(f"open {OUTPUT_IMAGE_PATH}")


# 1. RESEARCH TEAM SUBGRAPH (30a_research_team.png)
# __start__ ---> supervisor                      [Static edge via add_edge(START, "supervisor")]
# supervisor ---> search                         [Dynamic edge via Command(goto="search")]
# supervisor ---> web_scraper                    [Dynamic edge via Command(goto="web_scraper")]
# supervisor ---> __end__                        [Dynamic edge via Command(goto=END)]
# search ---> supervisor                         [Direct loop via Command(goto="supervisor")]
# web_scraper ---> supervisor                    [Direct loop via Command(goto="supervisor")]


# --- WRITING TEAM SUBGRAPH ---
doc_writer_agent = create_agent(llm,tools=[write_document, edit_document, read_document],
    system_prompt=(
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


note_taking_agent = create_agent(llm,tools=[create_outline, read_document],
    system_prompt=(
        "You can read documents and create outlines for the document writer. "
        "Don't ask follow-up questions."
    ),
)

# Invokes the note-taking agent to construct outlines and routes messages back to the paper writing supervisor.
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


chart_generating_agent = create_agent(llm, tools=[read_document, python_repl_tool])

# Calls the chart generator agent to execute visualization Python code and reports findings back to the writing supervisor.
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

OUTPUT_IMAGE_PATH = "Image_PNGs/30a_paperwriting_team.png"
paper_writing_graph.get_graph().draw_mermaid_png(output_file_path=OUTPUT_IMAGE_PATH)    
os.system(f"open {OUTPUT_IMAGE_PATH}")


# 2. PAPER WRITING TEAM SUBGRAPH (30a_paperwriting_team.png)
# __start__ ---> supervisor                      [Static edge via add_edge(START, "supervisor")]
# supervisor ---> chart_generator                [Dynamic edge via Command(goto="chart_generator")]
# supervisor ---> doc_writer                     [Dynamic edge via Command(goto="doc_writer")]
# supervisor ---> note_taker                     [Dynamic edge via Command(goto="note_taker")]
# supervisor ---> __end__                        [Dynamic edge via Command(goto=END)]
# chart_generator ---> supervisor                [Direct loop via Command(goto="supervisor")]
# doc_writer ---> supervisor                     [Direct loop via Command(goto="supervisor")]
# note_taker ---> supervisor                     [Direct loop via Command(goto="supervisor")]



# --- TOP LEVEL ROOT SUPERVISOR ---
teams_supervisor_node = make_supervisor_node(llm, ["research_team", "writing_team"])

# Invokes the entire compiled research team subgraph with incoming state messages and 
# hands results back to the root supervisor.
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

# Invokes the entire compiled paper writing team subgraph with incoming state messages and 
# hands results back to the root supervisor
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

OUTPUT_IMAGE_PATH = "Image_PNGs/30a_super_graph.png"
super_graph.get_graph().draw_mermaid_png(output_file_path=OUTPUT_IMAGE_PATH)    
os.system(f"open {OUTPUT_IMAGE_PATH}")


# 3. ROOT SUPERVISOR GRAPH (30a_super_graph.png)
# __start__ ---> supervisor                      [Static edge via add_edge(START, "supervisor")]
# supervisor ---> research_team                  [Dynamic edge via Command(goto="research_team")]
# supervisor ---> writing_team                   [Dynamic edge via Command(goto="writing_team")]
# supervisor ---> __end__                        [Dynamic edge via Command(goto=END)]
# research_team ---> supervisor                  [Direct loop via Command(goto="supervisor")]
# writing_team ---> supervisor                   [Direct loop via Command(goto="supervisor")]


# Example Run Execution (Uncomment below lines to run)
if __name__ == "__main__":
    

    # query = (
    #     "Research how modern Next-Gen WAFs move beyond legacy regex filters "
    #     "to stop User-Agent spoofing using SmartParse, behavioral signals, and TLS/JA3 fingerprinting. "
    #     "First, search for details and structure an outline. "
    #     "Then, generate a Python chart comparing legacy regex vs. Next-Gen WAF detection latency or accuracy. "
    #     "Finally, write a comprehensive blog post incorporating the outline and chart findings."
    # )

    query = (
        "Research our internal notes (referencing IRN-2026-WAF-UA) and web sources regarding "
        "how Next-Gen WAFs move beyond legacy regex filters to stop rotated User-Agent spoofing. "
        "Detail how SmartParse tokenization, behavioral signals, and TLS/JA3 fingerprinting prevent "
        "O(N) CPU overhead and latency spikes on origin servers. "
        "First, synthesize the research into a structured outline. "
        "Next, generate a Python chart plotting O(N) legacy regex latency overhead versus "
        "SmartParse structural parsing performance. "
        "Finally, compile everything into a polished, comprehensive technical blog post."
    )

    print(f"🚀 Executing Multi-Agent Pipeline for Query: '{query}'\n")
    response = super_graph.invoke(
        {
            "messages": [
                ("user", query)
            ],
        }
    )
    print("\n======================= FINAL GENERATED BLOG =======================")
    print(response["messages"][-1].content)
    print("====================================================================")

