"""
================================================================================

This script implements a Flat Multi-Agent Supervisor pattern using `langgraph-supervisor`.
A central supervisor agent coordinates task delegation between a specialized Research Agent
(equipped with search and document retrieval) and a specialized Math Agent (equipped with math tools).

HOW THIS CODE DIFFERS FROM SEQUENTIAL (`30a_Multiagent_Sequential`):
1. Centralized Routing vs. Sequential Hand-off: Instead of Agent A directly triggering Agent B 
   in a fixed sequential pipeline, a central LLM Supervisor evaluates the conversation and determines 
   which agent to call next dynamically.
2. Multi-domain Specialization: It introduces two completely different domains (Research and Math) 
   which do not have a natural linear ordering, making an intelligent supervisor necessary.
3. Simplified Orchestration: Uses the prebuilt `create_supervisor` helper from LangGraph, reducing 
   boilerplate routing code.
================================================================================
"""

import os
from dotenv import load_dotenv

os.environ["USER_AGENT"] = "Agentic-RAG-Cookbook/1.0 (contact: ash@codeaiwashnaiku.com)"

from langchain.chat_models import init_chat_model
from langchain_core.tools import Tool
from langchain_tavily import TavilySearch
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain.agents import create_agent
from langgraph_supervisor import create_supervisor

# Load environment variables
load_dotenv()

# Secure credential assignments
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY", "your-tavily-key")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "your-openai-key")


# Initialize Chat Model
llm = init_chat_model("openai:gpt-4o-mini")

# Configure Web Search Tools
tavily_tool = TavilySearch(max_results=3)


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

FILE_NAME = "cybersecurity_data/internal_docs.txt"
internal_tool_1 = make_retriever_tool_from_text(
    FILE_NAME,
    "CyberSecurityResearchNotes",
    "Search internal research notes for experimental results",
)

# --- RESEARCH SPECIALIST AGENT ---
supervisor_research_agent = create_agent(
    model=llm,
    tools=[tavily_tool, internal_tool_1],
    system_prompt=(
        "You are a research agent.\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with research-related tasks, DO NOT do any math.\n"
        "- After you're done with your tasks, respond to the supervisor directly.\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name="research_agent",
)


# --- MATH SPECIALIST AGENT ---

def add(a: int, b: int) -> int:
    """Adds a and b.
    Args:
        a: first int
        b: second int
    """
    print(f"🔢 [Math Tool Executing]: Adding {a} + {b}")
    return a + b    


def subtract(a: int, b: int) -> int:
    """Subtracts b from a.
    Args:
        a: first int
        b: second int
    """
    print(f"🔢 [Math Tool Executing]: Adding {a} + {b}")
    return a - b


def multiply(a: int, b: int) -> int:
    """Multiply a and b.
    Args:
        a: first int
        b: second int
    """
    print(f"🔢 [Math Tool Executing]: Multiplying {a} * {b}")
    return a * b


def divide(a: int, b: int) -> float:
    """Divide a and b.
    Args:
        a: first int
        b: second int
    """
    
    print(f"🔢 [Math Tool Executing]: Dividing {a} / {b}")
    return a / b



supervisor_math_agent = create_agent(
    model=llm,
    tools=[add, subtract, multiply, divide],
    system_prompt=(
        "You are a math agent.\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with math-related tasks.\n"
        "- After you're done with your tasks, respond to the supervisor directly.\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name="math_agent",
)

# --- COMPILE SUPERVISOR ARCHITECTURE ---
# create_supervisor - Create a multi-agent supervisor.
# https://reference.langchain.com/python/langgraph-supervisor/supervisor/create_supervisor
# `add_handoff_back_messages=True` 
#   Automatically injects synthetic tracking messages whenever control transfers back from a worker agent 
#   (research_agent or math_agent) to the supervisor.
#  It helps the supervisor's LLM maintain context and prevent execution loops

# `output_mode="full_history"` 
# Dictates how the final state object formats and returns the output messages when you run supervisor_graph.
#   Retains and returns the entire sequence of messages from every node in the graph—the user query, 
#   the supervisor's delegation commands, the specialized agents' intermediate thoughts and tool execution results, 
#   handoff logs, and the final response.

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


# Example Run Execution (Uncomment below lines to run)
if __name__ == "__main__":
    # Multi-domain query requiring both Research Agent (retriever) and Math Agent (arithmetic)
    user_query = (
        "1. According to our internal engineering notes, what architectural technology does "
        "the Next-Gen WAF use instead of legacy regex filters?\n"
        "2. Multiply 2500 by 4 using the multiply tool."
    )

    print(f"🚀 [Supervisor Workflow] Invoking graph for query:\n{user_query}\n")
    
    # Stream graph updates step-by-step
    print("======================= FINAL SUPERVISOR RESPONSE =======================")
    for event in supervisor_graph.stream({"messages": user_query}):
        for node_name, state_update in event.items():
            print(f"🤖 [Active Node]: {node_name}")
    print("========================================================================")

    # response = supervisor_graph.invoke({"messages": user_query})
    # print(response["messages"][-1].content)
