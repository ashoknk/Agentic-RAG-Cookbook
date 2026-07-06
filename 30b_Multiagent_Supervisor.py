"""
================================================================================
MAIN GOAL:
This script implements a Flat Multi-Agent Supervisor pattern using `langgraph-supervisor`.
A central supervisor agent coordinates task delegation between a specialized Research Agent
(equipped with search and document retrieval) and a specialized Math Agent (equipped with math tools).

HOW THIS CODE DIFFERS FROM SEQUENTIAL (`sequential_rag.py`):
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

from langchain.chat_models import init_chat_model
from langchain.agents import Tool
from langchain_tavily import TavilySearch
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor

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


internal_tool_1 = make_retriever_tool_from_text(
    "internal_docs.txt",
    "InternalResearchNotes",
    "Search internal research notes for experimental results",
)

# --- RESEARCH SPECIALIST AGENT ---
supervisor_research_agent = create_react_agent(
    model=llm,
    tools=[tavily_tool, internal_tool_1],
    prompt=(
        "You are a research agent.\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with research-related tasks, DO NOT do any math.\n"
        "- After you're done with your tasks, respond to the supervisor directly.\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name="research_agent",
)


# --- MATH SPECIALIST AGENT ---
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
        "- Assist ONLY with math-related tasks.\n"
        "- After you're done with your tasks, respond to the supervisor directly.\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name="math_agent",
)

# --- COMPILE SUPERVISOR ARCHITECTURE ---
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
# if __name__ == "__main__":
#     response = supervisor_graph.invoke({
#         "messages": "List all the transformer variants in production deployments from the retriever and then tell me what is 5 plus 10"
#     })
#     print(response["messages"][-1].content)