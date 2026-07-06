"""
React_agent_multi_source (Enterprise/Advanced Routing)
Goal: 
    Show how a ReAct agent scales when given a massive, mixed toolkit (RAG + Academic Search + Corporate Docs).

What it contains: 
    Your generic make_retriever_tool_from_text helper, the ArxivLoader, and the local file tools (INTERNAL_DOCS_PATH, RESEARCH_NOTES_PATH).

Core Lesson: 
    Focus on tool descriptions. Show how the LLM dynamically evaluates semantic descriptions to route questions between 
    highly specific technical repositories versus academic papers.
    
Educational Goal: 
    Focuses on routing decisions across highly specific custom sources. It abstraction-loads separate local documentation assets 
    and academic endpoints (ArxivLoader), showcasing how  agent acts as an enterprise traffic router based purely on tool descriptions.
    """


"""
### 🤖 Advanced ReAct with LangGraph - Multi-Source Corporate Routing
Demonstrates how a ReAct agent dynamically evaluates semantic tool descriptions 
to pick between proprietary internal docs, technical research notes, or global academic indexes.
"""

import os
from dotenv import load_dotenv
from typing import Annotated, TypedDict, Sequence

# Set a custom User-Agent identifying your application
os.environ["USER_AGENT"] = "Agentic-RAG-Cookbook/1.0 (contact: ashnaiku@codeaiwashnaiku.com)"


# from langgraph.prebuilt import create_react_agent
from langchain.agents import create_agent
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from langchain_core.tools import Tool
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader

from langchain_community.tools import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper

from langchain_text_splitters import RecursiveCharacterTextSplitter

# from langchain.tools import WikipediaQueryRun
# from langchain.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, HumanMessage

# Initialize Environment & LLM
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
# llm = init_chat_model("openai:gpt-4o")
llm = init_chat_model("openai:gpt-4o-mini")

# -----------------------------------------------
# 1. Generic Factory to Create Retriever Tools
# -----------------------------------------------
def make_retriever_tool_from_text(file_path: str, name: str, desc: str) -> Tool:
    # Ensure local mockup file exists to avoid crashes
    if not os.path.exists(file_path):
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"Mock context data for {name}. Mentioning transformer variants and custom system architectures.")
            
    docs = TextLoader(file_path, encoding="utf-8").load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    vs = FAISS.from_documents(chunks, embedding)
    retriever = vs.as_retriever()

    def tool_func(query: str) -> str:
        print(f"📚 Using internal tool: {name}")
        results = retriever.invoke(query)
        return "\n\n".join(doc.page_content for doc in results)
    
    return Tool(name=name, description=desc, func=tool_func)

# Instantiating Local Tools
INTERNAL_DOCS_PATH = "cybersecurity_data/public_docs.txt"
RESEARCH_NOTES_PATH = "cybersecurity_data/private_docs.txt"

internal_tool_1 = make_retriever_tool_from_text(
    INTERNAL_DOCS_PATH,
    "InternalTechDocs",
    "Search internal tech documents for proprietary architecture, WAF configurations, and infrastructure methods."
)

internal_tool_2 = make_retriever_tool_from_text(
    RESEARCH_NOTES_PATH,
    "InternalResearchNotes",
    "Search internal research notes for AI experimental results, transformer variants, and agent designs."
)

# Wikipedia Tool
# wiki_tool = Tool(
#     name="Wikipedia",
#     description="Use this tool to fetch general world knowledge from Wikipedia.",
#     func=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1500))
# )

wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

# ArXiv Tool
# def arxiv_search(query: str) -> str:
#     print("🧪 Searching ArXiv...")
#     results = ArxivLoader(query).load()
#     return "\n\n".join(doc.page_content[:1000] for doc in results[:2]) or "No academic papers found."

arxiv_client = ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=1000))

def arxiv_search(query: str) -> str:
    print("🧪 Searching ArXiv...")
    try:
        # Run using the core LangChain wrapper instead of ArxivLoader
        return arxiv_client.run(query)
    except Exception as e:
        return f"ArXiv search encountered a network error: {e}"

arxiv_tool = Tool(
    name="ArxivSearch",
    description="Use this tool to fetch recent academic papers on machine learning and technical engineering topics.",
    func=arxiv_search
)

# ---------------------------------------
# 2. Compile Graph with Multi-Source Tools
# ---------------------------------------
tools = [wiki_tool, arxiv_tool, internal_tool_1, internal_tool_2]
# react_node = create_react_agent(llm, tools)
react_node = create_agent(llm, tools)
# NOTE : below helps with debugging but is verbose
# react_node = create_agent(llm, tools, debug=True)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

builder = StateGraph(AgentState)
builder.add_node("agentic_rag", react_node)
builder.set_entry_point("agentic_rag")
builder.add_edge("agentic_rag", END)

graph = builder.compile()

# ---------------------------------------
# 3. Run Complex Cross-Cutting Query
# ---------------------------------------
if __name__ == "__main__":
    
    
    #  Questions  (All Docs and Arvix): 
    user_question = (
        "Why do legacy regex-based WAF filters suffer from high latency and CPU spikes when trying to block rotated User-Agents? "
        "How does Fastly's Next-Gen WAF eliminate false positives to help DevOps and security teams scale business operations?"
        "Cross-reference if we have any internal research notes about this or check academic ArXiv papers."
    )
    state = {"messages": [HumanMessage(content=user_question)]}
    result = graph.invoke(state)
    
    # NOTE: Uncomment the following lines to run individual questions w/o ArXiv papers
    # # Question 1 (Technical/Private Docs): 
    # user_question1 = "Why do legacy regex-based WAF filters suffer from high latency and CPU spikes when trying to block rotated User-Agents?"
    # state = {"messages": [HumanMessage(content=user_question1)]}
    # result = graph.invoke(state)

    # # Question 2 (Business/Public Docs): 
    # user_question2 =  "How does Fastly's Next-Gen WAF eliminate false positives to help DevOps and security teams scale business operations?"
    # state = {"messages": [HumanMessage(content=user_question2)]}
    # result = graph.invoke(state)

    print("\n✅ Final Answer:\n", result["messages"][-1].content)