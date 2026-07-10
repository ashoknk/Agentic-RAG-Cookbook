"""
 React_agent_basic (The Core Framework) with just 2 sources
Goal: Teach the standard ReAct framework loop using simple, predictable, public APIs.

Data Sources: 
    Just WebBaseLoader (from a WAF blog) and the standard WikipediaQueryRun.
Core Lesson: 
    Focus on how create_react_agent binds a chat model to tools and 
    executes a simple Think --> Act --> Observe loop.

Educational Goal: 
    Demonstrates the fundamental ReAct loop (Think --> Act --> Observe) using a prebuilt LangGraph agent linked 
    to a clean vector retriever and Wikipedia.

"""

"""
### 🤖 Implement ReAct with LangGraph - Basic Framework
ReAct (Reasoning + Acting) is a framework where an LLM:
- Reasons step-by-step (e.g., decomposes questions, makes decisions)
- Acts by calling tools like search, calculators, or retrievers

✅ Think → Retrieve → Observe → Reflect → Final Answer


"""

import os
from dotenv import load_dotenv
from typing import Annotated, TypedDict, Sequence

# Set a custom User-Agent identifying your application
os.environ["USER_AGENT"] = "Agentic-RAG-Cookbook/1.0 (contact: ash@codeaiwashnaiku.com)"


# from langgraph.prebuilt import create_react_agent
from langchain.agents import create_agent
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from langchain_core.tools import Tool
# from langchain.tools import WikipediaQueryRun
# from langchain.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, HumanMessage

# Initialize Environment & LLM
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
# llm = init_chat_model("openai:gpt-4o")
llm = init_chat_model("openai:gpt-4o-mini")

# --------------------------
# 1. Create Retriever Tool and Wikipedia Tool
# --------------------------
urls = [
    "https://www.fastly.com/blog/ua-spoofing-101-detection-defense-with-fastlys-next-gen-waf",
    "https://aws.amazon.com/blogs/networking-and-content-delivery/how-to-manage-ai-bots-with-aws-waf-and-enhance-security/",
]

docs = []
for url in urls:
    docs.extend(WebBaseLoader(url).load())

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

embedding = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=512)
vectorstore = FAISS.from_documents(chunks, embedding)
retriever = vectorstore.as_retriever()

def retriever_tool_func(query: str) -> str:
    print("📚 Using RAGRetriever tool")
    docs = retriever.invoke(query)
    return "\n".join([doc.page_content for doc in docs])

# Retriever tool
retriever_tool = Tool(
    name="RAGRetriever",
    description="Use this tool to fetch relevant knowledge base info about web security and WAFs.",
    func=retriever_tool_func
)

# Wikipedia tool
# wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
# Initialize the underlying Wikipedia API Wrapper
wiki_api = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1500)

# Define a custom function to intercept the execution and add your print statement
def wiki_tool_func(query: str) -> str:
    print("🌐 Using Wikipedia tool")  # Your new custom print statement
    return wiki_api.run(query)

# Re-create the wiki_tool using the standard Tool wrapper
wiki_tool = Tool(
    name="Wikipedia",
    description="A wrapper around Wikipedia. Use this tool to fetch general world knowledge from Wikipedia..",
    func=wiki_tool_func
)

# ----------------------------
# 2. Define the Agent Node & Tools
# ----------------------------
tools = [retriever_tool, wiki_tool]

# Create the native Langgraph react agent execution node
# NOTE Deprecated 
# react_node = create_react_agent(llm, tools) 
react_node = create_agent(llm, tools)
# NOTE : below helps with debugging but is verbose
# react_node = create_agent(llm, tools, debug=True)


# --------------------------
# 3. LangGraph Agent State
# --------------------------
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# --------------------------
# 4. Build LangGraph Graph
# --------------------------
builder = StateGraph(AgentState)

builder.add_node("react_agent", react_node)
builder.set_entry_point("react_agent")
builder.add_edge("react_agent", END)

graph = builder.compile()

# --------------------------
# 5. Run the ReAct Agent
# --------------------------
if __name__ == "__main__":
    user_query = "What is User Agent spoofing as per Fastly blog and how does Wikipedia describe XSS attacks?"
    state = {"messages": [HumanMessage(content=user_query)]}
    result = graph.invoke(state)

    print("\n✅ Final Answer:\n", result["messages"][-1].content)