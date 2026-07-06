"""
Main Goal: 
    To demonstrate an advanced Agentic RAG Flow with Self-Correction (Corrective RAG).
The Core Mechanism: 
    Instead of using a prebuilt tool, this file constructs a custom state graph from scratch using 
    individual nodes (agent, retrieve, rewrite, generate) and a custom conditional router (grade_documents).


The Role of the Grader: 
    This is the "Self-RAG" or "Corrective RAG" pattern. When a tool retrieves documents, 
    they are passed to an LLM grader via a custom PromptTemplate to evaluate relevance (yes or no).
    If yes, it proceeds to generate an answer.
    If no, it triggers a rewrite node to fix the user's query and tries again.

Summary: 
    This is a classic Agentic RAG setup. It builds a specialized, deterministic pipeline designed to guarantee the quality of 
    retrieved data before generating an answer.
"""

#NOTE - This file is based on sec16c_1-AgenticRAG.ipynb  
# Section 16 - 94 and 95 Detailed Agentic RAG  Project Implementation 

import os
from dotenv import load_dotenv
load_dotenv()

# Set a custom User-Agent identifying your application
os.environ["USER_AGENT"] = "MyAgenticRAGApp/1.0 (contact: ashoknkjp@gmail.com)"

os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

# Standard Python Utilities & Type Hinting
from typing import Annotated, Literal, Sequence
from typing_extensions import TypedDict

# Pydantic (Data Validation & Schema Modeling)
from pydantic import BaseModel, Field


# Core LangChain & Core Components
from langchain_classic import hub
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_core.tools import create_retriever_tool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage, HumanMessage

# LangChain Community Tools, Infrastructure, and Models
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq

# ==============================================================================
# 5. LangGraph Architecture & Runtime Flow Control
# ==============================================================================
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# -----------------------------
# 1. Document Preprocessing
# -----------------------------

urls = [
    # Cloudflare's breakdown of WAF ML models and threat intelligence
    "https://blog.cloudflare.com/tag/waf/", 
    # AWS technical guide on configuring WAF rules specifically for aggressive AI scrapers
    "https://aws.amazon.com/blogs/networking-and-content-delivery/how-to-manage-ai-bots-with-aws-waf-and-enhance-security/",
    # Indusface enterprise analysis of bot mitigation mechanics
    "https://www.indusface.com/blog/top-bot-management-software/"
]

docs=[WebBaseLoader(url).load() for url in urls]

docs_list = [item for sublist in docs for item in sublist]

## Recursive character text ssplitter an vectorstore
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100
)

doc_splits = text_splitter.split_documents(docs_list)

## Add alll these text to vectordb

# OpenAIEmbeddings(model="text-embedding-3-small", dimensions=512)
vectorstore=FAISS.from_documents(
    documents=doc_splits,
    embedding=OpenAIEmbeddings(model="text-embedding-3-small", dimensions=512)
)

retriever=vectorstore.as_retriever()
retriever.invoke("what is bot protection?")

### Retriever To Retriever Tools
retriever_tool_wafbot =create_retriever_tool(
    retriever,
    "retriever_waf_db_blog",
    "Search and run information about WAF & BOT protection"
)

langchain_urls=[
    "https://python.langchain.com/docs/tutorials/",
    "https://python.langchain.com/docs/tutorials/chatbot/",
    "https://python.langchain.com/docs/tutorials/qa_chat_history/"
]

docs=[WebBaseLoader(url).load() for url in langchain_urls]

docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100
)

doc_splits = text_splitter.split_documents(docs_list)

## Add alll these text to vectordb
vectorstorelangchain=FAISS.from_documents(
    documents=doc_splits,
    embedding=OpenAIEmbeddings(model="text-embedding-3-small", dimensions=512)
)

retrieverlangchain=vectorstorelangchain.as_retriever()

retriever_tool_chatbot=create_retriever_tool(
    retrieverlangchain,
    "retriever_chatbot_blog",
    "Search and run information about Langchain"
)
tools=[retriever_tool_wafbot,retriever_tool_chatbot]

### LangGraph Workflow



class AgentState(TypedDict):
    # The add_messages function defines how an update should be processed
    # Default is to replace. add_messages says "append"
    messages: Annotated[Sequence[BaseMessage], add_messages]


# NOTE  Might need to use a different model - qwen-2.5-32b
# llm=ChatGroq(model="qwen/qwen3-32b")
llm=init_chat_model("groq:qwen/qwen3.6-27b")
llm.invoke("Hi")


def agent(state):
    """
    Invokes the agent model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply end.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with the agent response appended to messages
    """
    print("---CALL AGENT---")
    messages = state["messages"]
    # model = ChatGroq(model="qwen/qwen3-32b")
    model = init_chat_model("groq:qwen/qwen3.6-27b")
    model = model.bind_tools(tools)
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}




### Edges
def grade_documents(state) -> Literal["generate", "rewrite"]:
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (messages): The current state

    Returns:
        str: A decision for whether the documents are relevant or not
    """

    print("---CHECK RELEVANCE---")

    # Data model
    class grade(BaseModel):
        """Binary score for relevance check."""

        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    # LLM
    model = init_chat_model("groq:qwen/qwen3.6-27b")


    # LLM with tool and validation
    llm_with_tool = model.with_structured_output(grade)

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )

    # Chain
    chain = prompt | llm_with_tool

    messages = state["messages"]
    last_message = messages[-1]

    question = messages[0].content
    docs = last_message.content

    scored_result = chain.invoke({"question": question, "context": docs})

    score = scored_result.binary_score

    if score == "yes":
        print("---DECISION: DOCS RELEVANT---")
        return "generate"

    else:
        print("---DECISION: DOCS NOT RELEVANT---")
        print(score)
        return "rewrite"
    
def generate(state):
    """
    Generate answer

    Args:
        state (messages): The current state

    Returns:
         dict: The updated message
    """
    print("---GENERATE---")
    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]

    docs = last_message.content

    # Prompt
    prompt = hub.pull("rlm/rag-prompt")

    # LLM
    # llm = ChatGroq(model="qwen/qwen3-32b")
    llm = init_chat_model("groq:qwen/qwen3.6-27b")

    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    # Run
    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [response]}


def rewrite(state):
    """
    Transform the query to produce a better question.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with re-phrased question
    """

    print("---TRANSFORM QUERY---")
    messages = state["messages"]
    question = messages[0].content

    msg = [
        HumanMessage(
            content=f""" \n 
    Look at the input and try to reason about the underlying semantic intent / meaning. \n 
    Here is the initial question:
    \n ------- \n
    {question} 
    \n ------- \n
    Formulate an improved question: """,
        )
    ]

    # Grader
    # model = ChatGroq(model="qwen/qwen3-32b")
    model = init_chat_model("groq:qwen/qwen3.6-27b")
    response = model.invoke(msg)
    return {"messages": [response]}

# Define a new graph
workflow = StateGraph(AgentState)

# Define the nodes we will cycle between
workflow.add_node("agent", agent)  # agent
retrieve = ToolNode([retriever_tool,retriever_tool_langchain])
workflow.add_node("retrieve", retrieve)  # retrieval
workflow.add_node("rewrite", rewrite)  # Re-writing the question
workflow.add_node(
    "generate", generate
)  # Generating a response after we know the documents are relevant
# Call agent node to decide to retrieve or not
workflow.add_edge(START, "agent")

# Decide whether to retrieve
workflow.add_conditional_edges(
    "agent",
    # Assess agent decision
    tools_condition,
    {
        # Translate the condition outputs to nodes in our graph
        "tools": "retrieve",
        END: END,
    },
)

# Edges taken after the `action` node is called.
workflow.add_conditional_edges(
    "retrieve",
    # Assess agent decision
    grade_documents,
)
workflow.add_edge("generate", END)
workflow.add_edge("rewrite", "agent")

# Compile
graph = workflow.compile()
# from IPython.display import Image, display
# display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
OUTPUT_IMAGE_PATH = "Image_PNGs/AgenticRAGDetailedProject_Grader.png"
graph.get_graph().draw_mermaid_png(output_file_path=OUTPUT_IMAGE_PATH)    
# 2. Automatically display/open the image on macOS
os.system(f"open {OUTPUT_IMAGE_PATH}")


graph.invoke({"messages": [HumanMessage(content="What is Langgraph?")]})

graph.invoke({"messages": [HumanMessage(content="What is Langchain?")]})

graph.invoke({"messages": [HumanMessage(content="What is Machine learning?")]})
