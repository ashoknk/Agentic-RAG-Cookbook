"""
Main Goal: 
    To demonstrate an advanced Agentic RAG Flow with Self-Correction (Corrective RAG).
The Core Mechanism: 
    Instead of using a prebuilt tool, this file constructs a custom state graph from scratch using 
    individual nodes (agent, retrieve, rewrite, generate) and a custom conditional router (grade_documents).


The Role of the Grader: 
    This is the "Self-RAG" or "Corrective RAG" pattern. When a tool retrieves documents, 
    they are passed to an LLM grader via a custom PromptTemplate to evaluate relevance (yes or no).
    -If yes, it proceeds to generate an answer.
    -If no, it triggers a rewrite node to fix the user's query and tries again.

Summary: 
    This is a classic Agentic RAG setup. It builds a specialized, deterministic pipeline designed to guarantee the quality of 
    retrieved data before generating an answer.
"""

import os
from typing import Annotated, Literal, Sequence
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Set a custom User-Agent identifying your application
os.environ["USER_AGENT"] = "MyAgenticRAGApp/1.0 (contact: ashoknkjp@gmail.com)"

# Load environment configs
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Core LangChain Framework
from langchain_classic import hub
from langchain.chat_models import init_chat_model
from langchain_core.tools import create_retriever_tool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage, HumanMessage

# LangChain Ecosystem Components
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

# LangGraph Architecture
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# ==============================================================================
# 1. INITIAL BASELINE & DATA PREPROCESSING
# ==============================================================================

# Common embedding engine instance configurations
embeddings_engine = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=512)
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# --- DB 1: WebSecurity WAF & Bot Protection ---
waf_urls = [
    "https://www.fastly.com/blog/ua-spoofing-101-detection-defense-with-fastlys-next-gen-waf",
    "https://www.fastly.com/blog/credential-stuffing-attacks-vs-brute-force-attacks-what-is-the-difference",
    "https://www.fastly.com/blog/what-is-cve-2026-23869-react-server-components-security-alert",
    "https://www.fastly.com/blog/ddos-in-december-2025"
]

# waf_urls = [
#     "https://blog.cloudflare.com/tag/waf/", 
#     "https://aws.amazon.com/blogs/networking-and-content-delivery/how-to-manage-ai-bots-with-aws-waf-and-enhance-security/",
#     "https://www.indusface.com/blog/top-bot-management-software/"
# ]
waf_docs = [item for url in waf_urls for item in WebBaseLoader(url).load()]
waf_splits = splitter.split_documents(waf_docs)
vectorstore_waf = FAISS.from_documents(documents=waf_splits, embedding=embeddings_engine)
retriever_waf = vectorstore_waf.as_retriever()

retriever_tool_wafbot = create_retriever_tool(
    retriever_waf,
    "retriever_waf_db_blog",
    "Search and run information about WAF & BOT protection"
)

# --- DB 2: Cybersecurity Infrastructure ---

cybersecurity_urls = [
    "https://www.splunk.com/en_us/blog/learn/siem-security-information-event-management.html",
    "https://stellarcyber.ai/learn/siem-vs-soc/",
    "https://www.todyl.com/blog/siem-incident-response"
]


cybersecurity_docs = [item for url in cybersecurity_urls for item in WebBaseLoader(url).load()]
cybersecurity_splits = splitter.split_documents(cybersecurity_docs)
vectorstore_cybersecurity = FAISS.from_documents(documents=cybersecurity_splits, embedding=embeddings_engine)
retriever_cybersecurity = vectorstore_cybersecurity.as_retriever()

retriever_tool_chatbot = create_retriever_tool(
    retriever_cybersecurity,
    "retriever_cybersecurity_blog",
    "Search and run information about Cybersecurity"
)

# Active Toolkit registration
tools = [retriever_tool_wafbot, retriever_tool_chatbot]

# ==============================================================================
# 2. CORE LLM & SCHEMAS DEFINITION
# ==============================================================================

# Consolidated Global Model definition to limit repeated construction calls
shared_llm = init_chat_model("groq:qwen/qwen3.6-27b")

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

class DocumentGrade(BaseModel):
    """Binary score for relevance checks."""
    binary_score: str = Field(description="Relevance score 'yes' or 'no'")

# ==============================================================================
# 3. WORKFLOW NODES & ROUTERS INTERPOLATION
# ==============================================================================

def agent(state: AgentState):
    print("---CALL AGENT---")
    messages = state["messages"]
    model_with_tools = shared_llm.bind_tools(tools)
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}

def grade_documents(state: AgentState) -> Literal["generate", "rewrite"]:
    print("---CHECK RELEVANCE---")
    
    llm_with_tool = shared_llm.with_structured_output(DocumentGrade)
    
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )
    
    chain = prompt | llm_with_tool
    messages = state["messages"]
    
    question = messages[0].content
    docs_content = messages[-1].content  # Extract text from latest ToolMessage response context
    
    scored_result = chain.invoke({"question": question, "context": docs_content})
    score = scored_result.binary_score
    
    if score == "yes":
        print("---DECISION: DOCS RELEVANT---")
        return "generate"
    else:
        print("---DECISION: DOCS NOT RELEVANT---")
        print(f"Confidence score evaluated as: {score}")
        return "rewrite"

def generate(state: AgentState):
    print("---GENERATE---")
    messages = state["messages"]
    question = messages[0].content
    docs_content = messages[-1].content

    prompt_template = hub.pull("rlm/rag-prompt")
    rag_chain = prompt_template | shared_llm | StrOutputParser()
    
    response = rag_chain.invoke({"context": docs_content, "question": question})
    return {"messages": [response]}

def rewrite(state: AgentState):
    print("---TRANSFORM QUERY---")
    messages = state["messages"]
    question = messages[0].content

    rewrite_message = [
        HumanMessage(content=f"""Look at the input and try to reason about the underlying semantic intent / meaning. \n 
        Here is the initial question:\n -------\n{question}\n -------\nFormulate an improved question: """)
    ]
    
    response = shared_llm.invoke(rewrite_message)
    return {"messages": [response]}

# ==============================================================================
# 4. GRAPH ARCHITECTURE COMPILATION
# ==============================================================================

workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("agent", agent)
workflow.add_node("retrieve", ToolNode(tools))  # Maps natively over our registered tools list
workflow.add_node("rewrite", rewrite)
workflow.add_node("generate", generate)

# Define Core Execution Pathways
workflow.add_edge(START, "agent")

workflow.add_conditional_edges(
    "agent",
    tools_condition,
    {
        "tools": "retrieve",
        END: END,
    },
)

workflow.add_conditional_edges(
    "retrieve",
    grade_documents,
)

workflow.add_edge("generate", END)
workflow.add_edge("rewrite", "agent")

graph = workflow.compile()

# 1. Save the file as a PNG
OUTPUT_IMAGE_PATH = "Image_PNGs/AgenticRAGDetailedProject_Grader.png"
graph.get_graph().draw_mermaid_png(output_file_path=OUTPUT_IMAGE_PATH)    
# 2. Automatically display/open the image on macOS NOTE _Just for testing purposes
# os.system(f"open {OUTPUT_IMAGE_PATH}")



# ==============================================================================
# 5. DIAGNOSTIC DIAGRAM PRODUCTION & INVOCATIONS
# ==============================================================================

# Execution runs
print("\n🚀 Executing CyberSecurity Query 1:")
# graph.invoke({"messages": [HumanMessage(content="What is Langgraph?")]})
# Question 1: Testing Role & Operational Boundaries (SOC vs. SIEM)
graph.invoke({"messages": [HumanMessage(content="What is the primary operational difference between a SIEM and a SOC according to Stellar Cyber?")]})

print("\n🚀 Executing CyberSecurity Query 2:")
# graph.invoke({"messages": [HumanMessage(content="What is Langchain?")]})
# Question 2: Testing Log Analysis & Capabilities (SIEM Technical Focus)
graph.invoke({"messages": [HumanMessage(content="How does a SIEM utilize event correlation and data normalization to identify potential security threats?")]})

print("\n🚀 Executing CyberSecurity Query 3:")
# Question 3: Testing Phase-Based Workflow (Incident Response Focus) NOTE - This is quetions purposefuly wrong
graph.invoke({"messages": [HumanMessage(content="What specific roles does a SIEM platform play during the Web Development and eradication phases of incident response?")]})

print("\n🛡️ Executing WebSecurity Query 1:")
# Question 1: User-Agent Spoofing & Fingerprinting Heuristics
graph.invoke({"messages": [HumanMessage(content="How does the Fastly Next-Gen WAF move beyond signature-based regex filters to unmask User-Agent spoofing, and what role do its behavioral signals and TLS/JA3 fingerprinting play in detecting these hidden bots?")]})

print("\n🛡️ Executing WebSecurity Query 2:")
# Question 2: Threat Taxonomy (Credential Stuffing vs. Brute Force)
graph.invoke({"messages": [HumanMessage(content="According to the Fastly threat breakdown, what is the core structural difference between a brute force attack and a credential stuffing attack, and why is credential stuffing considered a surgical subset?")]})

print("\n🛡️ Executing WebSecurity Query 3:")
# Question 3: Vulnerability Analysis & Real-World DDoS Response (CVE-2026-23869 & 'The Grinch' Attack) - NOTE - This is quetions purposefuly wrong
graph.invoke({"messages": [HumanMessage(content="Explain the mechanics of the high-severity CVE-2026-23869 DoS vulnerability in React Server Components, and describe how Fastly's automated DDoS protection handled 'The Grinch' botnet attack on Christmas Day 2025.")]})

