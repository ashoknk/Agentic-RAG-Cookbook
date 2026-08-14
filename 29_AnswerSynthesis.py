"""
=====================================================================
🧠 Answer Synthesis from Multiple Sources
=====================================================================
✅ What Is It?
Answer synthesis from multiple sources is the process where an AI agent 
collects information from different retrieval tools or knowledge bases, 
and merges that information into a single, coherent, and contextually rich answer.

This is a core capability in Agentic RAG, where the system is more than 
just a simple retriever — it plans, retrieves, and then synthesizes an 
answer that draws from multiple sources.

🎯 Why It’s Needed .Most real-world queries are:
- Multifaceted (require multiple types of information)
- Ambiguous or incomplete (need refinement)
- Open-ended (don’t map to a single document or source)

🔍 This makes retrieving from a single vector DB insufficient.
Instead, we want an agent that can:
- Decide what to fetch from where (retrieval planning)
- Retrieve content from multiple tools (e.g., TavilySearchResults, DuckDuckGoSearch, Text, PDFs, APIs, SQL)
- Evaluate and merge that context
- Produce a single human-like response

"""
import os
# Set a custom User-Agent identifying your application
os.environ["USER_AGENT"] = "Agentic-RAG-Cookbook/1.0 (contact: ash@codeaiwashnaiku.com)"

from typing import List
from pydantic import BaseModel
from dotenv import load_dotenv

# Suppress the specific LangChain serialization warning
import warnings
from langchain_core._api.deprecation import LangChainPendingDeprecationWarning
warnings.filterwarnings("ignore", category=LangChainPendingDeprecationWarning)

from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader,WebBaseLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import TavilySearchResults

# ---------------------------------------------------------------------
# Environment Setup and Model Initialization
# ---------------------------------------------------------------------
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

# Initialize chat model
llm = init_chat_model("openai:gpt-4o-mini")


# ---------------------------------------------------------------------
# Retriever Setup & Utility Search Functions
# ---------------------------------------------------------------------
def load_text_retriever(file_path):
    print("📄 Loading Internal Text Files...")
    docs = TextLoader(file_path, encoding="utf-8").load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    embedding = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=512)
    vs = FAISS.from_documents(chunks, embedding)
    return vs.as_retriever()


def load_youtube_retriever():
    # Mocked YouTube transcript text
    print("🎥 Loading Youtube...")
    content = """
    This video explains how at its core, a Web Application Firewall (WAF) operates as a security gateway positioned between external 
    clients and web application origin servers to inspect and filter application-layer (Layer 7) traffic. 
    Unlike traditional network firewalls that monitor transport-layer protocols and IP addresses, 
    a WAF is protocol-aware, deeply analyzing HTTP and HTTPS request-response cycles. 
    WAFs are commonly deployed in one of three architectural topologies: as a reverse proxy where all inbound traffic is 
    routed through the firewall before reaching the application, as a transparent bridge/inline appliance, 
    or via cloud-based edge delivery networks. By analyzing HTTP headers, query parameters, cookies, and POST bodies, 
    the firewall enforces access control policies, decodes transfer encodings, and strips out anomalous payloads before they can interact with the underlying application logic.
    """
    doc = Document(page_content=content, metadata={"source": "youtube"})
    embedding = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=512)
    vectorstore = FAISS.from_documents([doc], embedding)
    return vectorstore.as_retriever()


def ddg_search(query: str) -> str:
    print("🌐 Loading DuckDuckGo...")
    try:
        ddg = DuckDuckGoSearchRun()
        return ddg.run(query)
    except Exception as e:
        print(f"⚠️ DuckDuckGo search encountered an issue: {e}")
        return "No relevant information found on DuckDuckGo."


def tavily_search(query: str) -> str:
    print("🔍 Loading Tavily...")
    try:
        # Requires TAVILY_API_KEY set in your .env file
        tavily = TavilySearchResults(k=2)
        results = tavily.run(query)
        return str(results)
    except Exception as e:
        print(f"⚠️ Tavily search encountered an issue: {e}")
        return "No relevant information found on Tavily."


# Instantiate primary local retrievers
PRIVATE_DOCS_PATH = "cybersecurity_data/private_docs.txt"

text_retriever = load_text_retriever(PRIVATE_DOCS_PATH)
youtube_retriever = load_youtube_retriever()


# ---------------------------------------------------------------------
# State Management Definition
# ---------------------------------------------------------------------
class MultiSourceRAGState(BaseModel):
    question: str
    text_docs: List[Document] = []
    yt_docs: List[Document] = []
    ddg_context: str = ""     
    tavily_context: str = ""  
    final_answer: str = ""


# ---------------------------------------------------------------------
# Graph Retrieval Nodes
# ---------------------------------------------------------------------
# `model_copy` preserves immutability and ensures that LangGraph correctly tracks state transitions 
# from node to node without modifying history in unexpected ways
# https://pydantic.dev/docs/validation/2.4/concepts/models/

def retrieve_text(state: MultiSourceRAGState) -> MultiSourceRAGState:
    docs = text_retriever.invoke(state.question)
    return state.model_copy(update={"text_docs": docs})

def retrieve_yt(state: MultiSourceRAGState) -> MultiSourceRAGState:
    docs = youtube_retriever.invoke(state.question)
    return state.model_copy(update={"yt_docs": docs})

def retrieve_ddg(state: MultiSourceRAGState) -> MultiSourceRAGState:
    result = ddg_search(state.question)
    return state.model_copy(update={"ddg_context": result})

def retrieve_tavily(state: MultiSourceRAGState) -> MultiSourceRAGState:
    result = tavily_search(state.question)
    return state.model_copy(update={"tavily_context": result})


# ---------------------------------------------------------------------
# Synthesis Node
# ---------------------------------------------------------------------
def synthesize_answer(state: MultiSourceRAGState) -> MultiSourceRAGState:
    context = ""
    context += "\n\n[Internal Docs]\n" + "\n".join([doc.page_content for doc in state.text_docs])
    context += "\n\n[YouTube Transcript]\n" + "\n".join([doc.page_content for doc in state.yt_docs])
    context += "\n\n[DuckDuckGo]\n" + state.ddg_context
    context += "\n\n[Tavily]\n" + state.tavily_context

    prompt = f"""You have retrieved relevant context from multiple sources. Now synthesize a complete and coherent answer.

Question: {state.question}
Context:{context}

Final Answer:"""
    answer = llm.invoke(prompt).content.strip()
    return state.model_copy(update={"final_answer": answer})
    

# ---------------------------------------------------------------------
# Build and Compile LangGraph Framework
# ---------------------------------------------------------------------
builder = StateGraph(MultiSourceRAGState)

# Add processing nodes to graph
builder.add_node("retrieve_text", retrieve_text)
builder.add_node("retrieve_yt", retrieve_yt)
builder.add_node("retrieve_ddg", retrieve_ddg)
builder.add_node("retrieve_tavily", retrieve_tavily)
builder.add_node("synthesize", synthesize_answer)

# Linear execution sequence pipeline layout
builder.set_entry_point("retrieve_text")
builder.add_edge("retrieve_text", "retrieve_yt")
builder.add_edge("retrieve_yt", "retrieve_ddg")
builder.add_edge("retrieve_ddg", "retrieve_tavily")
builder.add_edge("retrieve_tavily", "synthesize") 
builder.add_edge("synthesize", END)

# Compile graph configuration
graph = builder.compile()

# Save the file as a PNG
OUTPUT_IMAGE_PATH = "Image_PNGs/AnswerSynthesis.png"
graph.get_graph().draw_mermaid_png(output_file_path=OUTPUT_IMAGE_PATH)    
# Automatically display/open the image on macOS NOTE _Just for testing purposes
os.system(f"open {OUTPUT_IMAGE_PATH}")


# ---------------------------------------------------------------------
# Execution Entry Point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # question = "What is a Web Application Firewall. Explain with examples as Fastly Next-Gen WAF . How are they evolving in recent research?"
    question = "Web Application Firewall Fastly Next Gen WAF research"
    state = MultiSourceRAGState(question=question)
    
    print(f"Starting Multi-Source Agent workflow for query: '{question}'\n")
    result = graph.invoke(state)
    
    print("\n======================= SYNTHESIZED RESULTS =======================")
    print("✅ Final Answer:\n")
    print(result["final_answer"])
    print("====================================================================")

    # NOTE - Just for testing purposes
    print("\n📚 PROVENANCE & SOURCES USED:")
    print("--------------------------------------------------------------------")
    
    # 1. Local Text Docs Snippet
    if result.get("text_docs"):
        # Grab just the first chunk as a snippet sample
        first_doc = result["text_docs"][0].page_content.strip().replace('\n', ' ')
        snippet = first_doc[:80] + "..." if len(first_doc) > 80 else first_doc
        print(f"🔹 [Internal Docs] Used {len(result['text_docs'])} chunk(s). Snippet: \"{snippet}\"")
    else:
        print("🔹 [Internal Docs] No relevant chunks found.")

    # 2. YouTube Transcript Snippet
    if result.get("yt_docs"):
        first_yt = result["yt_docs"][0].page_content.strip().replace('\n', ' ')
        snippet = first_yt[:80] + "..." if len(first_yt) > 80 else first_yt
        print(f"🔹 [YouTube Video] Used transcript. Snippet: \"{snippet}\"")
    else:
        print("🔹 [YouTube Video] No relevant transcripts found.")

    # 3. DuckDuckGo Preview
    ddg = result.get("ddg_context", "").strip()
    if ddg and "No relevant information" not in ddg:
        snippet = ddg[:80].replace('\n', ' ') + "..."
        print(f"🔹 [DuckDuckGo] Refreshed context. Snippet: \"{snippet}\"")
    else:
        print("🔹 [DuckDuckGo] No context matched.")

    # 4. Tavily Preview
    tavily = result.get("tavily_context", "").strip()
    if tavily and "No relevant information" not in tavily:
        snippet = tavily[:80].replace('\n', ' ') + "..."
        print(f"🔹 [Tavily] Found search details. Snippet: \"{snippet}\"")
    else:
        print("🔹 [Tavily] No results matched.")

        
    print("====================================================================")


    
