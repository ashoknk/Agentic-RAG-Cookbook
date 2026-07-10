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

🎯 Why It’s Needed
Most real-world queries are:
- Multifaceted (require multiple types of information)
- Ambiguous or incomplete (need refinement)
- Open-ended (don’t map to a single document or source)

🔍 This makes retrieving from a single vector DB insufficient.
Instead, we want an agent that can:
- Decide what to fetch from where (retrieval planning)
- Retrieve content from multiple tools (e.g., Wikipedia, PDFs, APIs, SQL)
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

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader,WebBaseLoader
from langchain_community.document_loaders.youtube import YoutubeLoader
# from langchain_community.document_loaders import ArxivLoader
from langchain_community.tools import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper


from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

from langgraph.graph import StateGraph, END

# ---------------------------------------------------------------------
# Environment Setup and Model Initialization
# ---------------------------------------------------------------------
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize chat model
llm = init_chat_model("openai:gpt-4o-mini")


# ---------------------------------------------------------------------
# Retriever Setup & Utility Search Functions
# ---------------------------------------------------------------------
def load_text_retriever(file_path):
    docs = TextLoader(file_path, encoding="utf-8").load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    embedding = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=512)
    vs = FAISS.from_documents(chunks, embedding)
    return vs.as_retriever()

def load_youtube_retriever():
    # Mocked YouTube transcript text
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


def wikipedia_search(query: str) -> str:
    print("🌐 Searching Wikipedia...")
    try:
        api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
        wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)
        return wiki.run(query)
    except Exception as e:
        # 💡 Catch any networking or decoding errors gracefully
        print(f"⚠️ Wikipedia API encountered an issue: {e}")
        return "No relevant information found on Wikipedia."

def arxiv_search(query: str) -> str:
    print("📄 Searching ArXiv...")
    try:
        # Try running the native LangChain tool
        return arxiv_client.run(query)
    except Exception as e:
        # 💡 Catch the error and return a clean fallback string 
        # so the LLM doesn't read a Python error message as real context.
        return "No relevant academic papers found on ArXiv for this topic."

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
    wiki_context: str = ""
    arxiv_context: str = ""
    final_answer: str = ""


# ---------------------------------------------------------------------
# Graph Retrieval Nodes
# ---------------------------------------------------------------------
def retrieve_text(state: MultiSourceRAGState) -> MultiSourceRAGState:
    docs = text_retriever.invoke(state.question)
    return state.model_copy(update={"text_docs": docs})

def retrieve_yt(state: MultiSourceRAGState) -> MultiSourceRAGState:
    docs = youtube_retriever.invoke(state.question)
    return state.model_copy(update={"yt_docs": docs})

def retrieve_wikipedia(state: MultiSourceRAGState) -> MultiSourceRAGState:
    result = wikipedia_search(state.question)
    return state.model_copy(update={"wiki_context": result})

def retrieve_arxiv(state: MultiSourceRAGState) -> MultiSourceRAGState:
    result = arxiv_search(state.question)
    return state.model_copy(update={"arxiv_context": result})


# ---------------------------------------------------------------------
# Synthesis Node
# ---------------------------------------------------------------------
def synthesize_answer(state: MultiSourceRAGState) -> MultiSourceRAGState:
    context = ""
    context += "\n\n[Internal Docs]\n" + "\n".join([doc.page_content for doc in state.text_docs])
    context += "\n\n[YouTube Transcript]\n" + "\n".join([doc.page_content for doc in state.yt_docs])
    context += "\n\n[Wikipedia]\n" + state.wiki_context
    context += "\n\n[ArXiv]\n" + state.arxiv_context

    prompt = f"""You have retrieved relevant context from multiple sources. Now synthesize a complete and coherent answer.

Question: {state.question}

Context:
{context}

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
builder.add_node("retrieve_wiki", retrieve_wikipedia)
builder.add_node("retrieve_arxiv", retrieve_arxiv)
builder.add_node("synthesize", synthesize_answer)

# Linear execution sequence pipeline layout
builder.set_entry_point("retrieve_text")
builder.add_edge("retrieve_text", "retrieve_yt")
builder.add_edge("retrieve_yt", "retrieve_wiki")
builder.add_edge("retrieve_wiki", "retrieve_arxiv")
builder.add_edge("retrieve_arxiv", "synthesize")
builder.add_edge("synthesize", END)

# Compile graph configuration
graph = builder.compile()

# Save the file as a PNG
OUTPUT_IMAGE_PATH = "Image_PNGs/AnswerSynthesis.png"
graph.get_graph().draw_mermaid_png(output_file_path=OUTPUT_IMAGE_PATH)    
# Automatically display/open the image on macOS NOTE _Just for testing purposes
# os.system(f"open {OUTPUT_IMAGE_PATH}")

# ---------------------------------------------------------------------
# Execution Entry Point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    question = "What is a Web Application Firewall. Explain with examples as Fastly Next-Gen WAF . How are they evolving in recent research?"
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

    # 3. Wikipedia Preview
    wiki = result.get("wiki_context", "").strip()
    if wiki and "No relevant information" not in wiki:
        snippet = wiki[:80].replace('\n', ' ') + "..."
        print(f"🔹 [Wikipedia] Refreshed context. Snippet: \"{snippet}\"")
    else:
        print("🔹 [Wikipedia] No context matched.")

    # 4. ArXiv Preview
    arxiv = result.get("arxiv_context", "").strip()
    if arxiv and "No relevant academic papers" not in arxiv:
        snippet = arxiv[:80].replace('\n', ' ') + "..."
        print(f"🔹 [ArXiv Academic] Found paper details. Snippet: \"{snippet}\"")
    else:
        print("🔹 [ArXiv Academic] No papers matched.")
        
    print("====================================================================")