# ### Agentic RAG
# 🤖 What is Agentic RAG?

# Agentic RAG stands for Agentic Retrieval-Augmented Generation — an advanced version of RAG where instead of a static, one-shot LLM response,the system uses an agent that:

# - reasons,
# - plans,
# - retrieves,
# - uses tools,
# - and even retries or reflects
# to generate better, more grounded answers.

# NOTE sec16a_1-agenticrag.ipynb - Section 16 - 91  Build a Basic RAG with Langgraph

import os
from typing import List, Annotated
from pydantic import BaseModel

# Set a custom User-Agent identifying your application
os.environ["USER_AGENT"] = "Agentic-RAG-Cookbook/1.0 (contact: ashnaiku@codeaiwashnaiku.com)"


from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langgraph.graph import StateGraph, END


import os
from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
# llm=init_chat_model("openai:gpt-4o")
llm=init_chat_model("openai:gpt-4o-mini")

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
loaders = [WebBaseLoader(url) for url in urls]
docs = []
for loader in loaders:
    docs.extend(loader.load())

## Recursive character text ssplitter an vectorstore
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(docs)

embedding = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=512
)

vectorstore = FAISS.from_documents(split_docs, embedding)
retriever = vectorstore.as_retriever()

retriever.invoke("What are agents")

# -----------------------------
# 2. Define RAG State
# -----------------------------

class RAGState(BaseModel):
    question: str
    retrieved_docs: List[Document] = []
    answer: str = ""

# -----------------------------
# 3. LangGraph Nodes
# -----------------------------

def retrieve_docs(state: RAGState) -> RAGState:
    docs = retriever.invoke(state.question)
    return RAGState(question=state.question, retrieved_docs=docs)

def generate_answer(state: RAGState) -> RAGState:
    
    context = "\n\n".join([doc.page_content for doc in state.retrieved_docs])
    prompt = f"Answer the question based on the context.\n\nContext:\n{context}\n\nQuestion: {state.question}"
    response = llm.invoke(prompt)
    return RAGState(question=state.question, retrieved_docs=state.retrieved_docs, answer=response.content)


# -----------------------------
# 4. Build LangGraph
# -----------------------------

builder = StateGraph(RAGState)

builder.add_node("retriever", retrieve_docs)
builder.add_node("responder", generate_answer)

builder.set_entry_point("retriever")
builder.add_edge("retriever", "responder")
builder.add_edge("responder", END)

graph = builder.compile()

# TODO: Add diagram to display the graph
# -----------------------------
# 5. Run the Agentic RAG
# -----------------------------

if __name__ == "__main__":
    # This question forces the RAG to synthesize data regarding WAF capabilities, 
    # behavioral analysis, and computational client challenges from the articles.
    user_question = (
        "What are the primary differences between traditional rule-based WAF filtering "
        "and behavioral-based Bot Management? Specifically, how do modern systems use "
        "silent client-side challenges (like proof-of-work) to stop evasive scrapers?"
    )
    initial_state = RAGState(question=user_question)
    final_state = graph.invoke(initial_state)

    print("\n✅ Final Answer:\n", final_state['answer'])

print("\n✅ Retrieved Documents:")
for doc in final_state['retrieved_docs']:
    print(f"- {doc.metadata.get('source', 'unknown source')}: {doc.page_content[:200]}...") 
    