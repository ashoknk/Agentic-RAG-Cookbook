"""
### 🧠 What is Query Decomposition?
Query decomposition is the process of taking a complex, multi-part question and 
breaking it into simpler, atomic sub-questions that can each be retrieved and answered individually.

#### ✅ Why Use Query Decomposition?
- Complex queries often involve multiple concepts
- LLMs or retrievers may miss parts of the original question
- It enables multi-hop reasoning (answering in steps)
- Allows parallelism (especially in multi-agent frameworks)
================================================================================
This script introduces **Query Decomposition**, a multi-hop reasoning strategy 
designed to conquer complex, multi-part questions. When a question contains 
cross-framework comparisons or multi-layered tasks, standard retrieval chains 
often suffer from information dilution. This workflow leverages an LLM to 
break a single problem down into an array of isolated, simple sub-questions.


1. INGESTION & SEARCH ENGINE SETUP: Ingests reference materials, indexing them 
   into a local FAISS vector store bound to a diversity-optimized MMR retriever.
2. ATOMIC DECOMPOSITION LAYER: Directs a specialized LLM sequence to analyze a 
   complex prompt (e.g., comparing memory and agents across LangChain and CrewAI) 
   and decompose it into 2-4 individual sub-questions.
3. SEQUENTIAL INVOCATION LOOP: Sanitizes and splits the generated list into a clean 
   Python array. The script iterates through the list sequentially, launching 
   separate retrieval and answer-generation queries for each sub-question.
4. REPORT FUSION: Accumulates the individually grounded answers, merging them 
   into a comprehensive, final response framework.
================================================================================
"""

import os
import logging
import warnings
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import  TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

# Suppress standard python warnings
warnings.filterwarnings("ignore")
# Suppress Transformer loading logs and general warnings
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
# Suppress the progress bar (The 100%|███ bar)
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# Step 1: Load and embed the document
# TODO change question to cybersecurity_data if needed
FILE_NAME = "cybersecurity_data/langchain_crewai_dataset.txt"
loader = TextLoader(FILE_NAME)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(docs)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vectorstore = FAISS.from_documents(chunks, embedding)
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 4, "lambda_mult": 0.7})


os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")

CHAT_MODEL="groq:llama-3.1-8b-instant"
llm=init_chat_model(model=CHAT_MODEL)
# Step 3: Query decomposition
decomposition_prompt = PromptTemplate.from_template("""
You are an AI assistant. Decompose the following complex question into 2 to 4 smaller sub-questions for better document retrieval.

Question: "{question}"

Sub-questions:
""")
decomposition_chain = decomposition_prompt | llm | StrOutputParser()

# NOTE: Just for testing 
# query = "How does LangChain use memory and agents compared to CrewAI?"
# decomposition_question=decomposition_chain.invoke({"question": query})
# print(decomposition_question)


# Step 4: QA chain per sub-question
qa_prompt = PromptTemplate.from_template("""
Use the context below to answer the question.

Context:
{context}

Question: {input}
""")
qa_chain = create_stuff_documents_chain(llm=llm, prompt=qa_prompt)

# Step 5: Full RAG pipeline logic
# def full_query_decomposition_rag_pipeline(user_query):
#     # Decompose the query
#     print(f"🔍 Question :{user_query}\n")
#     sub_qs_text = decomposition_chain.invoke({"question": user_query})
#     # strip away any numbered list formats (like 1., 2., 3.) that the LLM generates 
#     sub_questions = [q.strip("-•1234567890. ").strip() for q in sub_qs_text.split("\n") if q.strip()]
    
#     results = []
#     for subq in sub_questions:
#         docs = retriever.invoke(subq)
#         result = qa_chain.invoke({"input": subq, "context": docs})
#         results.append(f"Q: {subq}\nA: {result}")
    
#     return "\n\n".join(results)
def full_query_decomposition_rag_pipeline(user_query):
    print(f"========================================================")
    print(f"🔍 Processing Complex User Query: '{user_query}'")
    print(f"========================================================")
    
    # 1. Generate the sub-questions from the main query
    sub_qs_text = decomposition_chain.invoke({"question": user_query})
    
    print("\n✨ Step 1: Query Enhancement (Decomposed Sub-Questions):")
    print(sub_qs_text.strip())
    print("-" * 50)
    
    # Clean up and split the text into a clean Python list
    sub_questions = [q.strip("-•1234567890. ").strip() for q in sub_qs_text.split("\n") if q.strip()]
    
    # 2. Loop through individual sub-questions sequentially
    results = []
    for i, subq in enumerate(sub_questions, 1):
        # Retrieve diverse chunks using MMR (k=4, lambda=0.7)
        docs = retriever.invoke(subq)
        
        # Answer the atomic sub-question completely
        answer = qa_chain.invoke({"input": subq, "context": docs})
        results.append(f"🧩 Sub-Task {i}: {subq}\n💡 Grounded Answer: {answer}")
    
    return "\n\n".join(results)

# Step 6: Run
# query = "How does LangChain use memory and agents compared to CrewAI?"
# final_answer = full_query_decomposition_rag_pipeline(query)
# print("✅ Final Answer:\n")
# print(final_answer)
if __name__ == "__main__":
    complex_query = "How does LangChain use memory and agents compared to CrewAI?"
    final_report = full_query_decomposition_rag_pipeline(complex_query)
    
    print("\n✅ Final Combined Answer Framework:")
    print("=" * 50)
    print(final_report)