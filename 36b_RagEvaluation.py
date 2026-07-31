"""
================================================================================
FILE: 36_RagEvaluation_Intermediate.py
================================================================================

WHAT THIS FILE ADDS COMPARED TO 36_RagEvaluation.py:
----------------------------------------------------
1. FROM STATIC STRINGS TO AN ACTUAL RAG PIPELINE:
   - 36_RagEvaluation.py used hardcoded, static text strings (`test_context`, `test_answer`).
   - This file builds a lightweight, real-working RAG pipeline (using an InMemoryVectorStore 
     and document retriever) so you can evaluate dynamically retrieved context.

2. EXPANSION TO THE COMPLETE "RAG TRIAD" METRICS:
   - 36_RagEvaluation.py only evaluated a single metric: Faithfulness.
   - This file introduces all 3 pillars of the RAG Triad:
       a) Faithfulness / Groundedness (Is answer supported ONLY by retrieved context?)
       b) Answer Relevance (Does the answer directly address the user's question?)
       c) Context / Retrieval Relevance (Did the retriever fetch context relevant to the question?)

3. BATCH PROCESSING ON MULTIPLE TEST CASES:
   - Instead of evaluating just 1 question, this file loops through a list of test questions 
     (a mini test dataset) and aggregates evaluation scores using standard Python functions.

4. STEP-BY-STEP PREPARATION FOR LANGSMITH:
   - This file keeps everything local using standard LangChain chains so you can understand 
     LLM-as-a-Judge logic clearly 
================================================================================
"""

import os
from typing import TypedDict, List
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

# Load environment configuration
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# ==============================================================================
# 1. SETUP A MINIMAL RAG SYSTEM
# ==============================================================================

# Knowledge base documents
documents_text = [
    "The capital of Japan is Tokyo. Tokyo is famous for its ultra-modern urban landscape and landmarks like Shibuya Crossing.",
    "Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability.",
    "Retrieval-Augmented Generation (RAG) is a technique for enhancing LLM responses by fetching facts from external databases."
]

# Text splitting and Vector Store setup
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
doc_splits = text_splitter.create_documents(documents_text)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=512)
vectorstore = InMemoryVectorStore.from_documents(documents=doc_splits, embedding=embeddings)
retriever = vectorstore.as_retriever(k=2)

# Simple RAG Chain function
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def run_rag_pipeline(question: str) -> dict:
    """Runs retrieval and generation for a user question."""
    retrieved_docs = retriever.invoke(question)
    context_text = "\n".join([doc.page_content for doc in retrieved_docs])
    
    prompt = ChatPromptTemplate.from_template(
        "Answer the question using ONLY the following context:\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "Answer:"
    )
    chain = prompt | llm
    response = chain.invoke({"context": context_text, "question": question})
    
    return {
        "question": question,
        "context": context_text,
        "answer": response.content
    }

# ==============================================================================
# 2. DEFINE EVALUATION SCHEMAS (LLM-AS-A-JUDGE)
# ==============================================================================

class MetricScore(BaseModel):
    passed: bool = Field(description="True if the criteria is fully met, False otherwise.")
    # score: int = Field(description="Score from 1 to 5, where 5 is perfectly faithful to context")
    reason: str = Field(description="Brief explanation for the judgment.")

# Evaluator 1: Faithfulness (Is the answer grounded in context?)
faithfulness_prompt = ChatPromptTemplate.from_template(
    "Evaluate if the ANSWER is fully supported by the provided CONTEXT. "
    "Do not assume facts not present in context.\n\n"
    "Context: {context}\n"
    "Answer: {answer}"
)
faithfulness_evaluator = faithfulness_prompt | llm.with_structured_output(MetricScore)

# Evaluator 2: Answer Relevance (Does the answer address the user question?)
relevance_prompt = ChatPromptTemplate.from_template(
    "Evaluate if the ANSWER directly and concisely addresses the QUESTION.\n\n"
    "Question: {question}\n"
    "Answer: {answer}"
)
relevance_evaluator = relevance_prompt | llm.with_structured_output(MetricScore)

# Evaluator 3: Retrieval Relevance (Is the retrieved context relevant to the question?)
context_relevance_prompt = ChatPromptTemplate.from_template(
    "Evaluate if the CONTEXT contains information relevant to answering the QUESTION.\n\n"
    "Question: {question}\n"
    "Context: {context}"
)
context_relevance_evaluator = context_relevance_prompt | llm.with_structured_output(MetricScore)

# ==============================================================================
# 3. RUN EVALUATION OVER A TEST DATASET
# ==============================================================================

test_questions = [
    "What is the capital of Japan?",
    "What is Python known for?",
    "Who is the current Prime Minister of Japan?" # Out-of-knowledge query test
]

# ANSI Color Codes
# GREEN = "\033[92m"
BLUE = "\033[94m"  
RED = "\033[91m"
RESET = "\033[0m"

# Helper function to format status
def format_status(passed: bool) -> str:
    return f"{BLUE}PASSED{RESET}" if passed else f"{RED}FAILED{RESET}"


print("--- STARTING LOCAL BATCH RAG EVALUATION ---\n")

for idx, q in enumerate(test_questions, start=1):
    print(f"=== TEST CASE {idx}: '{q}' ===")
    
    # 1. Run RAG Pipeline
    rag_result = run_rag_pipeline(q)
    context_text = rag_result['context']
    half_length = len(context_text) // 2
    print(f"Retrieved Context:\n{context_text[:half_length]}...")
    print(f"Generated Answer: {rag_result['answer']}\n")
    
    # 2. Evaluate Faithfulness
    f_score: MetricScore = faithfulness_evaluator.invoke({
        "context": rag_result["context"],
        "answer": rag_result["answer"]
    })
    
    # 3. Evaluate Answer Relevance
    a_score: MetricScore = relevance_evaluator.invoke({
        "question": rag_result["question"],
        "answer": rag_result["answer"]
    })
    
    # 4. Evaluate Context Relevance
    c_score: MetricScore = context_relevance_evaluator.invoke({
        "question": rag_result["question"],
        "context": rag_result["context"]
    })
    
    # Print Report
    print("EVALUATION RESULTS:")
    # print(f" - Faithfulness: {'PASSED' if f_score.passed else 'FAILED'} | Reason: {f_score.reason}")
    # print(f" - Answer Relevance: {'PASSED' if a_score.passed else 'FAILED'} | Reason: {a_score.reason}")
    # print(f" - Context Relevance: {'PASSED' if c_score.passed else 'FAILED'} | Reason: {c_score.reason}")
    # print("-" * 60 + "\n")
    print(f" - Faithfulness: {format_status(f_score.passed)} | Reason: {f_score.reason}")
    print(f" - Answer Relevance: {format_status(a_score.passed)} | Reason: {a_score.reason}")
    print(f" - Context Relevance: {format_status(c_score.passed)} | Reason: {c_score.reason}")
    print("-" * 60 + "\n")