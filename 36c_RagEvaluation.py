"""
================================================================================
FILE: 36c_RagEvaluation.py
================================================================================

WHAT THIS FILE ADDS COMPARED TO 36b_RagEvaluation.py:
-----------------------------------------------------
1. LANGSMITH AUTOMATION & TRACING:
   - Replaces custom Python `for` loops with LangSmith's native evaluation framework 
     (`client.evaluate()`).
   - Integrates `@traceable()` decorators to monitor chain executions, latency, 
     and tokens directly in the LangSmith dashboard.

2. PERSISTENT TEST DATASETS & GROUND-TRUTH CORRECTNESS:
   - 36b_RagEvaluation.py evaluated without ground-truth answers (Reference-Free).
   - This file introduces LangSmith Datasets containing explicit "Ground Truth" 
     answers, allowing us to evaluate factual **Correctness** (Predicted Answer vs. Reference Answer).

3. WEB-BASED KNOWLEDGE INGESTION:
   - Upgrades from hardcoded text strings to live web page scraping using `WebBaseLoader` 
     to build a realistic production-grade vector database retriever.

4. REAL-TIME EXPERIMENT COMPARISONS:
   - Runs experiment sweeps (`experiment_prefix`) allowing developers to compare 
     different model architectures or prompts side-by-side in LangSmith.
================================================================================
"""

import os
from typing_extensions import Annotated, TypedDict
from dotenv import load_dotenv

from langsmith import Client, traceable
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Load environment variables
load_dotenv()
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"

# Initialize LangSmith Client - Synchronous client for interacting with the LangSmith API
# https://reference.langchain.com/python/langsmith/client
client = Client()

# ==============================================================================
# 1. KNOWLEDGE BASE & RAG PIPELINE SETUP
# ==============================================================================

# The `retriever` searches your vector database (which loaded Lilian Weng's blog post URLs) to find relevant document chunks.
# Retrieved chunks are formatted into a context string and passed directly to `llm.invoke()` to generate the answer `ai_msg.content`.

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

print("--- LOADING WEB DOCUMENTS & BUILDING VECTOR STORE ---")
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
doc_splits = text_splitter.split_documents(docs_list)

embedding = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=512)
vectorstore = InMemoryVectorStore.from_documents(documents=doc_splits, embedding=embedding)
retriever = vectorstore.as_retriever(k=4)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Decorate the target RAG bot so LangSmith traces all calls automatically
# https://reference.langchain.com/python/langsmith/run_helpers/traceable
@traceable()
def rag_bot(inputs: dict) -> dict:
    """RAG pipeline function passed into LangSmith evaluator."""
    question = inputs["question"]
    docs = retriever.invoke(question)
    docs_string = "\n\n".join(doc.page_content for doc in docs)

    instructions = (
        "You are a helpful assistant. Use the following source documents to answer the user's question.\n"
        "If you don't know the answer, just say that you don't know. Keep the answer concise.\n\n"
        f"Documents:\n{docs_string}"
    )

    ai_msg = llm.invoke([
        {"role": "system", "content": instructions},
        {"role": "user", "content": question},
    ])
    
    # Return both the answer and the retrieved documents for evaluator inspection
    return {"answer": ai_msg.content, "documents": docs}

# ==============================================================================
# 2. CREATE LANGSMITH DATASET WITH GROUND TRUTH
# ==============================================================================

# `dataset_name`` does NOT provide the context used to generate the answers. Instead, it acts as a test bank or benchmark.abs
# It supplies the ground-truth reference answers `outputs: {"answer": ...}`
# Used after generation to grade whether the answer returned by `rag_bot` was accurate.

#NOTE Look for this name in LANGSMITH https://smith.langchain.com/
dataset_name = "RAG Production Evaluation Dataset"

# Avoid duplicate dataset creation if re-running script
if not client.has_dataset(dataset_name=dataset_name):
    examples = [
        {
            "inputs": {"question": "How does the ReAct agent use self-reflection?"},
            "outputs": {"answer": "ReAct integrates reasoning and acting, performing actions like tool calls and observing/reasoning about outputs."},
        },
        {
            "inputs": {"question": "What are the types of biases that can arise with few-shot prompting?"},
            "outputs": {"answer": "Biases include (1) Majority label bias, (2) Recency bias, and (3) Common token bias."},
        },
        {
            "inputs": {"question": "What are five types of adversarial attacks?"},
            "outputs": {"answer": "Five types are (1) Token manipulation, (2) Gradient based attack, (3) Jailbreak prompting, (4) Human red-teaming, (5) Model red-teaming."},
        }
    ]
    # https://reference.langchain.com/python/langsmith/client/Client/create_dataset
    dataset = client.create_dataset(dataset_name=dataset_name)
    client.create_examples(dataset_id=dataset.id, examples=examples)
    print(f"Created LangSmith Dataset: '{dataset_name}'")
else:
    print(f"Using existing LangSmith Dataset: '{dataset_name}'")

# ==============================================================================
# 3. DEFINE LANGSMITH EVALUATORS (LLM-AS-A-JUDGE)
# ==============================================================================

# ---  Grader 1: Correctness (Predicted Answer vs. Ground Truth Reference Answer) --- 
class CorrectnessGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    correct: Annotated[bool, ..., "True if the answer is correct relative to ground truth"]

correctness_instructions = """You are a teacher grading a quiz.
Grade the student answer based ONLY on its factual accuracy relative to the GROUND TRUTH answer."""

correctness_llm = llm.with_structured_output(
    CorrectnessGrade, method="json_schema", strict=True
)

def correctness_evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    content = f"QUESTION: {inputs['question']}\nGROUND TRUTH: {reference_outputs['answer']}\nSTUDENT ANSWER: {outputs['answer']}"
    grade = correctness_llm.invoke([
        {"role": "system", "content": correctness_instructions},
        {"role": "user", "content": content}
    ])
    return grade["correct"]

# --- Grader 2: Groundedness / Faithfulness (Predicted Answer vs. Retrieved Context) --- 
class GroundedGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    grounded: Annotated[bool, ..., "True if answer is grounded in facts without hallucination"]

grounded_instructions = """Ensure the STUDENT ANSWER is supported by the FACTS without hallucinations."""
grounded_llm = llm.with_structured_output(
    GroundedGrade, method="json_schema", strict=True
)

def groundedness_evaluator(inputs: dict, outputs: dict) -> bool:
    doc_string = "\n\n".join(doc.page_content for doc in outputs["documents"])
    content = f"FACTS: {doc_string}\nSTUDENT ANSWER: {outputs['answer']}"
    grade = grounded_llm.invoke([
        {"role": "system", "content": grounded_instructions},
        {"role": "user", "content": content}
    ])
    return grade["grounded"]

# ==============================================================================
# 4. RUN AUTOMATED LANGSMITH EVALUATION
# ==============================================================================

print("\n--- RUNNING BATCH EVALUATION IN LANGSMITH ---")

experiment_results = client.evaluate(
    rag_bot,
    data=dataset_name,
    evaluators=[correctness_evaluator, groundedness_evaluator],
    experiment_prefix="rag-production-eval",
    metadata={"model": "gpt-4o-mini", "vectorstore": "InMemoryVectorStore"}
)

print("\nEvaluation complete! Check your LangSmith dashboard to review the full metrics trace.")