"""
RAG EVALUATION FRAMEWORK (RAGAS Style)

### Concept:
Evaluation measures the overall quality, accuracy, and trustworthiness of a 
Retrieval-Augmented Generation (RAG) system using key core metrics:

1. Faithfulness: 
   - Is the generated answer derived solely and accurately from the retrieved context?
2. Answer Relevance: 
   - Does the generated answer directly address the user's original question?
3. Context Precision: 
   - How relevant and noise-free are the retrieved documents relative to the query?


Complementary Role (Evaluation vs. Grading):
--------------------------------------------------------------------------------
This evaluation module works alongside grader scripts (e.g., 25c_AgenticRAG_Grader.py). 
They complement each other because they solve two fundamental, distinct problems 
in building RAG applications:

  - Agentic Graders (Inline / Runtime Control):
    Filter out bad retrievals and control workflow routing dynamically *during* 
    execution to prevent hallucinated generation.

  - RAGAS-Style Evaluators (Offline / Post-Hoc Audit):
    Provide structured scoring (e.g., 1–5 scale) and detailed reasoning to benchmark, 
    audit, and track overall system performance across datasets.
================================================================================
"""

import os
from typing import TypedDict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# ### 1. Define Evaluation Schemas
# Benchmarking, auditing, and measuring system performance. Quality metrics: Scores (e.g., 1 to 5) & detailed reasoning
# inspects - Final Generated Answer vs. Context.
class MetricScore(BaseModel):
    score: int = Field(description="Score from 1 to 5, where 5 is perfectly faithful to context")
    reason: str = Field(description="Reasoning for the score")

# ### 2. Define the State
class EvalState(TypedDict):
    question: str
    context: str
    answer: str
    faithfulness_result: MetricScore

# ### 3. Define Evaluation Node
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def evaluate_faithfulness(state: EvalState):
    print("---EVALUATING FAITHFULNESS---")
    structured_llm = llm.with_structured_output(MetricScore)
    
    prompt = ChatPromptTemplate.from_template(
        "Compare the answer to the provided context. Is the answer supported by the context?\n\n"
        "Context: {context}\n"
        "Answer: {answer}"
    )
    
    eval_chain = prompt | structured_llm
    result = eval_chain.invoke({"context": state["context"], "answer": state["answer"]})
    return {"faithfulness_result": result}

# ### 4. Run Evaluation
# Dummy data for evaluation
test_context = "The capital of Japan is Tokyo. Tokyo is famous for its ultra-modern urban landscape, historic temples, and world-class food scene, highlighted by landmarks like Shibuya Crossing, Senso-ji Temple, and its record number of Michelin-starred restaurants"
test_answer = "The capital of Japan is Tokyo and it is the most populous city in Asia."

input_data = {
    "question": "What is the capital of Japan?",
    "context": test_context,
    "answer": test_answer
}

# Run directly
result = evaluate_faithfulness(input_data)
eval_report = result["faithfulness_result"]

print(f"Score: {eval_report.score}/5")
print(f"Reason: {eval_report.reason}")
