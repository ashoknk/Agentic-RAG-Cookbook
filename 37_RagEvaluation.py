import os
from typing import TypedDict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# # RAG Evaluation (RAGAS Style)
# 
# ### Concept
# Evaluation measures the quality of a RAG system using metrics like:
# 1. Faithfulness: Is the answer derived solely from the context?
# 2. Answer Relevance: Does the answer actually address the question?
# 3. Context Precision: How relevant are the retrieved documents?

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# ### 1. Define Evaluation Schemas
class FaithfulnessScore(BaseModel):
    score: int = Field(description="Score from 1 to 5, where 5 is perfectly faithful to context")
    reason: str = Field(description="Reasoning for the score")

# ### 2. Define the State
class EvalState(TypedDict):
    question: str
    context: str
    answer: str
    faithfulness_result: FaithfulnessScore

# ### 3. Define Evaluation Node
#NOTE - use gpt-4o-mini if below does not work 
llm = ChatOpenAI(model="gpt-4o", temperature=0)

def evaluate_faithfulness(state: EvalState):
    print("---EVALUATING FAITHFULNESS---")
    structured_llm = llm.with_structured_output(FaithfulnessScore)
    
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
test_context = "The capital of France is Paris. It is known for the Eiffel Tower."
test_answer = "The capital of France is Paris and it is the most populous city in Europe."

input_data = {
    "question": "What is the capital of France?",
    "context": test_context,
    "answer": test_answer
}

# Run directly
result = evaluate_faithfulness(input_data)
eval_report = result["faithfulness_result"]

print(f"Score: {eval_report.score}/5")
print(f"Reason: {eval_report.reason}")

# # Note: 
# The score might be lower because the context did not mention 
# 'most populous city in Europe', even if it is a true fact.