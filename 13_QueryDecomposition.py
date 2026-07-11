"""
================================================================================
This script introduces **Query Decomposition**, a multi-hop reasoning strategy 
designed to conquer complex, multi-part questions. 

### 🧠 What is Query Decomposition?
Query decomposition is the process of taking a complex, multi-part question and 
breaking it into simpler, atomic sub-questions that can each be retrieved and answered individually.

#### ✅ Why Use Query Decomposition?
- Complex queries often involve multiple concepts
- LLMs or retrievers may miss parts of the original question
- It enables multi-hop reasoning (answering in steps)
- Allows parallelism (especially in multi-agent frameworks)


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
FILE_NAME = "cybersecurity_data/cybersecurity_dataset.txt"
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
# query = "How does IAM use for enterprise security?"
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


def execute_query_decomposition_demo(user_query):
    """
    A single unified function that runs the entire Query Decomposition pipeline
    and handles structured console printing for clear live demonstrations.
    """
    print("\n" + "=" * 75)
    print(f"🚀 PHASE 1: RECEIVING COMPLEX MULTI-HOP QUERY")
    print("=" * 75)
    print(f"👉 Original User Query: '{user_query}'")
    
    # 1. Generate the sub-questions from the main query
    sub_qs_text = decomposition_chain.invoke({"question": user_query})
    # print(f"\n📝 Decomposed Sub-Questions:\n{sub_qs_text}\n") #NOTE: Just for testing 
    
    # Splits the raw LLM string response into individual lines by newline characters.
    # Filters out empty lines and strips common list markers (numbers, dots, bullets, dashes).
    # Returns a clean Python list containing only the text of the atomic sub-questions.
    sub_questions = [q.strip("-•1234567890. ").strip() for q in sub_qs_text.split("\n") if q.strip()]
    
    print("\n✨ Step 1: Decomposition Engine Triggered")
    print("🤖 The LLM broke your complex question down into these atomic sub-tasks:")
    for idx, q in enumerate(sub_questions, 1):
        print(f"   └── [Sub-Question {idx}]: {q}")
    print("-" * 75)
    
    print("\n" + "=" * 75)
    print(f"⚙️ PHASE 2: EXECUTING PARALLEL RETRIEVAL & ISOLATED RAG")
    print("=" * 75)
    
    # 2. Loop through individual sub-questions sequentially to collect answers
    #NOTE - comment below print statement if you see too much output in the console
    sub_task_results = []
    for i, subq in enumerate(sub_questions, 1):
        print(f"\n🔄 Running Sub-Task {i}/{len(sub_questions)}:")
        print(f"🔍 Question: \"{subq}\"")
        
        # Retrieve diverse chunks using MMR (k=4, lambda=0.7)
        retrieved_docs = retriever.invoke(subq)
        
        # Print exactly where the data is being sourced from to show MMR working
        print(f"📚 MMR Retrieved {len(retrieved_docs)} relevant context blocks:")
        for doc in retrieved_docs:
            snippet = doc.page_content.replace("\n", " ")[:75]
            print(f"   📄 Match: {snippet}...")
            
        # Answer the atomic sub-question completely
        answer = qa_chain.invoke({"input": subq, "context": retrieved_docs})
        print(f"💡 Isolated Sub-Answer Generated ✔️")
        
        # Save results for the final summary phase
        sub_task_results.append({
            "num": i,
            "question": subq,
            "answer": answer.strip()
        })
        print("-" * 50)

    # 3. Print the Comprehensive Report Framework (Phase 3)
    print("\n" + "=" * 75)
    print("🏆 PHASE 3: FINAL SYNTHESIZED REPORT FRAMEWORK")
    print("=" * 75)
    print(f"🎯 ORIGINAL QUERY:\n   \"{user_query}\"\n")
    print("🧩 DECOMPOSED RECONSTRUCTION:")
    print("-" * 75)
    
    for item in sub_task_results:
         print(f"📌 [SUB-TASK {item['num']}]: {item['question']}")
         print(f"📝 ANSWER Framework:")
         # Indent the answer block cleanly so it doesn't look like a generic wall of text
         indented_answer = "\n         ".join(item['answer'].split("\n"))
         print(f"         {indented_answer}")
         print("-" * 75)
         
    print("\n" + "═" * 75 + "\n\n")


if __name__ == "__main__":
   #  complex_query_1 = "How do human cloud misconfigurations differ from vulnerabilities found in software dependencies, and how do automated tools address each?"
   #  execute_query_decomposition_demo(complex_query_1)

   # NOTE : Test this by uncommenting the below lines one at a time to see how the Query Decomposition pipeline handles each complex query.
   #  complex_query_2 = "What are the primary indicators of a compromised superuser account compared to an infected corporate laptop, and what specialized tools track these threats?"
   #  execute_query_decomposition_demo(complex_query_2)
    
    complex_query_5 = "What is the relationship between API security vulnerabilities and automated pipeline remediation tools?"    
    execute_query_decomposition_demo(complex_query_5)