# ========== Use custom prompt instead of web_search ==============

# If you are building a specialized RAG pipeline—for example, searching a database of medical 
# research papers or legal contracts—the generic web_search instruction will generate fake text 
# that sounds like a blog post. By writing a custom prompt template, you can tell the LLM: 
# "Imagine you are a heart surgeon writing a peer-reviewed medical abstract about {query}". 
# This forces the hypothetical text to match the highly technical jargon inside your actual documents, 
# drastically improving retrieval precision.
 
import logging
import os
import warnings
from dotenv import load_dotenv

from langchain_classic.chains.hyde.base import HypotheticalDocumentEmbedder
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain.chat_models import init_chat_model

from huggingface_hub import utils
import transformers

# Suppress warnings and heavy logs
warnings.filterwarnings("ignore")
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
utils.disable_progress_bars()
transformers.utils.logging.set_verbosity_error()

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
llm = init_chat_model("groq:llama-3.1-8b-instant")

# Step 1: Load and split documents
# TODO change question to cybersecurity_data if needed
FILE_NAME = "cybersecurity_data/langchain_crewai_dataset.txt"
loader = TextLoader(FILE_NAME)
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# Step 2: Set up base embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
base_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# Step 3: Define Custom Target Prompt Template
custom_prompt = PromptTemplate.from_template(
    "Generate a concise hypothetical answer for this topic: {query}"
)

# Step 4: Build Custom HyDE Embedder wrapping your engineered prompt
custom_hyde_embedding_function = HypotheticalDocumentEmbedder.from_llm(
    llm=llm,
    base_embeddings=base_embeddings,
    custom_prompt=custom_prompt
)

# Step 5: Store chunks into a separate database on disk 
print("🎯 Ingesting documents into Custom Prompt database...")
custom_vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=custom_hyde_embedding_function,
    persist_directory="output/langchain_custom"
)

# --- ADDED: Step 5b: Construct the RAG Answer Generation Chain ---
# Identical to the prompt abstraction in your web_search variant
rag_prompt = PromptTemplate.from_template("""
Use the context below to answer the question.

Context:
{context}

Question: {input}
""")
rag_chain = create_stuff_documents_chain(llm=llm, prompt=rag_prompt)


# --- ADDED: Step 6: Define Full Pipeline Search and Synthesis Loop ---
def custom_hyde_rag_pipeline(query):
    # Pass query; Custom HyDE maps query -> custom hypothetical text -> vector match
    matched_docs = custom_vectorstore.similarity_search(query, k=4)
    
    print("\n🎯 [Custom Prompt Mode] Retrieved Context Chunks:")
    print("-" * 50)
    for i, doc in enumerate(matched_docs):
        print(f"--- Chunk {i+1} ---\n{doc.page_content}\n")
        
    response = rag_chain.invoke({
        "input": query,
        "context": matched_docs
    })
    return response


# --- ADDED/UPDATED: Step 7: Execution and Presentation ---
if __name__ == "__main__":
    target_query = "What memory modules does LangChain provide?"
    answer = custom_hyde_rag_pipeline(target_query)
    
    print("=" * 60)
    print("✅ Final Answer (Custom Prompt):")
    print("=" * 60)
    print(answer)