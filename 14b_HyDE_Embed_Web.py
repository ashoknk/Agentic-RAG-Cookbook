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
os.environ["HF_TOKEN"]=os.getenv("HF_TOKEN")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
llm = init_chat_model("groq:llama-3.1-8b-instant")

# TODO change question to cybersecurity_data if needed
FILE_NAME = "cybersecurity_data/cybersecurity_dataset.txt"
# Step 1: Load and split documents
loader = TextLoader(FILE_NAME)
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# Step 2: Set up base embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
base_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# Step 3: Build HyDE Embedder using standard 'web_search' preset
# This uses a pre-defined prompt that mimics a web search result.
# Merge your standard embedding engine (HuggingFaceEmbeddings) and 
# your LLM into a singular object called hyde_embedding_function

OUTPUT_DIR = "output/langchain_web"
hyde_embedding_function = HypotheticalDocumentEmbedder.from_llm(
    llm=llm,
    base_embeddings=base_embeddings,
    prompt_key="web_search"
)

# Step 4: Store chunks into Chroma DB using HyDE embeddings
print("📦 Ingesting documents into standard Web Search database...")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=hyde_embedding_function,
    persist_directory=OUTPUT_DIR
)

# Step 5: Construct the RAG Answer Generation Chain
rag_prompt = PromptTemplate.from_template("""
Use the context below to answer the question.

Context:
{context}

Question: {input}
""")
rag_chain = create_stuff_documents_chain(llm=llm, prompt=rag_prompt)

# Step 6: Define full pipeline search and synthesis loop
# Under the hood, the HypotheticalDocumentEmbedder intercepts your question, silently ships it 
# out to the LLM to generate the fake text using standard prompt structures (like web_search), 
# immediately converts that hidden result into numbers, and queries the database.

def hyde_rag_pipeline(query):
    # Pass query; HyDE silently maps query -> hypothetical text -> vector match
    matched_docs = vectorstore.similarity_search(query, k=4)
    
    print("\n📚 [Web Search Mode] Retrieved Context Chunks:")
    for i, doc in enumerate(matched_docs):
        print(f"--- Chunk {i+1} ---\n{doc.page_content}\n")
        
    response = rag_chain.invoke({
        "input": query,
        "context": matched_docs
    })
    return response

# Step 7: Run execution
query = "What memory modules does LangChain provide?"
answer = hyde_rag_pipeline(query)
print("✅ Final Answer (Web Search):\n", answer)