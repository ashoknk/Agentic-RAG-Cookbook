import os
import base64
import io
import warnings
from dotenv import load_dotenv
from PIL import Image

# Import purely from LangChain and LangChain-OpenAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# Clean up console logs
warnings.filterwarnings("ignore")

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# ==============================================================================
# 1. INITIALIZE OPENAI MODELS (No Torch, No Local Models!)
# ==============================================================================
# Use standard OpenAI text embeddings for our Vector DB indexing
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

# Initialize the ChatOpenAI Model for both Summarization and Final Visual Analysis
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ==============================================================================
# 2. SOURCE INGESTION & TEXT-SUMMARIZATION INDEXING
# ==============================================================================
IMAGE_PATH = "01_top_sources.png"  # Your source chart image
image_data_store = {}              # In-memory dictionary to hold base64 data for the LLM
image_id = "top_ip_sources_chart"

# 2a. Convert the PNG image to Base64 using standard Python libraries
with open(IMAGE_PATH, "rb") as image_file:
    base64_string = base64.b64encode(image_file.read()).decode("utf-8")
    image_data_store[image_id] = base64_string

# 2b. Generating an Image Summary via API (Instead of local CLIP embedding)
print("1. Sending image to OpenAI for an ingestion summary...")
summarize_message = HumanMessage(
    content=[
        {"type": "text", "text": "Describe this chart in detail. List key metrics, titles, data points, IP addresses, and labels visible so it can be indexed for search queries later."},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_string}"}}
    ]
)
image_summary = llm.invoke([summarize_message]).content
print(f"Generated Ingestion Summary:\n{image_summary}\n")

# 2c. Pack the image summary into a Document and index it into FAISS
# The vector database will index the TEXT summary, but retain the metadata linking to the image!
image_document = Document(
    page_content=image_summary,
    metadata={"type": "image", "image_id": image_id}
)

vector_store = FAISS.from_documents(
    documents=[image_document],
    embedding=embeddings_model
)

# ==============================================================================
# 3. RUNNABLE COMPONENT & LCEL CHAIN PIPELINE
# ==============================================================================
def format_multimodal_prompt(input_dict):
    """
    Accepts user query, queries the text-based vector store, 
    and returns a formatted HumanMessage payload with the original image attached.
    """
    query = input_dict["question"]
    
    # Step A: Query vector database using regular text embeddings
    retrieved_docs = vector_store.similarity_search(query, k=1)
    
    # Step B: Construct standard text question block
    message_content = [{
        "type": "text", 
        "text": f"Question: {query}\n\nPlease analyze the provided chart image and answer the question accurately."
    }]
    
    # Step C: Match the retrieved text summary back to the raw image payload
    for doc in retrieved_docs:
        img_id = doc.metadata.get("image_id")
        if img_id in image_data_store:
            message_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_data_store[img_id]}"}
            })
            
    return [HumanMessage(content=message_content)]

# --- Modern LCEL Multimodal Chain Structure ---
multimodal_rag_chain = (
    RunnableLambda(format_multimodal_prompt)
    | llm
    | StrOutputParser()
)

# ==============================================================================
# 4. EXECUTION ENTRANCE
# ==============================================================================
if __name__ == "__main__":
    queries = [
        "What is the top IP address source seen in the chart and what is its approximate request count?",
        "List the top 3 source IP addresses shown on this chart.",
        "What metric is listed on the horizontal X-axis of this visualization?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 60)
        
        query_dict = {"question": query}
        answer = multimodal_rag_chain.invoke(query_dict)
        
        print(f"Answer:\n{answer}")
        print("=" * 70)