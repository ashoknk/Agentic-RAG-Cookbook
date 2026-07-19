import os
import base64
import io
import warnings
from dotenv import load_dotenv
from PIL import Image
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage
from langchain.chat_models import init_chat_model

# Clean up console logs
warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
if os.getenv("HF_TOKEN"):
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

# ==============================================================================
# 1. INITIALIZE CLIP MODEL
# ==============================================================================
CLIP_MODEL = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(CLIP_MODEL)
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
clip_model.eval()

def embed_image(pil_image):
    """Generate normalized vector embedding for an image using CLIP"""
    inputs = clip_processor(images=pil_image, return_tensors="pt")
    with torch.no_grad():
        outputs = clip_model.get_image_features(**inputs)
        features = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().numpy()

def embed_text(text):
    """Generate normalized vector embedding for a text string using CLIP"""
    inputs = clip_processor(text=text, return_tensors="pt", padding=True, truncation=True, max_length=77)
    with torch.no_grad():
        outputs = clip_model.get_text_features(**inputs)
        features = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().numpy()

"""
CLIPLangChainEmbeddings class is to act as a translator or bridge between your 
custom CLIP functions and the LangChain ecosystem
LangChain wrapper class to handle query conversions during retrieval
LangChain's vector databases (like FAISS) are built to expect a very specific "shape" or structure for embeddings.
"""
class CLIPLangChainEmbeddings(Embeddings):
    
    # When the user types a search question, use our custom embed_text() function to turn it into a CLIP vector
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [embed_text(t).tolist() for t in texts]
    # If you ever need to embed a list of text strings into vectors, run them through our embed_text() loop        
    def embed_query(self, text: str) -> list[float]:
        return embed_text(text).tolist()

# ==============================================================================
# 2. SOURCE INGESTION & VECTOR STORE INDEXING
# ==============================================================================
IMAGE_PATH = "01_top_sources.png"  # Your source chart image
image_data_store = {}              # In-memory dictionary to hold base64 data for the LLM

# 2a. Open and process the target image
pil_image = Image.open(IMAGE_PATH).convert("RGB")
image_id = "top_ip_sources_chart"

# 2b. Convert image to Base64 (Required format for sending images to OpenAI Vision models)
buffered = io.BytesIO()
pil_image.save(buffered, format="PNG")
image_data_store[image_id] = base64.b64encode(buffered.getvalue()).decode()

# 2c. Embed image with CLIP and wrap inside a LangChain Document structure
img_embedding = embed_image(pil_image)
image_document = Document(
    page_content=f"[Image content covering network request metrics: {image_id}]",
    metadata={"type": "image", "image_id": image_id}
)

# 2d. Ingest into the vector database
vector_store = FAISS.from_embeddings(
    text_embeddings=[(image_document.page_content, img_embedding)],
    embedding=CLIPLangChainEmbeddings(),
    metadatas=[image_document.metadata]
)

# ==============================================================================
# 3. LLM INITIALIZATION & MULTIMODAL PIPELINE
# ==============================================================================
llm = init_chat_model("openai:gpt-4o-mini")

def multimodal_rag_pipeline(query):
    """Executes the standard RAG sequence: Retrieve -> Construct Prompt -> Generate response"""
    
    # Step A: Retrieve relevant images by querying the vector database with text
    query_embedding = embed_text(query)
    retrieved_docs = vector_store.similarity_search_by_vector(embedding=query_embedding, k=1)
    
    # Step B: Construct a multimodal payload message containing both the string question and the image
    message_content = [{"type": "text", "text": f"Question: {query}\n\nPlease analyze the provided chart image and answer the question accurately."}]
    
    for doc in retrieved_docs:
        img_id = doc.metadata.get("image_id")
        if img_id in image_data_store:
            message_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_data_store[img_id]}"}
            })
            
    # Step C: Send the multimodal payload message to the vision LLM
    response = llm.invoke([HumanMessage(content=message_content)])
    return response.content

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
        answer = multimodal_rag_pipeline(query)
        print(f"Answer:\n{answer}")
        print("=" * 70)