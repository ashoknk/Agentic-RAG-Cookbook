"""
Instead of using raw functions like we did in the previous step, or a rigid text chain, the modern LangChain 
way to build this is to wrap your retrieval and prompt creation into clean Runnables.

By using RunnableLambda, you get all the benefits of LangChain's chaining system 
(streaming, logging, tracing, and clean structure), but you retain complete control over 
building your custom base64 image strings and message structures before they land inside your Vision LLM!
"""
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

# Import standard LCEL runnables and output parsers
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

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

class CLIPLangChainEmbeddings(Embeddings):
    """Bridge wrapper to allow FAISS database queries via CLIP embeddings"""
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [embed_text(t).tolist() for t in texts]
               
    def embed_query(self, text: str) -> list[float]:
        return embed_text(text).tolist()

# ==============================================================================
# 2. SOURCE INGESTION & VECTOR STORE INDEXING
# ==============================================================================
IMAGE_PATH = "01_top_sources.png"  # Your source chart image
image_data_store = {}              # In-memory dictionary to hold base64 data for the LLM

# 2a. Open and process the target image
pil_image = Image.open(IMAGE_PATH).convert("RGB")
image_id = "data/top_ip_sources_chart"

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
# 3. RUNNABLE COMPONENT & LCEL CHAIN PIPELINE
# ==============================================================================
# Initialize the Vision LLM
llm = init_chat_model("openai:gpt-4o-mini")

def format_multimodal_prompt(input_dict):
    """
    Accepts an input dictionary with a 'question' key. 
    Retrieves matching visual context and constructs a list containing a HumanMessage.
    """
    query = input_dict["question"]
    
    # Step A: Query vector database using the custom CLIP embedding logic
    query_embedding = embed_text(query)
    retrieved_docs = vector_store.similarity_search_by_vector(embedding=query_embedding, k=1)
    
    # Step B: Construct standard text query block
    message_content = [{
        "type": "text", 
        "text": f"Question: {query}\n\nPlease analyze the provided chart image and answer the question accurately."
    }]
    
    # Step C: Inject any corresponding images as structured base64 blocks 
    for doc in retrieved_docs:
        img_id = doc.metadata.get("image_id")
        if img_id in image_data_store:
            message_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_data_store[img_id]}"}
            })
            
    # Return the message structured within a format the ChatModel accepts
    return [HumanMessage(content=message_content)]

# --- Modern LCEL Multimodal Chain Structure ---
# RunnableLambda wraps our custom retrieval/formatting function cleanly into the pipeline.
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
        
        # Format the query as a dictionary to flow correctly into our chain
        query_dict = {"question": query}
        answer = multimodal_rag_chain.invoke(query_dict)
        
        print(f"Answer:\n{answer}")
        print("=" * 70)