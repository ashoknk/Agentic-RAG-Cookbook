### Multimodal RAG (PDF With Images)
"""
================================================================================
This script introduces **Multimodal Retrieval-Augmented Generation (RAG)**. 
Instead of extracting only text from documents, this architecture processes and 
indexes both unstructured text blocks and physical image files (such as charts, 
diagrams, or photographs) found within complex source PDFs.


THE COGNITIVE ENGINE (CLIP):
The system utilizes OpenAI's `clip-vit-base-patch32` multimodal model. 
CLIP maps both language tokens and visual imagery into a **shared, unified vector space**. 
This means a raw string query and a visual chart representing the same concept will 
yield highly matching mathematical coordinate embeddings, allowing for simultaneous retrieval.

1. DUAL INGESTION & PIPELINE MAPPING: Iterates through PDF pages. Text blocks are 
   chunked and vectorized via `embed_text()`, while extracted images are converted 
   to PIL format, embedded via `embed_image()`, and backed up as Base64 strings.
2. UNIFIED EXTENSION INDEX: Compiles text and visual placeholders into a unified 
   LangChain `FAISS` database managed by a custom `CLIPLangChainEmbeddings` wrapper.
3. RETRIEVAL & RESPONSE GENERATION: Executes cross-modal searches. The script 
   segregates text and image matches, structuralizing them into an advanced, 
   multimodal `HumanMessage` payload dispatched to an OpenAI vision-capable model.
================================================================================
"""

import fitz  # PyMuPDF
from langchain_core.documents import Document
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy as np
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
import os
import warnings
import logging
from dotenv import load_dotenv
import base64
import io
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Suppress standard python warnings
warnings.filterwarnings("ignore")
# Suppress Transformer loading logs and general warnings
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
# Suppress the progress bar (The 100%|███ bar)
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

from huggingface_hub import utils
utils.disable_progress_bars()

load_dotenv()

## set up the environment
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

###Clip Model
### initialize the Clip Model for unified embeddings
CLIP_MODEL="openai/clip-vit-base-patch32"
CLIP_PROCESSOR="openai/clip-vit-base-patch32"

clip_model=CLIPModel.from_pretrained(CLIP_MODEL)
clip_processor=CLIPProcessor.from_pretrained(CLIP_PROCESSOR)
clip_model.eval()

### Embedding functions
def embed_image(image_data):
    """Embed image using CLIP"""
    if isinstance(image_data, str):  # If path
        image = Image.open(image_data).convert("RGB")
    else:  # If PIL Image
        image = image_data
    
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        # 1. Get the output object
        outputs = clip_model.get_image_features(**inputs)
        
        # 2. Extract the actual tensor features safely
        features = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs
        
        # 3. Normalize the tensor safely
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().numpy()

# def embed_image(image_data):
#     """Embed image using CLIP"""
#     if isinstance(image_data, str):  # If path
#         image = Image.open(image_data).convert("RGB")
#     else:  # If PIL Image
#         image = image_data
    
#     inputs=clip_processor(images=image,return_tensors="pt")
#     with torch.no_grad():
#         features = clip_model.get_image_features(**inputs)
#         # Normalize embeddings to unit vector
#         features = features / features.norm(dim=-1, keepdim=True)
#         return features.squeeze().numpy()
    
# def embed_text(text):
#     """Embed text using CLIP."""
#     inputs = clip_processor(
#         text=text, 
#         return_tensors="pt", 
#         padding=True,
#         truncation=True,
#         max_length=77  # CLIP's max token length
#     )
#     with torch.no_grad():
#         features = clip_model.get_text_features(**inputs)
#         # Normalize embeddings
#         features = features / features.norm(dim=-1, keepdim=True)
#         return features.squeeze().numpy()
    
def embed_text(text):
    """Embed text using CLIP."""
    inputs = clip_processor(
        text=text, 
        return_tensors="pt", 
        padding=True,
        truncation=True,
        max_length=77  # CLIP's max token length
    )
    with torch.no_grad():
        # 1. Get the output object
        outputs = clip_model.get_text_features(**inputs)
        
        # 2. Extract the actual tensor features
        # If 'outputs' is a tensor already, use it; otherwise, get the pooler_output
        features = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs
        
        # 3. Now you can normalize the tensor
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().numpy()
    

## Process PDF
#TODO replace with the actual PDF file path
# mercedes-benz-2025.pdf
# tesla-2025.pdf
PDF_FILE="pdf_data/multimodal_sample.pdf"
doc=fitz.open(PDF_FILE)
# Storage for all documents and embeddings
all_docs = []
all_embeddings = []
image_data_store = {}  # Store actual image data for LLM

# Text splitter
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

for i,page in enumerate(doc):
    ## process text
    text=page.get_text()
    if text.strip():
        ##create temporary document for splitting
        temp_doc = Document(page_content=text, metadata={"page": i, "type": "text"})
        text_chunks = splitter.split_documents([temp_doc])

        #Embed each chunk using CLIP
        for chunk in text_chunks:
            embedding = embed_text(chunk.page_content)
            all_embeddings.append(embedding)
            all_docs.append(chunk)

    ## Process images. Three Important Actions:
    ##Convert PDF image to PIL format
    ##Store as base64 for GPT-4V (which needs base64 images)
    ##Create CLIP embedding for retrieval

    for img_index, img in enumerate(page.get_images(full=True)):
        try:
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            
            # Convert to PIL Image
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            # Create unique identifier
            image_id = f"page_{i}_img_{img_index}"
            
            # Store image as base64 for later use with GPT-4V
            buffered = io.BytesIO()
            pil_image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            image_data_store[image_id] = img_base64
            
            # Embed image using CLIP
            embedding = embed_image(pil_image)
            all_embeddings.append(embedding)
            
            # Create document for image
            image_doc = Document(
                page_content=f"[Image: {image_id}]",
                metadata={"page": i, "type": "image", "image_id": image_id}
            )
            all_docs.append(image_doc)
            
        except Exception as e:
            print(f"Error processing image {img_index} on page {i}: {e}")
            continue

doc.close()

# Create unified FAISS vector store with CLIP embeddings
embeddings_array = np.array(all_embeddings)

#NOTE: to remove warning `embedding_function` is expected to be an Embeddings object, support for passing in a function will soon be removed.
from langchain_core.embeddings import Embeddings

class CLIPLangChainEmbeddings(Embeddings):
    """Custom wrapper class to present our CLIP text embedder to LangChain"""
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [embed_text(t).tolist() for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return embed_text(text).tolist()


# Create custom FAISS index since we have precomputed embeddings
# vector_store = FAISS.from_embeddings(
#     text_embeddings=[(doc.page_content, emb) for doc, emb in zip(all_docs, embeddings_array)],
#     embedding=None,  # We're using precomputed embeddings
#     metadatas=[doc.metadata for doc in all_docs]
# )

# Create unified FAISS vector store with our official class wrapper
vector_store = FAISS.from_embeddings(
    text_embeddings=[(doc.page_content, emb) for doc, emb in zip(all_docs, embeddings_array)],
    embedding=CLIPLangChainEmbeddings(),  # Wrapped cleanly!
    metadatas=[doc.metadata for doc in all_docs]
)

# Initialize GPT-4 Vision model
CHAT_MODEL="openai:gpt-4.1"
llm = init_chat_model(CHAT_MODEL)

def retrieve_multimodal(query, k=5):
    """Unified retrieval using CLIP embeddings for both text and images."""
    # Embed query using CLIP
    query_embedding = embed_text(query)
    
    # Search in unified vector store
    results = vector_store.similarity_search_by_vector(
        embedding=query_embedding,
        k=k
    )
    
    return results

def create_multimodal_message(query, retrieved_docs):
    """Create a message with both text and images for GPT-4V."""
    content = []
    
    # Add the query
    content.append({
        "type": "text",
        "text": f"Question: {query}\n\nContext:\n"
    })
    
    # Separate text and image documents
    text_docs = [doc for doc in retrieved_docs if doc.metadata.get("type") == "text"]
    image_docs = [doc for doc in retrieved_docs if doc.metadata.get("type") == "image"]
    
    # Add text context
    if text_docs:
        text_context = "\n\n".join([
            f"[Page {doc.metadata['page']}]: {doc.page_content}"
            for doc in text_docs
        ])
        content.append({
            "type": "text",
            "text": f"Text excerpts:\n{text_context}\n"
        })
    
    # Add images
    for doc in image_docs:
        image_id = doc.metadata.get("image_id")
        if image_id and image_id in image_data_store:
            content.append({
                "type": "text",
                "text": f"\n[Image from page {doc.metadata['page']}]:\n"
            })
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_data_store[image_id]}"
                }
            })
    
    # Add instruction
    content.append({
        "type": "text",
        "text": "\n\nPlease answer the question based on the provided text and images."
    })
    
    return HumanMessage(content=content)

def multimodal_pdf_rag_pipeline(query):
    """Main pipeline for multimodal RAG."""
    # Retrieve relevant documents
    context_docs = retrieve_multimodal(query, k=5)
    
    # Create multimodal message
    message = create_multimodal_message(query, context_docs)
    
    # Get response from GPT-4V
    response = llm.invoke([message])
    
    # Print retrieved context info
    print(f"\nRetrieved {len(context_docs)} documents:")
    for doc in context_docs:
        doc_type = doc.metadata.get("type", "unknown")
        page = doc.metadata.get("page", "?")
        if doc_type == "text":
            preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
            print(f"  - Text from page {page}: {preview}")
        else:
            print(f"  - Image from page {page}")
    print("\n")
    
    return response.content

if __name__ == "__main__":
    # Example queries
    queries = [
        "What does the chart on page 1 show about revenue trends?",
        "Summarize the main findings from the document",
        "What visual elements are present in the document?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 50)
        answer = multimodal_pdf_rag_pipeline(query)
        print(f"Answer: {answer}")
        print("=" * 70)