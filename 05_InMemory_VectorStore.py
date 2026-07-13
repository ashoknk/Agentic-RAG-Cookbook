"""
================================================================================
This script serves as an introductory laboratory for understanding the base 
mechanics of LangChain vector stores using an ephemeral, in-memory database 
(`InMemoryVectorStore`). It is designed to isolate and teach vector storage 
and retrieval dynamics without the added complexity of setting up external cloud infrastructure.

- `InMemoryVectorStore`: Stores high-dimensional embedding vectors inside a simple 
  Python dictionary or NumPy array residing fully in volatile RAM. Data is 
  completely wiped upon program shutdown.
- Native Engines (FAISS, Chroma): Built with optimized indexing 
  architectures, supporting native disk persistence and massive scalability.

1. INITIALIZATION: Setup OpenAI embedding configurations using `text-embedding` 
   compressed to a lightweight 512 dimensions for rapid runtime calculation.
2. RAM INGESTION: Creates a diverse mockup dataset of raw text `Document` objects 
   (tweets, news articles, web text) and writes them to the RAM array.
3. CORE RETRIEVAL PATTERNS:
   - **Method A (`similarity_search`)**: Directly executes mathematical vector 
     comparisons on the store object, returning standard Python document lists.
   - **Method B (`as_retriever`)**: Wraps the raw store inside the standardized LangChain 
     Retriever interface, outputting an execution-ready `Runnable` component.
================================================================================
"""

import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain.chat_models import init_chat_model


# ==============================================================================
# 1. INITIAL PREPARATION: Models & Store Setup
# ==============================================================================

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize LLM and Embeddings
llm = init_chat_model("openai:gpt-4o-mini")
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=512
)

# Initialize InMemoryVectorStore
# This is a lightweight store using a Python dictionary and NumPy
vector_store = InMemoryVectorStore(embedding=embeddings)

# ==============================================================================
# 2. DATA PREPARATION: Documents
# ==============================================================================

documents = [
    Document(page_content="Just spilled coffee all over my keyboard right before a major demo. Peak Monday vibes.", metadata={"source": "tweet"}),
    Document(page_content="Can anyone recommend a good lightweight code editor for Mac? Thinking of switching.", metadata={"source": "tweet"}),
    Document(page_content="Finally hit 10k followers today! Huge thank you to everyone who supports my tech journey.", metadata={"source": "tweet"}),
    Document(page_content="Spent the weekend refactoring my codebase and honestly, it's so satisfying.", metadata={"source": "tweet"}),
    Document(page_content="Local authorities have issued a flash flood warning for the tri-state area until midnight.", metadata={"source": "news"}),
    Document(page_content="The global tech summit kicks off tomorrow with major announcements expected in AI regulation.", metadata={"source": "news"}),
    Document(page_content="Scientists discover a new deep-sea coral species off the coast of Australia.", metadata={"source": "news"}),
    Document(page_content="The city transit authority announced a 10% fare increase starting next month.", metadata={"source": "news"}),
    Document(page_content="A beginner's guide to mastering CSS Grid and Flexbox layouts in 2026.", metadata={"source": "website"}),
    Document(page_content="Top 5 meal prep strategies for busy working professionals looking to eat healthier.", metadata={"source": "website"}),
    Document(page_content="Click here for our comprehensive, step-by-step tutorial on deploying your first Next.js app.", metadata={"source": "website"}),
    Document(page_content="Discover the best budget-friendly hiking gear for your next outdoor adventure.", metadata={"source": "website"}),
]

# Add documents to the store
vector_store.add_documents(documents=documents)

# ==============================================================================
# 3. HELPER FUNCTION: Formatted Printing
# ==============================================================================

def print_formatted_results(question: str, results: list[Document]):
    """Prints Document details on separate lines with tab indentation"""
    print(f"### Question: {question}")
    for i, doc in enumerate(results):
        # Using \t to indent the contents neatly under the main question
        print(f"\t--- Result {i+1} ---")
        print(f"\tDocument->id='{getattr(doc, 'id', 'N/A')}',")
        print(f"\tmetadata->'source': '{doc.metadata.get('source')}',")
        print(f"\tpage_content='{doc.page_content}')")
        print()

# ==============================================================================
# 4. EXECUTION: Similarity Search & Retrieval
# ==============================================================================

# 1. vector_store.similarity_search() This is a direct method call on the vector store object itself.
# What it returns: A simple Python list of Document objects.
# How it works: It takes a string, converts it to an embedding using your defined embedding model, and immediately calculates the distance to other vectors in the store.

# Matches Doc 1 (coffee on keyboard) and Doc 2 (code editor recommendation)
question1 = "What are the latest tech-related updates and frustrations posted on social media?"

# Matches Doc 5 (flash flood warning) and Doc 8 (transit fare increase)
question2 = "What local city alerts and public transportation updates were announced?"

# Matches Doc 9 (CSS Grid/Flexbox guide) and Doc 11 (Next.js deployment tutorial)
question3 = "Where can I find step-by-step web development guides and coding tutorials?"

# Matches Doc 4 (refactoring codebase) and Doc 6 (tech summit/AI regulation news)
question4 = "What major software engineering tasks or global tech topics are currently being discussed?"

print("\n### =======Using similarity_search (k=2)========")

res1 = vector_store.similarity_search(question1, k=2)
print_formatted_results(question1, res1)

print("-" * 50)

res2 = vector_store.similarity_search(question2, k=2)
print_formatted_results(question2, res2)
print("-" * 50)


res3 = vector_store.similarity_search(question3, k=2)
print_formatted_results(question3, res3)
print("-" * 50)

res4 = vector_store.similarity_search(question4, k=2)
print_formatted_results(question4, res4)
print("-" * 50)

# 2. vector_store.as_retriever() and .invoke()
# This wraps the vector store in a Retriever interface, which is one of the core "Base Components" of LangChain.
# What it returns: Technically it returns a list of documents, but it is wrapped in a Runnable object.


print("### =======Using as_retriever (k=2)========")
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

retriever_results1 = retriever.invoke(question1)
print_formatted_results(question1, retriever_results1)

print("-" * 50)

retriever_results2 = retriever.invoke(question2)
print_formatted_results(question2, retriever_results2)