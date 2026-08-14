# Prompt to Create a React + FastAPI App for the Agentic RAG System

Use the prompt below with an AI assistant to generate a modern, minimal React frontend and FastAPI backend that replaces the existing Streamlit interface.

***

```markdown
I have a Python-based Agentic RAG system that uses Streamlit for its user interface (`streamlit_app.py`). I want to replace the Streamlit UI with a modern, simple, and minimal React frontend. 

Because the RAG pipeline (document ingestion, vector store, and graph building) is written in Python, I need:
1. A lightweight **FastAPI backend** that exposes the RAG system via API endpoints.
2. A clean, single-page **React frontend** (using Vite, Tailwind CSS or clean CSS) that communicates with this backend.

Please generate the code for both components based on the details below.

---

### Part 1: Python FastAPI Backend (`app.py`)
Create a FastAPI server (`app.py`) that wraps my existing RAG components (`src.config.config`, `src.document_ingestion.document_processor`, `src.vectorstore.vectorstore`, `src.graph_builder.graph_builder`).

Requirements:
- **CORS Middleware:** Enable CORS so the React app running on `localhost:5173` (or similar) can communicate with it.
- **Initialization Endpoint (`POST /api/init`):** 
  - Initializes the RAG system once (cached in server memory).
  - Returns the status and the number of loaded document chunks.
- **Search Endpoint (`POST /api/search`):** 
  - Takes a JSON payload with `question` (string).
  - Runs the query using `rag_system.run(question)`.
  - Measures execution/elapsed time.
  - Returns a JSON response containing:
    - `answer`: The text response.
    - `time`: Time taken in seconds.
    - `retrieved_docs`: A list of objects with `page_content` (first 300 characters) and metadata.

---

### Part 2: React Frontend
Create a modern, minimal, responsive single-page React app (using TypeScript or JavaScript).

Requirements:
- **Layout & Style:** Minimalist, centered layout with a clean search interface (modern input field and search button).
- **Initialization Screen:** Show a loading spinner / message while the system is initializing (`POST /api/init` is running). Once initialized, show a success toast or banner.
- **Search Interface:**
  - An input field for the question with a submit button.
  - While fetching the answer, show an elegant skeleton loader or spinner.
- **Results Display:**
  - Show the generated answer in a prominent card with a light green success background.
  - Show a collapsible section (Accordion/Expander) called "📄 Source Documents" showing cards for each retrieved document's snippet.
  - Show the elapsed time at the bottom.
- **History Section:**
  - Display a section with the last 3 search queries, showing the question, truncated answer, and response time. Clicking a past question should re-populate the input.

Please provide:
1. The complete `app.py` for FastAPI.
2. The complete React component (`App.jsx` or `App.tsx`) and any setup instructions (including Vite dependencies).
```
