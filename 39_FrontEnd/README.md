# 🤖 Agentic Retrieval-Augmented Generation (RAG) System

An advanced, production-ready **Agentic RAG** system that uses **LangGraph** to orchestrate document retrieval and model execution. At its core, the system utilizes a **ReAct (Reasoning and Acting) Agent** equipped with dynamic tools: a local semantic retriever (backed by **FAISS** and **OpenAI Embeddings**) and an external web tool (**Wikipedia Search**) to answer complex queries intelligently.

The project features a **Command-Line Interface (CLI)** with built-in batch evaluation and interactive prompt loops, as well as a beautiful, high-performance **Streamlit Web Application** for a visual demo experience.

---

## 🏗️ Architecture & Workflow

The system is designed as a state-machine using **LangGraph**. Below is the flow of document ingestion and query execution:

```
[ PDF Directory (data/) ] ──┐
                            ├──► [ Document Processor ] ──► [ FAISS Vector Store ]
[ Web URLs (data/url.txt) ] ─┘        (Chunking)           (OpenAI Embeddings)
                                                                 │
                                                            (Retriever Tool)
                                                                 │
 [ User Query ] ──► [ LangGraph State ] ──► [ ReAct Agent ] ◄────┘
                                                   │
                                                   ├────► [ Wikipedia Tool ] (External)
                                                   ▼
                                         [ Final Answer Response ]
```

### 1. Ingestion Pipeline
* **Local Assets**: Automatically reads and processes all PDFs placed within the `data/` directory (using `PyPDFDirectoryLoader`).
* **Web Scraping**: Loads and parses remote URLs listed in `data/url.txt` or configuration defaults (using `WebBaseLoader`).
* **Embedding**: Text is split into chunks of `500` characters with a `50`-character overlap using a `RecursiveCharacterTextSplitter`, embedded using OpenAI's `text-embedding-3-small` (512 dimensions), and indexed in a local high-performance **FAISS** vector store.

### 2. LangGraph State Workflow
* **Retriever Node**: Runs semantic search across indexed data using the user query to retrieve top contexts.
* **Responder (ReAct Agent) Node**: Launches a LangGraph-prebuilt ReAct Agent. Unlike classic RAG, the agent evaluates the query and can choose to:
  1. Call the `retriever` tool to query the local FAISS index for document-specific queries.
  2. Call the `wikipedia` tool to query general knowledge when the indexed documents lack necessary information.
  3. Formulate and return the final reasoned response back to the state workflow.

---

## ⚡ Quick Start & Setup (using `uv`)

This project uses [uv](https://github.com/astral-sh/uv), an extremely fast Python package installer and resolver. Follow these steps to set up the project environment:

### Prerequisites

Ensure you have `uv` installed. If you don't have it installed, you can get it via:
* **macOS/Linux**: `curl -LsSf https://astral-sh/uv/install.sh | sh`
* **Windows (PowerShell)**: `powershell -c "irm https://astral-sh/uv/install.ps1 | iex"`
* **Homebrew**: `brew install uv`
* **Pip**: `pip install uv`

### 1. Environment Configuration

Create a `.env` file in the root directory of the project and add your OpenAI API key:

```env
OPENAI_API_KEY=your-api-key-here
```

### 2. Virtual Environment & Dependency Installation

Initialize a virtual environment and synchronize dependencies defined in `pyproject.toml` and `uv.lock` by running:

```bash
# Create virtual environment and install all packages
uv sync
```

Alternatively, if you prefer installing directly from the `requirements.txt` file, run:

```bash
# Create virtual environment
uv venv

# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies using uv pip
uv pip install -r requirements.txt
```

---

## 🚀 Running the Applications

Once your dependencies are installed and the `.env` file is configured, you are ready to run the demo.

### 💻 Command-Line Interface (CLI)

The CLI application (`main.py`) performs a batch evaluation on several pre-defined topics, outputting the generated answer for each topic, and then prompts you to enter an **Interactive Chat Mode**.

Run the CLI using `uv`:

```bash
uv run main.py
```

*Or if you have activated your virtual environment:*

```bash
python main.py
```

#### CLI Interaction
1. Upon launch, it will process the ingested source PDFs/URLs.
2. It will automatically run 8 example evaluation queries covering:
   - **Autonomous Agents & Memory Architectures**
   - **Generative Video & Diffusion Models**
   - **Agent Security & Edge Cases**
3. At the end, type `y` to enter the **Interactive Mode**.
4. Type your custom questions, or enter `q`, `x`, `quit` to exit.

---

### 🌐 Streamlit Web Dashboard

The web interface (`streamlit_app.py`) provides an elegant and responsive UI to query the Agentic RAG system, check source documents, and see recent query execution times and history.

Run the Streamlit application using `uv`:

```bash
uv run streamlit run streamlit_app.py
```

*Or if you have activated your virtual environment:*

```bash
streamlit run streamlit_app.py
```

#### Streamlit Features
* **Status Updates**: Indicates how many document chunks have been successfully indexed.
* **Interactive Search**: Type any question, hit **Search**, and see the agent's real-time response.
* **Interactive Expanders**: Expand **Source Documents** to inspect the raw content and metadata of the segments retrieved by the vector database.
* **Search History**: Displays your last 3 questions and answers, including their processing latency in seconds.

---

## 📁 Repository Structure

Below is an overview of the key folders and files in this repository:

* **`main.py`**: The CLI entry point that initializes the RAG system and runs sample/interactive prompts.
* **`streamlit_app.py`**: The Streamlit application hosting the user-friendly Web UI.
* **`pyproject.toml` / `uv.lock`**: Configuration files detailing python requirements and package pins for `uv`.
* **`requirements.txt`**: Standard dependencies list.
* **`data/`**: Source documents folder containing:
  - `.pdf` files regarding Agent Memory, OpenClaw, and Diffusion Video Models.
  - `url.txt`: Web URLs to scrape and index.
* **`src/`**: Core codebase:
  - **`config/config.py`**: Model selection (`openai:gpt-4o`), chunking configuration, and default URL settings.
  - **`document_ingestion/document_processor.py`**: Text loading, directory scanning, and splitting pipeline.
  - **`vectorstore/vectorstore.py`**: FAISS index creator and query retriever using OpenAI embeddings.
  - **`state/rag_state.py`**: LangGraph workflow state data model (`RAGState`).
  - **`graph_builder/graph_builder.py`**: Compilation and definition of the LangGraph workflow structure.
  - **`node/`**: Core functional nodes:
    - `reactnode.py`: Implements the **ReAct Agent** equipped with the `retriever` tool and the `wikipedia` tool.
    - `nodes.py`: Simple context concatenation node (non-agentic reference).

---

## 💡 Example Questions to Try

To test the multi-tool reasoning capability of the agent, try these types of questions:

1. **Document-Specific Questions** (requires the local `retriever` tool):
   * *"What are the core components of the OpenClaw execution framework?"*
   - *"What strategies make video diffusion models more computationally efficient?"*
   - *"What are the primary security vulnerabilities when deploying autonomous OpenClaw agents?"*

2. **General Knowledge Questions** (requires the `wikipedia` search tool):
   * *"Who is Alan Turing and what was his contribution to AI?"*
   - *"When was the transformer architecture first introduced and by whom?"*

3. **Hybrid Questions** (agent may use both or reason dynamically):
   * *"How does the memory architecture in autonomous agents compare to standard computer RAM?"*
