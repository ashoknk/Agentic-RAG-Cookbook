"""Document processing module for loading and splitting documents"""


from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

from typing import List, Union
from pathlib import Path


class DocumentProcessor:
    """Handles document loading and processing"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize document processor
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
    def load_from_url(self, url: str) -> List[Document]:
        """Load document(s) from a URL"""
        loader = WebBaseLoader(url)
        return loader.load()

    
    # Union helps function to receive a path as either text or as a Path object.
    def load_from_pdf_dir(self, directory: Union[str, Path]) -> List[Document]:
        """Load documents from all PDFs inside a directory using PyPDFDirectoryLoader.

        This project assumes PDFs live in a single directory (e.g., `data/`).
        """
        d = Path(directory)
        if not d.exists() or not d.is_dir():
            raise FileNotFoundError(f"PDF directory does not exist: {d}")
        loader = PyPDFDirectoryLoader(str(d))
        return loader.load()
        
    def load_documents(self, sources: List[str]) -> List[Document]:
        """
        Load documents from URLs, PDF directories

        Args:
            sources: List of URLs, PDF folder paths

        Returns:
            List of loaded documents
        """
        docs: List[Document] = []
        for src in sources:
            # URLs
            if isinstance(src, str) and (src.startswith("http://") or src.startswith("https://")):
                docs.extend(self.load_from_url(src))
                # print(f" src {src}")
                continue

            # treat src as a local path (file or directory)
            # path = Path(Config.PDF_FOLDER)
            path = Path(src)
            # print(f" path {path}")
            if not path.exists():
                raise FileNotFoundError(f"Source path does not exist: {path}")

            if path.is_dir():
                docs.extend(self.load_from_pdf_dir(path))
                print(f"Loaded {len(docs)} documents from directory: {path}")
            elif path.suffix.lower() == ".pdf":
                raise ValueError(
                    "Single .pdf files are not supported in this loader. "
                    "Provide a directory containing PDFs (e.g., 'data/')."
                )
            else:
                raise ValueError(
                    f"Unsupported source type: {src}. Use URL or PDF directory."
                )
        return docs
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of split documents
        """
        return self.splitter.split_documents(documents)
    
    def process_sources(self, sources: List[str]) -> List[Document]:
        """
        Load and split documents from one or more source inputs.

        This method accepts both web URLs and local PDF directories, then
        normalizes them into a flat list of chunked document objects ready for
        vector indexing.

        Args:
            sources: List of source entries to process. Each item may be either
                a URL or a local path to a PDF directory.

        Returns:
            List of processed document chunks.
        """
        # print(f"Document procesor - Loading sources: {sources}")  #NOTE Testing print statement to verify sources loaded from file and PDF directory      
        docs = self.load_documents(sources)
        return self.split_documents(docs)
