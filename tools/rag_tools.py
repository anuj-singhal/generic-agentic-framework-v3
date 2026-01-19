"""
RAG (Retrieval-Augmented Generation) Tools - In-memory document storage and semantic search.
Uses FAISS for vector indexing and OpenAI embeddings for semantic search.
"""

import os
import json
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from langchain_core.tools import tool
from dataclasses import dataclass, field

from core.tools_base import tool_registry

# Try to import optional dependencies
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# ============================================
# CONFIGURATION
# ============================================

DEFAULT_CHUNK_SIZE = 1000  # characters
DEFAULT_CHUNK_OVERLAP = 200  # characters
DEFAULT_TOP_K = 5
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536


# ============================================
# MODULE-LEVEL STORAGE
# ============================================

@dataclass
class DocumentChunk:
    """Represents a chunk of a document."""
    text: str
    document_name: str
    chunk_index: int
    start_char: int
    end_char: int
    metadata: Dict[str, Any] = field(default_factory=dict)


_rag_store = {
    "chunks": [],           # List of DocumentChunk objects
    "embeddings": None,     # numpy array of embeddings
    "faiss_index": None,    # FAISS IndexFlatL2
    "documents": {},        # Dict[document_name] -> {chunks, tokens, upload_time, file_path}
    "metadata": {
        "total_chunks": 0,
        "total_tokens": 0,
        "total_documents": 0,
        "last_updated": None
    }
}


# ============================================
# HELPER FUNCTIONS
# ============================================

def _get_openai_client() -> Optional[Any]:
    """Get OpenAI client from environment."""
    if not OPENAI_AVAILABLE:
        return None
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def _count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Count tokens in text using tiktoken."""
    if not TIKTOKEN_AVAILABLE:
        # Rough estimate: 1 token ~= 4 characters
        return len(text) // 4
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception:
        return len(text) // 4


def _load_document(file_path: str) -> Tuple[str, str]:
    """
    Load document content from file path.
    Returns (content, document_type)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    file_ext = os.path.splitext(file_path)[1].lower()
    document_name = os.path.basename(file_path)

    if file_ext == ".txt" or file_ext == ".md":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read(), "text"

    elif file_ext == ".pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            text_parts = []
            for page in reader.pages:
                text_parts.append(page.extract_text() or "")
            return "\n\n".join(text_parts), "pdf"
        except ImportError:
            raise ImportError("pypdf is required to read PDF files. Install with: pip install pypdf")

    elif file_ext == ".docx":
        try:
            from docx import Document
            doc = Document(file_path)
            text_parts = []
            for para in doc.paragraphs:
                text_parts.append(para.text)
            return "\n\n".join(text_parts), "docx"
        except ImportError:
            raise ImportError("python-docx is required to read DOCX files. Install with: pip install python-docx")

    elif file_ext == ".csv":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read(), "csv"

    elif file_ext == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return json.dumps(data, indent=2), "json"

    else:
        # Try to read as text
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read(), "text"
        except UnicodeDecodeError:
            raise ValueError(f"Unsupported file type: {file_ext}")


def _chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
) -> List[Tuple[str, int, int]]:
    """
    Split text into overlapping chunks.
    Returns list of (chunk_text, start_char, end_char)
    """
    if not text:
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)

        # Try to find a good break point (sentence end, paragraph, or word boundary)
        if end < text_length:
            # Look for paragraph break
            para_break = text.rfind("\n\n", start, end)
            if para_break > start + chunk_size // 2:
                end = para_break + 2
            else:
                # Look for sentence end
                for sep in [". ", ".\n", "? ", "!\n", "! ", "?\n"]:
                    sent_break = text.rfind(sep, start, end)
                    if sent_break > start + chunk_size // 2:
                        end = sent_break + len(sep)
                        break
                else:
                    # Look for word boundary
                    space = text.rfind(" ", start, end)
                    if space > start + chunk_size // 2:
                        end = space + 1

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append((chunk_text, start, end))

        # Move start with overlap
        start = end - chunk_overlap
        if start < 0:
            start = 0
        if start >= text_length:
            break
        # Ensure we make progress
        if len(chunks) > 0 and start <= chunks[-1][1]:
            start = end

    return chunks


def _get_embeddings(texts: List[str]) -> Optional[Any]:
    """Get embeddings for a list of texts using OpenAI."""
    if not NUMPY_AVAILABLE or not OPENAI_AVAILABLE:
        return None

    client = _get_openai_client()
    if not client:
        return None

    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts
        )
        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings, dtype=np.float32)
    except Exception as e:
        print(f"Error getting embeddings: {e}")
        return None


def _build_faiss_index(embeddings: Any) -> Optional[Any]:
    """Build a FAISS index from embeddings."""
    if not FAISS_AVAILABLE or not NUMPY_AVAILABLE:
        return None

    if embeddings is None or len(embeddings) == 0:
        return None

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


def _rebuild_index():
    """Rebuild the FAISS index from all chunks."""
    global _rag_store

    if not _rag_store["chunks"]:
        _rag_store["embeddings"] = None
        _rag_store["faiss_index"] = None
        return

    # Get embeddings for all chunks
    texts = [chunk.text for chunk in _rag_store["chunks"]]
    embeddings = _get_embeddings(texts)

    if embeddings is not None:
        _rag_store["embeddings"] = embeddings
        _rag_store["faiss_index"] = _build_faiss_index(embeddings)


# ============================================
# INTERNAL FUNCTIONS (for UI)
# ============================================

def ingest_document_internal(
    file_path: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
) -> Dict[str, Any]:
    """
    Internal function to ingest a document (called from Streamlit UI).
    Returns dict with status and stats.
    """
    global _rag_store

    if not NUMPY_AVAILABLE:
        return {"success": False, "error": "numpy is required. Install with: pip install numpy"}

    if not FAISS_AVAILABLE:
        return {"success": False, "error": "faiss-cpu is required. Install with: pip install faiss-cpu"}

    if not OPENAI_AVAILABLE:
        return {"success": False, "error": "openai is required. Install with: pip install openai"}

    if not _get_openai_client():
        return {"success": False, "error": "OpenAI API key not found. Please set OPENAI_API_KEY environment variable."}

    try:
        # Load document
        content, doc_type = _load_document(file_path)
        document_name = os.path.basename(file_path)

        if not content.strip():
            return {"success": False, "error": "Document is empty"}

        # Check if document already exists
        if document_name in _rag_store["documents"]:
            # Remove existing document first
            remove_document_internal(document_name)

        # Chunk the document
        chunk_data = _chunk_text(content, chunk_size, chunk_overlap)

        if not chunk_data:
            return {"success": False, "error": "No chunks could be created from document"}

        # Create DocumentChunk objects
        new_chunks = []
        total_tokens = 0
        for i, (chunk_text, start, end) in enumerate(chunk_data):
            chunk = DocumentChunk(
                text=chunk_text,
                document_name=document_name,
                chunk_index=i,
                start_char=start,
                end_char=end,
                metadata={"doc_type": doc_type}
            )
            new_chunks.append(chunk)
            total_tokens += _count_tokens(chunk_text)

        # Add chunks to store
        _rag_store["chunks"].extend(new_chunks)

        # Record document info
        _rag_store["documents"][document_name] = {
            "chunks": len(new_chunks),
            "tokens": total_tokens,
            "upload_time": datetime.now().isoformat(),
            "file_path": file_path,
            "doc_type": doc_type
        }

        # Update metadata
        _rag_store["metadata"]["total_documents"] = len(_rag_store["documents"])
        _rag_store["metadata"]["total_chunks"] = len(_rag_store["chunks"])
        _rag_store["metadata"]["total_tokens"] = sum(
            doc["tokens"] for doc in _rag_store["documents"].values()
        )
        _rag_store["metadata"]["last_updated"] = datetime.now().isoformat()

        # Rebuild FAISS index
        _rebuild_index()

        return {
            "success": True,
            "document_name": document_name,
            "chunks": len(new_chunks),
            "tokens": total_tokens,
            "doc_type": doc_type
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


def remove_document_internal(document_name: str) -> Dict[str, Any]:
    """
    Internal function to remove a document (called from Streamlit UI).
    """
    global _rag_store

    if document_name not in _rag_store["documents"]:
        return {"success": False, "error": f"Document '{document_name}' not found"}

    # Remove chunks for this document
    _rag_store["chunks"] = [
        chunk for chunk in _rag_store["chunks"]
        if chunk.document_name != document_name
    ]

    # Remove document record
    del _rag_store["documents"][document_name]

    # Update metadata
    _rag_store["metadata"]["total_documents"] = len(_rag_store["documents"])
    _rag_store["metadata"]["total_chunks"] = len(_rag_store["chunks"])
    _rag_store["metadata"]["total_tokens"] = sum(
        doc["tokens"] for doc in _rag_store["documents"].values()
    )
    _rag_store["metadata"]["last_updated"] = datetime.now().isoformat()

    # Rebuild FAISS index
    _rebuild_index()

    return {"success": True, "document_name": document_name}


def _reset_rag_store():
    """Reset the entire RAG store (clear all documents)."""
    global _rag_store
    _rag_store = {
        "chunks": [],
        "embeddings": None,
        "faiss_index": None,
        "documents": {},
        "metadata": {
            "total_chunks": 0,
            "total_tokens": 0,
            "total_documents": 0,
            "last_updated": None
        }
    }


def get_rag_store_state() -> Dict[str, Any]:
    """Get current state of RAG store (for UI sync)."""
    return {
        "documents": dict(_rag_store["documents"]),
        "metadata": dict(_rag_store["metadata"])
    }


# ============================================
# AUTOMATIC CONTEXT RETRIEVAL (for query augmentation)
# ============================================

def get_relevant_context(
    query: str,
    top_k: int = 5,
    min_score: float = 0.3
) -> Optional[str]:
    """
    Search the knowledge base and return relevant context for a query.
    This is called automatically before sending queries to agents.

    Args:
        query: The user's query
        top_k: Maximum number of chunks to return (default: 5)
        min_score: Minimum similarity score to include (default: 0.3)

    Returns:
        Formatted context string to inject into prompt, or None if no relevant content
    """
    if not _rag_store["chunks"]:
        return None

    if not _rag_store["faiss_index"]:
        return None

    if not NUMPY_AVAILABLE or not FAISS_AVAILABLE:
        return None

    # Get query embedding
    query_embedding = _get_embeddings([query])
    if query_embedding is None:
        return None

    # Search FAISS index
    k = min(top_k, len(_rag_store["chunks"]))
    distances, indices = _rag_store["faiss_index"].search(query_embedding, k)

    # Convert L2 distances to similarity scores
    # Use a better normalization approach
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(_rag_store["chunks"]):
            continue

        # Convert L2 distance to similarity score
        # L2 distance of 0 = perfect match, higher = less similar
        # Use exponential decay for better score distribution
        similarity = 1.0 / (1.0 + dist)

        if similarity < min_score:
            continue

        chunk = _rag_store["chunks"][idx]
        results.append({
            "score": similarity,
            "source": chunk.document_name,
            "text": chunk.text
        })

    if not results:
        return None

    # Format context for injection
    context_parts = [
        "[KNOWLEDGE BASE CONTEXT]",
        "The following information from uploaded documents may be relevant to your query:",
        ""
    ]

    for i, r in enumerate(results, 1):
        context_parts.append(f"--- Source: {r['source']} (Relevance: {r['score']:.0%}) ---")
        context_parts.append(r['text'])
        context_parts.append("")

    context_parts.append("[END OF KNOWLEDGE BASE CONTEXT]")
    context_parts.append("")

    return "\n".join(context_parts)


def has_knowledge_base_content() -> bool:
    """Check if there is any content in the knowledge base."""
    return len(_rag_store["chunks"]) > 0


def get_rag_tools() -> List:
    """Get all RAG tools as a list (empty - RAG works automatically now)."""
    return []
