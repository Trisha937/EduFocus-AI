"""FAISS vector store utilities for EduFocus AI.

Local FAISS (CPU) indexing for similarity search with preserved metadata.
"""

from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


# Directory where FAISS indexes are saved/loaded
VECTOR_DB_DIR = Path("vector_db")


def create_vector_store(
    chunks: list[Document],
    embeddings_model: HuggingFaceEmbeddings,
    index_name: str = "faiss_index"
) -> FAISS:
    """Create a FAISS vector store from document chunks.

    Args:
        chunks: List of chunked Document objects with preserved metadata.
        embeddings_model: Initialized HuggingFace embeddings model.
        index_name: Name for the saved FAISS index.

    Returns:
        FAISS vector store instance ready for similarity search.
    """
    # from_documents handles embedding generation and index creation in one step
    # Each chunk's metadata (including page numbers) is preserved in the index
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings_model
    )
    return vector_store


def save_vector_store(vector_store: FAISS, index_name: str = "faiss_index") -> None:
    """Save FAISS vector store to disk for later retrieval.

    Args:
        vector_store: FAISS instance to save.
        index_name: Name of the index directory.
    """
    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(VECTOR_DB_DIR / index_name))


def load_vector_store(
    embeddings_model: HuggingFaceEmbeddings,
    index_name: str = "faiss_index"
) -> FAISS | None:
    """Load a previously saved FAISS vector store.

    Args:
        embeddings_model: Initialized HuggingFace embeddings model (required for loading).
        index_name: Name of the index directory to load.

    Returns:
        FAISS vector store instance, or None if not found.
    """
    index_path = VECTOR_DB_DIR / index_name
    if not index_path.exists():
        return None

    try:
        return FAISS.load_local(
            str(index_path),
            embeddings_model,
            allow_dangerous_deserialization=True  # Required for FAISS local loading
        )
    except Exception:
        return None


def retrieve_similar_chunks(
    vector_store: FAISS,
    query: str,
    k: int = 4
) -> list[Document]:
    """Retrieve the most similar document chunks for a query.

    Args:
        vector_store: FAISS vector store instance.
        query: User's question to search for.
        k: Number of top similar chunks to retrieve.

    Returns:
        List of Document objects with similarity scores and preserved metadata.
    """
    # similarity_search_with_score returns tuples of (Document, score)
    # where score is the cosine similarity (higher = more similar)
    results = vector_store.similarity_search_with_score(query, k=k)

    # Extract just the Document objects, preserving metadata
    return [doc for doc, score in results]