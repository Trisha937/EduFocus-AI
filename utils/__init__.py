"""Utility package for EduFocus.

Exported modules:
- loader: PDF loading utilities (Sub-Group A)
- splitter: Document chunking utilities (Sub-Group A)
- embeddings: Local embedding model utilities (Track 2)
- vector_store: FAISS vector database utilities (Track 2)
- chat_memory: Sliding window chat history (Track 2)
- prompts: System prompts and context formatting (Track 1 placeholders + Track 2)
- chat_engine: Groq streaming chat integration (Track 2)
"""

from .loader import get_document_info, load_pdf
from .splitter import average_chunk_size, chunk_statistics, split_document
from .embeddings import get_embeddings_model, embed_chunks
from .vector_store import create_vector_store, save_vector_store, load_vector_store, retrieve_similar_chunks
from .chat_memory import format_chat_history, slice_sliding_history, add_message
from .prompts import format_context_with_citations, extract_unique_pages
from .chat_engine import get_groq_chat_model, stream_chat_response, get_page_citations

__all__ = [
    "get_document_info",
    "load_pdf",
    "average_chunk_size",
    "chunk_statistics",
    "split_document",
    "get_embeddings_model",
    "embed_chunks",
    "create_vector_store",
    "save_vector_store",
    "load_vector_store",
    "retrieve_similar_chunks",
    "format_chat_history",
    "slice_sliding_history",
    "add_message",
    "format_context_with_citations",
    "extract_unique_pages",
    "get_groq_chat_model",
    "stream_chat_response",
    "get_page_citations",
]
