"""Utility package for EduFocus AI."""

from .loader import get_document_info, load_pdf
from .splitter import average_chunk_size, chunk_statistics, split_document

__all__ = [
    "get_document_info",
    "load_pdf",
    "average_chunk_size",
    "chunk_statistics",
    "split_document",
]
