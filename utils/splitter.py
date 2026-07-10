"""Document chunking utilities for EduFocus AI."""

from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_document(documents: list[Document]) -> list[Document]:
    """Split loaded documents into overlapping chunks while preserving metadata.

    Args:
        documents: LangChain Document objects to split.

    Returns:
        A list of chunked LangChain Document objects.
    """
    if not documents:
        raise ValueError("No documents were provided for chunking.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = splitter.split_documents(documents)
    return chunks


def average_chunk_size(chunks: list[Document]) -> int:
    """Return the average size of the provided document chunks."""
    if not chunks:
        return 0

    total_length = sum(len(chunk.page_content) for chunk in chunks)
    return round(total_length / len(chunks))


def chunk_statistics(chunks: list[Document]) -> dict[str, Any]:
    """Return total chunk count and average chunk size."""
    return {
        "total_chunks": len(chunks),
        "average_chunk_size": average_chunk_size(chunks),
    }
