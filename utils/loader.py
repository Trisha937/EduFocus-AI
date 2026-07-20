"""PDF loading utilities for EduFocus."""

from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader


def load_pdf(file_path: str) -> list[Document]:
    """Load a PDF file into LangChain Document objects.

    Args:
        file_path: Path to the PDF file to load.

    Returns:
        A list of LangChain Document objects.

    Raises:
        FileNotFoundError: If the provided file does not exist.
        ValueError: If the PDF cannot be read or contains no text.
    """
    if not file_path:
        raise ValueError("A file path is required.")

    pdf_path = Path(file_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"The PDF file was not found: {file_path}")

    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()

    if not documents:
        raise ValueError("The PDF did not contain any readable text.")

    return documents


def get_document_info(documents: list[Document], file_name: str) -> dict[str, Any]:
    """Create a summary dictionary for the loaded document.

    Args:
        documents: Loaded LangChain Document objects.
        file_name: Original name of the uploaded file.

    Returns:
        A dictionary containing the file name and total page count.

    Raises:
        ValueError: If no documents are provided.
    """
    if not documents:
        raise ValueError("No documents were provided.")

    return {
        "file_name": file_name,
        "total_pages": len(documents),
    }
