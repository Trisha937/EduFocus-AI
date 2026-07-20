"""Local embedding utilities for EduFocus.

Uses HuggingFace's sentence-transformers/all-MiniLM-L6-v2 model running locally
to avoid cloud embedding rate limits and costs.
"""

from typing import Any

from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings


def get_embeddings_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> HuggingFaceEmbeddings:
    """Initialize the local sentence-transformers embedding model.

    Args:
        model_name: HuggingFace model identifier. Defaults to all-MiniLM-L6-v2
                   which is lightweight and CPU-optimized.

    Returns:
        HuggingFaceEmbeddings instance configured for local execution.
    """
    # model_kwargs ensures we run on CPU (no GPU required)
    # encode_kwargs controls how the model processes text
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )


def embed_chunks(chunks: list[Document], embeddings_model: HuggingFaceEmbeddings) -> Any:
    """Create embeddings for document chunks.

    Args:
        chunks: List of LangChain Document objects to embed.
        embeddings_model: Initialized HuggingFaceEmbeddings model.

    Returns:
        Embedding vectors (numpy array) for each chunk.
    """
    texts = [chunk.page_content for chunk in chunks]
    return embeddings_model.embed_documents(texts)