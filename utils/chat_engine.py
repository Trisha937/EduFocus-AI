"""Chat engine for EduFocus.

Integrates FAISS retrieval with Groq streaming for conversational RAG.
Uses Llama-3.1-8b-instant for fast, streamed chat responses.
"""

import os  # Used to read environment variables
from typing import Any, Generator  # Type hints for function signatures

# Import dotenv for .env file fallback - CRITICAL for secure key management
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from .prompts import format_context_with_citations, extract_unique_pages
from .chat_memory import format_chat_history


# Groq model for fast streaming chat (meets <2.5s latency target)
GROQ_MODEL = "llama-3.1-8b-instant"


def get_groq_chat_model() -> ChatGroq:
    """Initialize the Groq chat model for streaming.

    Uses llama-3.1-8b-instant for:
    - Fast response (meets <2.5s latency target)
    - Streaming support via LangChain
    - Free tier with generous rate limits

    Returns:
        Configured ChatGroq instance.

    Raises:
        ValueError: If GROQ_API_KEY is not found in environment or .env file.
    """
    # SECURITY PATTERN: Load .env file as fallback BEFORE checking environment
    # load_dotenv() looks for .env in current directory and loads it into os.environ
    load_dotenv()

    # Check environment first (allows override in production)
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. Please set it in your environment or add it to a .env file.\n"
            "Example .env content:\nGROQ_API_KEY=your_key_here"
        )

    return ChatGroq(
        model=GROQ_MODEL,
        temperature=0.7,
        max_tokens=1024,
        streaming=True,
        api_key=api_key,
    )


def stream_chat_response(
    query: str,
    chat_history: list[dict[str, str]],
    retrieved_chunks: list,
    system_prompt: str,
    source_file: str
) -> Generator[str, None, Any]:
    """Stream chat response with citations.

    This is the core Track 2 function that:
    1. Formats chat history into LangChain messages
    2. Retrieves relevant context from FAISS
    3. Streams response from Groq
    4. Extracts page citations for footnote display

    Args:
        query: User's current question.
        chat_history: List of {'role', 'content'} dicts (sliding window applied).
        retrieved_chunks: Documents from FAISS similarity search.
        system_prompt: System prompt (from Track 1 or placeholder).
        source_file: Name of source PDF for citations.

    Yields:
        Streaming text tokens as they arrive from Groq.
    """
    # Format retrieved context with page markers for LLM
    context = format_context_with_citations(retrieved_chunks, source_file)

    # Build message list: system prompt + history + current query
    # SystemMessage sets behavior, HumanMessage/AIMessage maintain dialogue
    messages = [SystemMessage(content=system_prompt)]

    # Add formatted chat history (sliding window already applied)
    # format_chat_history converts dicts to LangChain message objects
    messages.extend(format_chat_history(chat_history))

    # Add current query with context
    messages.append(HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}"))

    # Get streaming chat model
    chat_model = get_groq_chat_model()

    # Stream response - ChatGroq handles token-by-token generation
    # Each yielded token is a string chunk that Streamlit will display
    for chunk in chat_model.stream(messages):
        if hasattr(chunk, "content"):
            yield chunk.content
        else:
            # Handle string chunks
            yield str(chunk)


def get_page_citations(retrieved_chunks: list) -> list[int]:
    """Extract page numbers for citation footnotes.

    Called after streaming completes to show "Source: Pages 1, 3, 5"

    Args:
        retrieved_chunks: Documents that were used for the answer.

    Returns:
        Sorted list of unique page numbers.
    """
    return extract_unique_pages(retrieved_chunks)