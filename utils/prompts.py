"""System prompts for EduFocus AI.

This module contains placeholder functions for Track 1 (Persona & Quiz Architect).
Track 2 developers should call these functions to get appropriate prompts.
"""

# ============================================================================
# TRACK 1 - PLACEHOLDER FUNCTIONS (to be implemented by Quiz Architect)
# ============================================================================

def get_system_prompt(learning_level: str) -> str:
    """Return the system prompt based on learning level.

    TRACK 1 TODO: Implement level-specific prompts.
    - Beginner: Simple explanations, analogies, avoid jargon
    - Intermediate: Advanced terminology, concise explanations

    Args:
        learning_level: 'Beginner' or 'Intermediate'.

    Returns:
        System prompt string for the LLM.
    """
    # Placeholder implementation - Track 1 will provide the actual logic
    return ""


def format_quiz_prompt(context: str, num_questions: int = 5) -> str:
    """Format the prompt for quiz generation.

    TRACK 1 TODO: Implement quiz prompt formatting with JSON output structure.

    Args:
        context: Retrieved document chunks to base quiz on.
        num_questions: Number of quiz questions to generate.

    Returns:
        Formatted prompt for quiz generation.
    """
    # Placeholder implementation - Track 1 will provide the actual logic
    return ""


def parse_quiz_response(response: str) -> dict:
    """Parse the LLM quiz response into structured JSON format.

    TRACK 1 TODO: Implement JSON parsing with error handling.

    Args:
        response: Raw LLM response containing quiz JSON.

    Returns:
        Parsed quiz dictionary.
    """
    # Placeholder implementation - Track 1 will provide the actual logic
    return {"questions": []}


# ============================================================================
# TRACK 2 - CONTEXT FORMATTING (uses retrieved chunks)
# ============================================================================

def format_context_with_citations(chunks: list, source_file: str) -> str:
    """Format retrieved chunks into context string with page citations.

    This formats the retrieved context so the LLM can reference page numbers.
    After streaming, we extract unique pages for citation display.

    Args:
        chunks: List of retrieved Document objects with metadata.
        source_file: Name of the source PDF file.

    Returns:
        Formatted context string with [Page X] markers (1-indexed).
    """
    formatted_parts = []
    for chunk in chunks:
        # DEFAULT to 0 if page metadata is missing - CRITICAL: must be int, not string
        # This prevents "unknown" from breaking downstream code expecting integers
        page_num = chunk.metadata.get("page", 0)
        # Add +1 to convert 0-indexed FAISS page numbers to 1-indexed for users
        # Users think "Page 1" not "Page 0"
        formatted_parts.append(f"[Page {page_num + 1}]\n{chunk.page_content}")

    return "\n\n".join(formatted_parts)


def extract_unique_pages(chunks: list) -> list[int]:
    """Extract unique page numbers from retrieved chunks.

    Used after chat streaming to display citation footnotes (1-indexed).

    Args:
        chunks: List of retrieved Document objects.

    Returns:
        Sorted list of unique page numbers (1-indexed).
    """
    pages = set()
    for chunk in chunks:
        # Get page number, default to 0 if missing
        page = chunk.metadata.get("page", 0)
        # Add +1 to convert to 1-indexed for user display
        pages.add(page + 1)

    # Return sorted list for clean display (e.g., "Pages 1, 3, 5")
    return sorted(pages)