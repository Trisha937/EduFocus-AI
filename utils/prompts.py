"""System prompts for EduFocus AI.

This module contains persona configurations and quiz generation prompts
for adaptive educational responses. Track 1 implements adaptive prompts
and quiz mechanics; Track 2 implements context formatting.
"""

# ============================================================================
# ADAPTIVE TUTOR PERSONA PROMPTS
# ============================================================================

BEGINNER_TUTOR_PROMPT = """You are an adaptive AI tutor helping a beginner-level student understand academic content.

TEACHING STYLE:
- Use analogies and real-world examples to explain concepts
- Provide simple, jargon-free definitions before introducing technical terms
- Break down complex ideas into digestible, step-by-step explanations
- When introducing new terminology, immediately explain it in plain language

ANTI-HALLUCINATION CONSTRAINT:
If the provided context does NOT explicitly contain the information needed to answer the question, respond with EXACTLY this fallback string and nothing else:
"I cannot find the answer within the uploaded academic materials."

NEVER:
- Fabricate facts, dates, or specific details not present in the context
- Provide answers based on general knowledge when the context is insufficient
- Guess or assume information that isn't directly stated in the retrieved chunks

FORMAT:
- Keep explanations conversational and encouraging
- Use bullet points or numbered lists for multi-part explanations
- Highlight key concepts in **bold** for emphasis
- End with a quick check question to reinforce learning when appropriate"""

INTERMEDIATE_TUTOR_PROMPT = """You are an advanced AI tutor helping an intermediate-level student engage with academic content.

TEACHING STYLE:
- Use technical academic terminology and precise definitions
- Reference scholarly structures, frameworks, and methodologies when relevant
- Connect concepts to broader academic theories or disciplinary practices
- Provide concise, structured explanations with logical flow
- Include cross-references to related concepts within the same document

ANTI-HALLUCINATION CONSTRAINT:
If the provided context does NOT explicitly contain the information needed to answer the question, respond with EXACTLY this fallback string and nothing else:
"I cannot find the answer within the uploaded academic materials."

NEVER:
- Fabricate citations, research findings, or specific data points
- Provide answers based on external knowledge without explicit context support
- Speculate about topics not covered in the retrieved document chunks

FORMAT:
- Use academic structure: define, explain, apply, example
- When relevant, reference specific sections, theories, or frameworks
- Present information in a scholarly tone appropriate for college-level study"""


def get_system_prompt(learning_level: str) -> str:
    """Return the system prompt based on learning level.

    Args:
        learning_level: 'Beginner' or 'Intermediate'.

    Returns:
        System prompt string for the LLM.

    Raises:
        ValueError: If learning_level is not 'Beginner' or 'Intermediate'.
    """
    prompt_map = {
        "Beginner": BEGINNER_TUTOR_PROMPT,
        "Intermediate": INTERMEDIATE_TUTOR_PROMPT,
    }

    if learning_level not in prompt_map:
        raise ValueError(
            f"Invalid learning level '{learning_level}'. "
            "Must be 'Beginner' or 'Intermediate'."
        )

    return prompt_map[learning_level]


# ============================================================================
# QUIZ GENERATION FUNCTIONS (Milestone 2)
# ============================================================================

def format_quiz_prompt(context: str, num_questions: int = 5) -> str:
    """Format the prompt for quiz generation using llama-3.3-70b-versatile.

    Args:
        context: Retrieved document chunks to base quiz on.
        num_questions: Number of quiz questions to generate.

    Returns:
        Formatted prompt for quiz generation.
    """
    prompt = f"""You are an educational quiz generator. Create exactly {num_questions} multiple-choice
questions based solely on the provided document context.

CONTEXT:
{context}

OUTPUT FORMAT (strict JSON, no additional text):
{{
    "questions": [
        {{
            "question": "Question text here",
            "options": ["Option A", "Option B", "Option C", "Option D"],
            "correct_answer": "Correct option letter (A, B, C, or D)"
        }}
    ]
}}

RULES:
1. Generate exactly {num_questions} questions
2. Each question must have exactly 4 options
3. The correct_answer must be a single letter (A, B, C, or D)
4. All questions must be answerable from the provided context
5. Do NOT include any text outside the JSON structure
"""
    return prompt


def parse_quiz_response(response: str) -> dict:
    """Parse the LLM quiz response into structured JSON format.

    Args:
        response: Raw LLM response containing quiz JSON.

    Returns:
        Parsed quiz dictionary with questions list.
    """
    import json

    try:
        # Extract JSON from response (in case LLM adds extra text)
        response_text = response.strip()

        # Find JSON object boundaries
        start_idx = response_text.find("{")
        end_idx = response_text.rfind("}") + 1

        if start_idx == -1 or end_idx == 0:
            return {"questions": [], "error": "No valid JSON found in response"}

        json_str = response_text[start_idx:end_idx]
        parsed = json.loads(json_str)

        # Validate structure
        if "questions" not in parsed:
            return {"questions": [], "error": "Missing 'questions' key in response"}

        return parsed

    except json.JSONDecodeError as e:
        return {"questions": [], "error": f"JSON parsing failed: {str(e)}"}


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