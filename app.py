import pickle
import json
from dotenv import load_dotenv
import os
load_dotenv()  # This loads the environment variable from the .env file
from datetime import datetime
from pathlib import Path

import streamlit as st
from groq import Groq

from utils.chat_engine import get_page_citations, stream_chat_response
from utils.chat_memory import add_message
from utils.embeddings import get_embeddings_model
from utils.loader import get_document_info, load_pdf
from utils.prompts import get_system_prompt, format_quiz_prompt
from utils.splitter import chunk_statistics, split_document
from utils.vector_store import create_vector_store, retrieve_similar_chunks, save_vector_store

from database.vector_store import (
    create_and_save_vector_db,
    load_local_vector_db,
    retrieve_context_with_citations
)

def get_default_chat_state() -> dict[str, object]:
    """Return the default session state structure for the chat experience."""
    return {
        "chat_messages": [],
        "chat_history": [],
        "vector_store": None,
        "source_file": None,
    }


def initialize_chat_session_state() -> None:
    """Ensure the Streamlit session state contains the chat state keys."""
    defaults = get_default_chat_state()
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def save_chunks(chunks, output_path: Path) -> None:
    """Persist processed document chunks to disk using pickle."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(chunks, handle)


def format_file_size(size_bytes: int) -> str:
    """Format file sizes into a readable KB or MB string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"


def format_preview_text(text: str, limit: int = 500) -> str:
    """Trim preview text and append an ellipsis when needed."""
    if len(text) <= limit:
        return text
    return f"{text[:limit].rstrip()}..."


def main() -> None:
    """Run the Streamlit application for EduFocus."""
    st.set_page_config(page_title="EduFocus", page_icon="📚", layout="wide")
    initialize_chat_session_state()

    uploaded_file = None
    uploaded_time = None
    if "is_processed" not in st.session_state:
        st.session_state.is_processed = False
    if "doc_info" not in st.session_state:
        st.session_state.doc_info = {}
    if "stats" not in st.session_state:
        st.session_state.stats = {}
    if "first_chunk_preview" not in st.session_state:
        st.session_state.first_chunk_preview = ""
    if "first_chunk_metadata" not in st.session_state:
        st.session_state.first_chunk_metadata = {}
    if "vector_db" not in st.session_state:
        st.session_state.vector_db = load_local_vector_db()
    if "uploaded_file_name" not in st.session_state:
        st.session_state.uploaded_file_name = ""
    if "uploaded_file_size" not in st.session_state:
        st.session_state.uploaded_file_size = 0
    if "uploaded_time" not in st.session_state:
        st.session_state.uploaded_time = ""    

    st.markdown(
        "<div style='padding: 0.4rem 0 1rem 0;'>"
        "<h1 style='margin-bottom: 0.2rem; font-size: 2.4rem;'>📚 EduFocus</h1>"
        "<h3 style='margin-top: 0; color: #5b5b5b;'>Adaptive RAG Assistant for Grounded Academic Learning</h3>"
        "<p style='margin-top: 0.3rem; color: #6b7280;'>"
        "Transform static academic PDFs into structured knowledge while preserving source metadata for downstream AI workflows."
        "</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    with st.sidebar:
        st.markdown("## 📄 Upload Academic Material")
        st.caption("Prepare academic PDFs for downstream AI workflows.")

        uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

        st.markdown("### Supported Formats")
        st.markdown("- ✓ PDF")
        st.markdown("- DOCX (Coming Soon)")
        st.markdown("- TXT (Coming Soon)")

        st.divider()
        process_button = st.button("Process Document", type="primary", use_container_width=True)
        st.caption("Processing preserves metadata and creates chunk-ready output.")

    # --- Step 1: Processing Logic ---
    if process_button:
        if uploaded_file is None:
            st.error("Please upload a PDF file before processing.")
            return

        if not uploaded_file.name.lower().endswith(".pdf"):
            st.error("Unsupported file type. Please upload a PDF document.")
            return

        uploaded_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.chat_messages = []
        st.session_state.chat_history = []
        st.session_state.vector_store = None
        st.session_state.source_file = None

        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)
        temp_pdf_path = temp_dir / uploaded_file.name

        try:
            uploaded_bytes = uploaded_file.getvalue()
            if not uploaded_bytes:
                st.error("The uploaded PDF is empty.")
                return

            with temp_pdf_path.open("wb") as handle:
                handle.write(uploaded_bytes)

            progress_bar = st.progress(0.0)
            status_text = st.empty()
            with st.spinner("🔄 Processing document..."):
                status_text.markdown("**Loading PDF...**")
                progress_bar.progress(0.2)
                documents = load_pdf(str(temp_pdf_path))
                if not documents:
                    st.error("The uploaded PDF did not contain any readable text.")
                    return

                status_text.markdown("**Extracting text...**")
                progress_bar.progress(0.4)
                document_info = get_document_info(documents, uploaded_file.name)

                status_text.markdown("**Splitting into chunks...**")
                progress_bar.progress(0.7)
                chunks = split_document(documents)

                if not chunks:
                    st.error("The PDF produced no readable chunks.")
                    return

                first_chunk = chunks[0]
                if not getattr(first_chunk, "metadata", None):
                    st.error("Chunk metadata was not preserved correctly.")
                    return

                if "page" not in first_chunk.metadata or "source" not in first_chunk.metadata:
                    st.error("Chunk metadata is missing required page or source information.")
                    return
                

                status_text.markdown("**Generating local vector database...**")
                progress_bar.progress(0.8)
                vector_db = create_and_save_vector_db(chunks)
                st.session_state.vector_db = vector_db

                status_text.markdown("**Saving processed data...**")
                progress_bar.progress(0.9)
                save_chunks(chunks, temp_dir / "chunks.pkl")

                try:
                    embeddings_model = get_embeddings_model()
                    vector_store = create_vector_store(chunks, embeddings_model)
                    save_vector_store(vector_store)
                    st.session_state.vector_store = vector_store
                    st.session_state.source_file = uploaded_file.name
                except Exception as exc:  # pragma: no cover - graceful fallback for local indexing
                    st.session_state.vector_store = None
                    st.session_state.source_file = uploaded_file.name
                    st.warning(f"The document was processed, but chat indexing could not be built: {exc}")

                statistics = chunk_statistics(chunks)
                progress_bar.progress(1.0)
                status_text.markdown("**Completed**")

        except FileNotFoundError:
            st.error("The selected PDF file could not be found.")
        except ValueError as exc:
            st.error(f"Invalid PDF content: {exc}")
        except Exception as exc:  # pragma: no cover - defensive UI error handling
            st.error(
                "Unable to process the PDF. Please ensure the file is a valid, readable PDF. "
                f"Details: {exc}"
            )
        else:
            # Safely store execution results to state
            st.session_state.is_processed = True
            st.session_state.uploaded_file_name = uploaded_file.name
            st.session_state.uploaded_file_size = uploaded_file.size
            st.session_state.uploaded_time = uploaded_time
            st.session_state.doc_info = document_info
            st.session_state.stats = statistics
            st.session_state.first_chunk_preview = format_preview_text(first_chunk.page_content)
            st.session_state.first_chunk_metadata = first_chunk.metadata
            st.rerun()

    # --- Step 1.5: Learning Level Selector ---
    if st.session_state.is_processed:
        st.session_state.learning_level = st.selectbox(
            "🎓 Learning Level",
            options=["Beginner", "Intermediate"],
            index=0,
            help="Beginner: Simple explanations with analogies | Intermediate: Technical academic terminology"
        )

    # --- Step 2: Main Interface Rendering Logic ---
    if st.session_state.is_processed:
        st.success(
            "✅ Document processed successfully! The document has been prepared and is ready for vector indexing."
        )
        
        # with st.container():
        #     st.markdown("### 📊 Processing Summary")
        #     st.caption("A concise view of the processed document and its preparation status.")
        #     summary_cols = st.columns(2)
        #     with summary_cols[0]:
        #         st.markdown(
        #             "**File Name**\n"
        #             f"{st.session_state.uploaded_file_name}"
        #         )
        #         st.markdown(
        #             "**File Size**\n"
        #             f"{format_file_size(st.session_state.uploaded_file_size)}"
        #         )
        #         st.markdown(
        #             "**Upload Time**\n"
        #             f"{st.session_state.uploaded_time}"
        #         )
        #     with summary_cols[1]:
        #         st.markdown(
        #             "**Total Pages**\n"
        #             f"{st.session_state.doc_info['total_pages']}"
        #         )
        #         st.markdown(
        #             "**Total Chunks**\n"
        #             f"{st.session_state.stats['total_chunks']}"
        #         )
        #         st.markdown(
        #             "**Average Chunk Size**\n"
        #             f"{st.session_state.stats['average_chunk_size']}"
        #         )
        #         st.markdown("**Processing Status**\nReady")

        # st.divider()
        # st.markdown("## 📄 Document Information")
        # metric_columns = st.columns(4)
        # with metric_columns[0]:
        #     st.metric("📄 Document", st.session_state.uploaded_file_name)
        # with metric_columns[1]:
        #     st.metric("📚 Pages", st.session_state.doc_info["total_pages"])
        # with metric_columns[2]:
        #     st.metric("🧩 Chunks", st.session_state.stats["total_chunks"])
        # with metric_columns[3]:
        #     st.metric("📏 Avg Chunk Size", st.session_state.stats["average_chunk_size"])

        # st.divider()
        # st.markdown("## 🔍 Metadata Preview")
        # st.caption("This metadata will later be used to generate page citations.")
        # with st.expander("View metadata", expanded=False):
        #     st.json(st.session_state.first_chunk_metadata)

        # st.divider()
        # st.markdown("## 📄 First Chunk Preview")
        # st.caption("The first chunk is shown below for inspection and validation.")
        # with st.container():
        #     st.code(st.session_state.first_chunk_preview, language="text")

        st.divider()
        chunks_path = Path("temp/chunks.pkl")
        if chunks_path.exists():
            with chunks_path.open("rb") as handle:
                pass
                # st.download_button(
                #     label="📥 Download Processed Chunks",
                #     data=handle,
                #     file_name="chunks.pkl",
                #     mime="application/octet-stream",
                # )

            st.divider()
            st.markdown("## 💬 Chat with the Document")
            st.caption("Ask questions about the uploaded PDF using the indexed chunks and the built-in RAG flow.")

            for message in st.session_state.chat_messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    # Display persistent citations for assistant messages
                    if message["role"] == "assistant" and message.get("citations"):
                        st.caption(f"Sources: Pages {', '.join(map(str, message['citations']))}")

            if st.session_state.vector_store is None:
                st.info("The chat index is not available yet. The document was processed, but the embedding/indexing step could not be completed.")
            else:
                st.info("The chat module is ready. Ask a question about the document below.")

            prompt = st.chat_input("Ask a question about the uploaded PDF")
            if prompt:
                st.session_state.chat_messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                if st.session_state.vector_store is None:
                    response = "The document index is not ready yet. Please reprocess the PDF or check the embedding setup."
                    st.session_state.chat_messages.append({"role": "assistant", "content": response})
                    with st.chat_message("assistant"):
                        st.markdown(response)
                else:
                    try:
                        retrieved_chunks = retrieve_similar_chunks(st.session_state.vector_store, prompt, k=4)
                        if not retrieved_chunks:
                            raise ValueError("No relevant document chunks were found.")

                        system_prompt = get_system_prompt(st.session_state.learning_level) or "You are an academic assistant helping a student understand the uploaded document."
                        response_placeholder = st.empty()
                        full_response = ""
                        for token in stream_chat_response(
                            query=prompt,
                            chat_history=st.session_state.chat_history,
                            retrieved_chunks=retrieved_chunks,
                            system_prompt=system_prompt,
                            source_file=st.session_state.source_file or uploaded_file.name,
                        ):
                            full_response += token
                            response_placeholder.markdown(full_response + "▌")

                        response_placeholder.markdown(full_response)
                        citations = get_page_citations(retrieved_chunks)

                        st.session_state.chat_history = add_message(st.session_state.chat_history, "human", prompt)
                        st.session_state.chat_history = add_message(st.session_state.chat_history, "ai", full_response)
                        st.session_state.chat_messages.append({"role": "assistant", "content": full_response, "citations": citations})

                        # Display inline citation (appears below response)
                        if citations:
                            st.caption(f"Sources: Pages {', '.join(map(str, citations))}")
                    except Exception as exc:
                        error_response = f"I could not generate a response right now: {exc}"
                        st.session_state.chat_messages.append({"role": "assistant", "content": error_response})
                        with st.chat_message("assistant"):
                            st.markdown(error_response)

            st.divider()
            st.markdown("## ✅ Project Status")
            st.markdown("- ✅ PDF Loading Complete")
            st.markdown("- ✅ Metadata Preserved")
            st.markdown("- ✅ Document Chunking Complete")
            st.markdown("- ✅ Statistics Generated")
            st.markdown("- ✅ Chat Module Connected")

            # ============================================================================
            # QUIZ DASHBOARD
            # ============================================================================

            st.divider()
            st.markdown("## 📝 Interactive Quiz Dashboard")
            st.caption("Test your knowledge with AI-generated questions based on the document.")

            # Initialize quiz state
            if "quiz_data" not in st.session_state:
                st.session_state.quiz_data = {"questions": []}
            if "user_answers" not in st.session_state:
                st.session_state.user_answers = {}
            if "quiz_generated" not in st.session_state:
                st.session_state.quiz_generated = False

            quiz_form = st.form("quiz_form")
            with quiz_form:
                generate_col, _ = st.columns([2, 3])
                with generate_col:
                    generate_quiz = st.form_submit_button("🔄 Generate Quiz", type="primary")

                if generate_quiz:
                    if st.session_state.vector_db is not None:
                        with st.spinner("Generating quiz questions..."):
                            # Retrieve context for quiz generation
                            import random
                            random_queries = [
                                "Extract key facts and figures for quiz questions",
                                "Identify important concepts for learning assessment",
                                "Find critical definitions and principles to test",
                                "Retrieve core arguments and evidence for questioning",
                            ]
                            quiz_context, _ = retrieve_context_with_citations(
                                random.choice(random_queries),  # RANDOM QUERY
                                st.session_state.vector_db,
                            )

                            if quiz_context:
                                quiz_prompt = format_quiz_prompt(quiz_context, 5)

                                try:
                                    # Initialize Groq Client
                                    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

                                    # Request structured JSON payload matching your UI format
                                    completion = client.chat.completions.create(
                                        model="llama-3.3-70b-versatile",
                                        messages=[
                                            {
                                                "role": "system",
                                                "content": (
                                                    "You are an evaluator. You must output raw JSON ONLY. "
                                                    "Follow this schema precisely: {\"questions\": [{\"question\": \"...\", "
                                                    "\"options\": [\"Ans A\", \"Ans B\", \"Ans C\", \"Ans D\"], "
                                                    "\"answer\": \"A\"}]}"
                                                ),
                                            },
                                            {"role": "user", "content": quiz_prompt},
                                        ],
                                        temperature=0.7,
                                        response_format={"type": "json_object"},
                                    )

                                    # Parse out JSON and store into session state
                                    parsed_quiz = json.loads(completion.choices[0].message.content)
                                    st.session_state.quiz_data = parsed_quiz
                                    st.session_state.quiz_generated = True
                                    st.rerun()

                                except Exception as exc:
                                    st.error(f"Failed to generate quiz from Groq: {exc}")
                            else:
                                st.warning("Could not generate quiz - no document context available.")
                    else:
                        st.error("Please process a document first to generate a quiz.")

            # Display quiz questions if available
            if st.session_state.quiz_data.get("questions"):
                with st.form("quiz_answers_form"):
                    correct_answers = {}
                    for i, q in enumerate(st.session_state.quiz_data["questions"]):
                        st.markdown(f"**Question {i+1}:** {q['question']}")
                        correct_answers[i] = q.get('answer', 'A')

                        # Radio buttons for options
                        answer_key = f"q_{i}"
                        st.radio(
                            f"Select answer for Q{i+1}",
                            options=["A", "B", "C", "D"],
                            format_func=lambda x: f"{x}) {q['options'][ord(x)-65]}",
                            key=answer_key,
                            horizontal=True
                        )

                    submitted = st.form_submit_button("Check Answers")
                    if submitted:
                        score = 0
                        total = len(st.session_state.quiz_data["questions"])

                        for i in range(total):
                            user_pick = st.session_state.get(f"q_{i}")
                            correct_ans = correct_answers[i]
                            if user_pick == correct_ans:
                                score += 1
                                st.success(f"✅ Question {i+1}: Correct! You selected {user_pick}.")
                            else:
                                # Incorrect answer formatting
                                st.error(f"❌ Question {i+1}: Incorrect")
                                options = st.session_state.quiz_data["questions"][i]["options"]
                                user_label = f"{user_pick}) {options[ord(user_pick)-65]}" if user_pick else "No answer"
                                correct_label = f"{correct_ans}) {options[ord(correct_ans)-65]}"

                                st.markdown(f"**Question:** {st.session_state.quiz_data['questions'][i]['question']}")
                                st.markdown(f"**Your Selection:** :red[{user_label}]")
                                st.markdown(f"**Correct Answer:** :green[{correct_label}]")
                                if 'explanation' in st.session_state.quiz_data["questions"][i]:
                                    st.info(st.session_state.quiz_data["questions"][i]["explanation"])


                        st.success(f"Quiz Evaluation Complete! Your score: {score}/{total}")

            st.divider()
            st.markdown("## 🔍 Semantic Document Search")
            st.caption("Ask questions about this document to retrieve context and page citations.")

            user_query = st.text_input(
                "What information are you looking for in this document?",
                placeholder="e.g., What are the grading rules or schedule policies?",
                key="doc_search_query"
            )

            if user_query:
                if st.session_state.vector_db is not None:
                    with st.spinner("Searching local FAISS index..."):
                        context, pages = retrieve_context_with_citations(user_query, st.session_state.vector_db)

                        if context:
                            st.success("Matching content retrieved!")
                            st.markdown(f"**📚 Source References:** Page(s) `{', '.join(map(str, pages))}`")
                            with st.expander("📖 View Retrieved Text Fragment", expanded=True):
                                st.write(context)
                        else:
                            st.warning("No matches found. Try matching key terms in your query.")
                else:
                    st.error("Database is not initialized. Try re-processing the document.")

            # st.divider()
            # st.markdown(
            #     "<div style='text-align: center; padding: 1.2rem 0 0.5rem 0; color: #6b7280;'>"
            #     "<h4 style='margin-bottom: 0.2rem;'>EduFocus</h4>"
            #     "<p style='margin: 0.1rem 0;'>Capstone Project</p>"
            #     "<p style='margin: 0.25rem 0 0;'>Member 1: Document Ingestion & Metadata Pipeline</p>"
            #     "<p style='margin-top: 0.5rem;'>Built with<br>• Streamlit<br>• LangChain<br>• PyPDFLoader</p>"
            #     "</div>",
            #     unsafe_allow_html=True,
            # )
    else:
        # Shown only if no document has been processed yet
        with st.container():
            st.markdown(
                "<div style='text-align: center; padding: 2rem 1rem; border: 1px solid #e5e7eb; border-radius: 14px; background: #f9fafb;'>"
                "<h3 style='margin-bottom: 0.4rem;'>👈 Upload an academic PDF from the sidebar to begin processing.</h3>"
                "<p style='margin-top: 0.2rem; color: #6b7280;'>Start with a course handout, paper, or lecture note.</p>"
                "<div style='font-size: 3rem; margin-top: 0.8rem;'>📘📄</div>"
                "</div>",
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()