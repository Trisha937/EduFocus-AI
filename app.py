import pickle
from datetime import datetime
from pathlib import Path

import streamlit as st

from utils.chat_engine import get_page_citations, stream_chat_response
from utils.chat_memory import add_message
from utils.embeddings import get_embeddings_model
from utils.loader import get_document_info, load_pdf
from utils.prompts import get_system_prompt
from utils.splitter import chunk_statistics, split_document
from utils.vector_store import create_vector_store, retrieve_similar_chunks, save_vector_store


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
    """Run the Streamlit application for EduFocus AI."""
    st.set_page_config(page_title="EduFocus AI", page_icon="📚", layout="wide")
    initialize_chat_session_state()

    uploaded_file = None
    uploaded_time = None

    st.markdown(
        "<div style='padding: 0.4rem 0 1rem 0;'>"
        "<h1 style='margin-bottom: 0.2rem; font-size: 2.4rem;'>📚 EduFocus AI</h1>"
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
            st.success(
                "✅ Document processed successfully! The document has been prepared and is ready for vector indexing."
            )

            with st.container():
                st.markdown("### 📊 Processing Summary")
                st.caption("A concise view of the processed document and its preparation status.")
                summary_cols = st.columns(2)
                with summary_cols[0]:
                    st.markdown(
                        "**File Name**\n"
                        f"{uploaded_file.name}"
                    )
                    st.markdown(
                        "**File Size**\n"
                        f"{format_file_size(uploaded_file.size)}"
                    )
                    st.markdown(
                        "**Upload Time**\n"
                        f"{uploaded_time}"
                    )
                with summary_cols[1]:
                    st.markdown(
                        "**Total Pages**\n"
                        f"{document_info['total_pages']}"
                    )
                    st.markdown(
                        "**Total Chunks**\n"
                        f"{statistics['total_chunks']}"
                    )
                    st.markdown(
                        "**Average Chunk Size**\n"
                        f"{statistics['average_chunk_size']}"
                    )
                    st.markdown("**Processing Status**\nReady")

            st.divider()
            st.markdown("## 📄 Document Information")
            metric_columns = st.columns(4)
            with metric_columns[0]:
                st.metric("📄 Document", uploaded_file.name)
            with metric_columns[1]:
                st.metric("📚 Pages", document_info["total_pages"])
            with metric_columns[2]:
                st.metric("🧩 Chunks", statistics["total_chunks"])
            with metric_columns[3]:
                st.metric("📏 Avg Chunk Size", statistics["average_chunk_size"])

            st.divider()
            st.markdown("## 🔍 Metadata Preview")
            st.caption("This metadata will later be used to generate page citations.")
            with st.expander("View metadata", expanded=False):
                st.json(first_chunk.metadata)

            st.divider()
            st.markdown("## 📄 First Chunk Preview")
            st.caption("The first chunk is shown below for inspection and validation.")
            with st.container():
                preview_text = format_preview_text(first_chunk.page_content)
                st.code(preview_text, language="text")

            st.divider()
            chunks_path = Path("temp/chunks.pkl")
            if chunks_path.exists():
                with chunks_path.open("rb") as handle:
                    st.download_button(
                        label="📥 Download Processed Chunks",
                        data=handle,
                        file_name="chunks.pkl",
                        mime="application/octet-stream",
                    )

            st.divider()
            st.markdown("## 💬 Chat with the Document")
            st.caption("Ask questions about the uploaded PDF using the indexed chunks and the built-in RAG flow.")

            for message in st.session_state.chat_messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

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

                        system_prompt = get_system_prompt("Beginner") or "You are an academic assistant helping a student understand the uploaded document."
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
                        if citations:
                            st.caption(f"Sources: Pages {', '.join(map(str, citations))}")

                        st.session_state.chat_history = add_message(st.session_state.chat_history, "human", prompt)
                        st.session_state.chat_history = add_message(st.session_state.chat_history, "ai", full_response)
                        st.session_state.chat_messages.append({"role": "assistant", "content": full_response})
                        with st.chat_message("assistant"):
                            st.markdown(full_response)
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

            st.divider()
            st.markdown(
                "<div style='text-align: center; padding: 1.2rem 0 0.5rem 0; color: #6b7280;'>"
                "<h4 style='margin-bottom: 0.2rem;'>EduFocus AI</h4>"
                "<p style='margin: 0.1rem 0;'>Capstone Project</p>"
                "<p style='margin: 0.25rem 0 0;'>Member 1: Document Ingestion & Metadata Pipeline</p>"
                "<p style='margin-top: 0.5rem;'>Built with<br>• Streamlit<br>• LangChain<br>• PyPDFLoader</p>"
                "</div>",
                unsafe_allow_html=True,
            )
    else:
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
