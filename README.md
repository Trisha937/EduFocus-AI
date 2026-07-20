# 🎓 EduFocus AI: Adaptive Academic RAG Assistant

**EduFocus AI** is an intelligent, grounded Retrieval-Augmented Generation (RAG) educational platform that converts static academic materials (PDFs) into an interactive, personalized learning environment. 

Designed to adapt dynamically to a student's technical proficiency, EduFocus AI provides level-tailored explanations, verifiable page-level citations, zero-hallucination guardrails, and automated self-assessment quizzes.

---

## ✨ Key Features

1️⃣🧠 **Adaptive Learning Modes:**
  * **Beginner Tier:** Simplifies complex concepts using relatable analogies, intuitive breakdowns, and accessible language.
  * **Intermediate Tier:** Delivers precise, technical, and academically rigorous explanations suited for advanced study.
  
2️⃣📚 **Verifiable Source Citations:** Every response extracts and displays exact page numbers directly from the uploaded PDF document (`[Source: file.pdf, Page X]`).


3️⃣🛡️ **Zero-Hallucination Guardrails:** Enforces strict system-prompt boundaries to ensure answers are strictly derived from the provided document context.

4️⃣⚡ **High-Speed Real-Time Streaming:** Powered by Groq's high-velocity inference engine (`llama-3.1-8b-instant`) to stream responses token-by-token with sub-2.5-second latency.

5️⃣📝 **Automated Evaluation Quizzes:** Leverages `llama-3.3-70b-versatile` to dynamically compile 5-question multiple-choice quizzes directly from uploaded textbook content.


6️⃣🔄 **Smart Conversational Memory:** Implements a rolling 3-turn history window (last 6 messages) to maintain contextual coherence without exceeding token windows.

---

## 🛠️ System Architecture & Tech Stack

EduFocus AI is engineered entirely with high-performance, **100% free and open-source** technologies:

| Layer | Component | Technology | Role & Architecture |
| :---: | :--- | :--- | :--- |
| **Frontend** | User Interface | [![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=Streamlit&logoColor=white)](https://streamlit.io/) | Pure Python UI with real-time `st.write_stream` token rendering and tabbed dashboards |
| **Framework** | Orchestration | [![LangChain](https://img.shields.io/badge/LangChain-🦜🔗-black?style=flat-square)](https://www.langchain.com/) | Document chunking pipelines, prompt management, and rolling chat memory buffers |
| **Embeddings** | Local Vector Math | `sentence-transformers` | `all-MiniLM-L6-v2` running CPU-locally for zero-cost, high-velocity semantic vector generation |
| **Vector DB** | Local Storage | [![FAISS](https://img.shields.io/badge/FAISS-CPU-green?style=flat-square)](https://github.com/facebookresearch/faiss) | Binary vector index stored locally for instant similarity search without cloud DB overhead |
| **Inference** | LLM Cloud API | [![Groq](https://img.shields.io/badge/Groq-Cloud-orange?style=flat-square)](https://groq.com/) | • `llama-3.1-8b-instant`: Ultra-fast streamed chat (<2.5s latency)<br>• `llama-3.3-70b-versatile`: Complex quiz generation |



 ┌────────────────┐       ┌───────────────────┐       ┌──────────────────────┐
 │  PDF Uploads   │ ────► │  PyPDF + Chunking │ ────► │ sentence-transformer │
 └────────────────┘       └───────────────────┘       └──────────┬───────────┘
                                                                 │
 ┌────────────────┐       ┌───────────────────┐                  ▼
 │ Streamlit UI   │ ◄──── │ Groq Cloud Stream │ ◄────  ┌─────────────────────┐
 │ (Chat + Quiz)  │       │ (Llama-3.1 / 70B) │        │  Local FAISS Index  │
 └────────────────┘       └───────────────────┘        └─────────────────────┘


## 📁 Repository Structure

EduFocus-AI/
 ├── 📄 app.py                      # Main Streamlit UI entry point
 ├── 📂 utils/
 │    ├── ⚙️ loader.py               # Document loading via PyPDFLoader
 │    ├── ✂️ splitter.py             # Text chunking (1000 chars, 200 overlap)
 │    ├── 🧮 embeddings.py           # CPU sentence-transformer model
 │    ├── 🗄️ vector_store.py         # FAISS similarity search utilities
 │    ├── 🧠 chat_memory.py          # 3-turn sliding window history logic
 │    ├── 🚀 chat_engine.py          # Groq streaming & citation extraction
 │    └── 📝 prompts.py             # Adaptive system prompts & quiz formatters
 ├── 📂 database/
 │    ├── 📥 ingestion.py            # Document processing pipeline
 │    └── 💾 vector_store.py         # Persistence wrappers
 ├── 📂 faiss_db/                   # Local vector index storage
 ├── 📄 requirements.txt            # Python dependencies
 └── 🔒 .env                        # Environment variables (API key)



## 🚀 Quick Start Guide

1️⃣ Clone & Navigate
git clone [https://github.com/Trisha937/EduFocus-AI.git](https://github.com/Trisha937/EduFocus-AI.git)
cd EduFocus-AI

2️⃣ Set Up Environment
# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Configure Groq API Key
Get your free key from the Groq Console. Create a .env file in the root folder:

Code snippet
GROQ_API_KEY=your_groq_api_key_here

5️⃣ Launch the Application
streamlit run app.py

🌐 Local Access: Open your browser to http://localhost:8501 to start learning!