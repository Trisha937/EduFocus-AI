# 🎓 EduFocus: Adaptive Academic RAG Assistant

**EduFocus** is an intelligent, grounded Retrieval-Augmented Generation (RAG) educational platform that converts static academic materials (PDFs) into an interactive, personalized learning environment. 

Designed to adapt dynamically to a student's technical proficiency, EduFocus AI provides level-tailored explanations, verifiable page-level citations, zero-hallucination guardrails, and automated self-assessment quizzes.

---

## ✨ Key Features

- 🧠 **Adaptive Learning Modes**
  - **Beginner Mode:** Explains complex concepts using simple language, relatable analogies, and intuitive examples.
  - **Intermediate Mode:** Provides technically accurate, academically rigorous explanations for deeper understanding.

- 📚 **Grounded Page-Level Citations**
  - Every response includes verifiable citations in the format:
    ```
    [Source: document.pdf, Page X]
    ```
  - Enables students to validate answers directly from the uploaded study material.

- 🛡️ **Hallucination-Resistant Responses**
  - Strict prompt engineering ensures answers are generated only from the retrieved document context.
  - If the required information is unavailable, the assistant clearly states that it is not present in the uploaded material.

- ⚡ **Real-Time AI Streaming**
  - Powered by **Groq's `llama-3.1-8b-instant`** for low-latency, token-by-token response generation using Streamlit streaming.

- 📝 **Automatic Quiz Generation**
  - Uses **`llama-3.3-70b-versatile`** to generate contextual **5-question MCQ quizzes** directly from the uploaded academic content.

- 🔄 **Context-Aware Chat Memory**
  - Maintains a rolling **3-turn conversation history (6 messages)** to preserve context while keeping token usage efficient.

---

## 🛠️ System Architecture & Tech Stack

EduFocus is engineered entirely with high-performance, **100% free and open-source** technologies:

| Layer | Component | Technology | Role & Architecture |
| :---: | :--- | :--- | :--- |
| **Frontend** | User Interface | [![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=Streamlit&logoColor=white)](https://streamlit.io/) | Pure Python UI with real-time `st.write_stream` token rendering and tabbed dashboards |
| **Framework** | Orchestration | [![LangChain](https://img.shields.io/badge/LangChain-🦜🔗-black?style=flat-square)](https://www.langchain.com/) | Document chunking pipelines, prompt management, and rolling chat memory buffers |
| **Embeddings** | Local Vector Math | `sentence-transformers` | `all-MiniLM-L6-v2` running CPU-locally for zero-cost, high-velocity semantic vector generation |
| **Vector DB** | Local Storage | [![FAISS](https://img.shields.io/badge/FAISS-CPU-green?style=flat-square)](https://github.com/facebookresearch/faiss) | Binary vector index stored locally for instant similarity search without cloud DB overhead |
| **Inference** | LLM Cloud API | [![Groq](https://img.shields.io/badge/Groq-Cloud-orange?style=flat-square)](https://groq.com/) | • `llama-3.1-8b-instant`: Ultra-fast streamed chat (<2.5s latency)<br>• `llama-3.3-70b-versatile`: Complex quiz generation |

```text
┌─────────────────┐       ┌────────────────────┐       ┌──────────────────────┐
│   PDF Uploads   │ ────► │ PyPDF + Chunking   │ ────► │ sentence-transformer │
└─────────────────┘       └────────────────────┘       └──────────┬───────────┘
                                                                  │
┌─────────────────┐       ┌────────────────────┐                  ▼
│ Streamlit UI    │ ◄──── │ Groq Cloud Stream  │ ◄──── ┌──────────────────────┐
│  (Chat + Quiz)  │       │ (Llama-3.1 / 70B)  │       │  Local FAISS Index   │
└─────────────────┘       └────────────────────┘       └──────────────────────┘
```

## 📁 Repository Structure

```text
EduFocus/
├── 📄 app.py                      # Main Streamlit UI entry point
├── 📂 utils/
│   ├── ⚙️ loader.py               # Document loading via PyPDFLoader
│   ├── ✂️ splitter.py             # Text chunking (1000 chars, 200 overlap)
│   ├── 🧮 embeddings.py           # CPU sentence-transformer model
│   ├── 🗄️ vector_store.py         # FAISS similarity search utilities
│   ├── 🧠 chat_memory.py          # 3-turn sliding window history logic
│   ├── 🚀 chat_engine.py          # Groq streaming & citation extraction
│   └── 📝 prompts.py             # Adaptive system prompts & quiz formatters
├── 📂 database/
│   ├── 📥 ingestion.py            # Document processing pipeline
│   └── 💾 vector_store.py         # Persistence wrappers
├── 📂 faiss_db/                   # Local vector index storage
├── 📄 requirements.txt            # Python dependencies
└── 🔒 .env                        # Environment variables (API key)
```

## 🚀 Quick Start

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Trisha937/EduFocus.git
cd EduFocus
```

---

### 2️⃣ Set Up a Virtual Environment

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux**

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Configure Environment Variables

Create a `.env` file in the project root and add your Groq API key:

```env
GROQ_API_KEY=your_groq_api_key_here
```

---

### 5️⃣ Launch the Application

```bash
streamlit run app.py
```

---

🌐 **Application URL:** http://localhost:8501

Open the above URL in your browser after the Streamlit server starts.