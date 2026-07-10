# EduFocus AI

EduFocus AI is a Streamlit-based Python project for preparing academic PDF documents into clean LangChain chunks with preserved metadata.

## Project Structure

- app.py: Main Streamlit application
- utils/loader.py: PDF loading logic using PyPDFLoader
- utils/splitter.py: Chunking logic using RecursiveCharacterTextSplitter
- temp/: Folder for temporary processing artifacts

## Installation

```bash
pip install -r requirements.txt
```

## Run the App

```bash
streamlit run app.py
```
