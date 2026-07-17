# database/ingestion.py


import os
from utils.loader import load_pdf
from utils.splitter import split_document, chunk_statistics

def process_pdf(file_path: str) -> list:
    """
    Processes a PDF file by loading and splitting it using Member 1's utility scripts.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at: {file_path}")

    # 1. Use Member 1's pdf loader
    documents = load_pdf(file_path)
    
    # 2. Use Member 1's text splitter
    chunks = split_document(documents)
    
    # 3. Print the file statistics calculated by Member 1's utilities
    stats = chunk_statistics(chunks)
    print(f"✅ Ingestion Summary -> Total Chunks: {stats['total_chunks']} | Avg Chunk Size: {stats['average_chunk_size']} chars.")
    
    return chunks