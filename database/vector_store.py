# database/vector_store.py


import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# This dynamically finds or creates the 'faiss_db' folder at the root of your project
DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "faiss_db")

def get_embedding_model():
    """
    Initializes and returns the local HuggingFace embedding model.
    Runs entirely on your CPU for free.
    """
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def create_and_save_vector_db(chunks):
    """
    Takes text chunks from Member 1, converts them to numbers, 
    and saves them in the 'faiss_db' folder.
    """
    embeddings = get_embedding_model()
    
    # Create the vector database using the chunks and the model
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    # Save the database locally
    vector_store.save_local(DB_DIR)
    
    return vector_store


def load_local_vector_db():
    """
    Loads your saved database from the 'faiss_db' folder back into memory.
    """
    embeddings = get_embedding_model()
    
    # Check if the folder actually exists first
    if not os.path.exists(DB_DIR):
        return None
        
    try:
        vector_store = FAISS.load_local(
            DB_DIR, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        return vector_store
    except Exception as e:
        print(f"Error loading FAISS database: {e}")
        return None


def retrieve_context_with_citations(query, vector_store):
    """
    Searches the database for the 3 most relevant paragraphs
    and extracts their page numbers.
    """
    if vector_store is None:
        return "", []
        
    # Retrieve top 3 most relevant document chunks
    matched_documents = vector_store.similarity_search(query, k=3)
    
    paragraphs = []
    page_numbers = []
    
    for doc in matched_documents:
        paragraphs.append(doc.page_content)
        
        # PDFs are 0-indexed, so we add 1 to show correct pages to students
        page_num = doc.metadata.get("page", 0) + 1
        page_numbers.append(page_num)
        
    # Combine matching texts with a clear separator
    combined_context = "\n\n---\n\n".join(paragraphs)
    
    # Remove duplicate page numbers and sort them (e.g. [2, 5] instead of [5, 2, 2])
    unique_pages = sorted(list(set(page_numbers)))
    
    return combined_context, unique_pages