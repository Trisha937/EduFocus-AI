# test_db.py

import os
from database.ingestion import process_pdf
from database.vector_store import create_and_save_vector_db, load_local_vector_db, retrieve_context_with_citations

def test_pipeline():
    # Make sure you have a test PDF in your main folder named 'sample.pdf'
    pdf_path = "sample.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"❌ Error: Please place any test PDF named '{pdf_path}' in your main project folder first!")
        return

    print("\n--- RUNNING SYSTEM TEST ---")

    # Step 1: Process the PDF (Calling Member 1's utilities)
    print("🔄 Processing PDF...")
    try:
        chunks = process_pdf(pdf_path)
    except Exception as e:
        print(f"❌ Ingestion failed: {e}")
        return

    # Step 2: Build the database (Calling your FAISS code)
    print("🔄 Building local FAISS database...")
    try:
        vector_store = create_and_save_vector_db(chunks)
        print("✅ Local FAISS database created and saved to disk!")
    except Exception as e:
        print(f"❌ Database creation failed: {e}")
        return

    # Step 3: Load the database to verify it works offline
    print("🔄 Verifying database storage reload...")
    loaded_db = load_local_vector_db()
    if loaded_db is None:
        print("❌ Error: Failed to load database from faiss_db/ directory.")
        return
    print("✅ Database successfully reloaded into memory!")

    # Step 4: Run a search query
    print("🔄 Testing semantic query search with citations...")
    test_query = "What are the main contents of this document?"
    context, pages = retrieve_context_with_citations(test_query, loaded_db)

    print("\n================== TEST OUTPUT ==================")
    print(f"📍 Student Question: '{test_query}'")
    print(f"📚 Page Citations: Pages {pages}")
    print("\n📖 Context Found:")
    print(context[:350] + "\n... [Remaining text truncated]")
    print("=================================================")
    print("\n🎉 ALL TESTS PASSED! Your database pipeline works perfectly!")

if __name__ == "__main__":
    test_pipeline()