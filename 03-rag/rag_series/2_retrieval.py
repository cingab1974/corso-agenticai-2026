
# rag_series/2_retrieval.py

import chromadb

# --- 1. Configuration ---
# CHROMA_PERSIST_DIRECTORY = "rag_series/chroma_db"
# CHROMA_COLLECTION_NAME = "rag_collection"
CHROMA_PERSIST_DIRECTORY = "rag_series/chroma_db_docling"
CHROMA_COLLECTION_NAME = "rag_collection_docling_hf_chunker"

# --- 2. Querying ---
def query_collection(collection, query_text, n_results=3):
    """Queries the collection for relevant chunks."""
    if not query_text:
        print("Error: No query text provided.")
        return None
    
    try:
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        return results
    except Exception as e:
        print(f"Error querying collection: {e}")
        return None

# --- Main Execution ---
if __name__ == "__main__":
    print("--- Starting Retrieval for RAG ---")

    # Connect to the persistent ChromaDB client
    try:
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)
        collection = client.get_collection(name=CHROMA_COLLECTION_NAME)
        print(f"Successfully connected to collection '{CHROMA_COLLECTION_NAME}'.")
        
        # --- Get user query ---
        user_query = input("Please enter your search query: ")
        
        print(f"\nQuerying for: '{user_query}'")
        
        # Retrieve relevant chunks
        retrieved_chunks = query_collection(collection, user_query)
        
        if retrieved_chunks:
            print("\n--- Retrieved Chunks ---")
            documents = retrieved_chunks.get('documents', [[]])[0]
            if documents:
                for i, doc in enumerate(documents):
                    print(f"Chunk {i+1}:")
                    print(doc)
                    print("-" * 25)
            else:
                print("No relevant chunks found for your query.")
                
    except Exception as e:
        print(f"\nAn error occurred. Did you run '1_data_ingestion.py' first?")
        print(f"Error details: {e}")

    print("\n--- Retrieval Complete ---")
