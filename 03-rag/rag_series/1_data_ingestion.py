
# rag_series/1_data_ingestion.py

import chromadb
import os
import tiktoken
from pypdf import PdfReader

# --- 1. Configuration ---
SOURCE_DIRECTORY = "rag_series/source_docs"
DOCUMENTS_PATTERN = ".pdf"
CHROMA_PERSIST_DIRECTORY = "rag_series/chroma_db"
CHROMA_COLLECTION_NAME = "rag_collection"
CHUNK_SIZE = 256
CHUNK_OVERLAP = 32

# --- 2. Document Loading ---
def load_pdf_documents(directory):
    """Loads and extracts text from PDF documents in a directory."""
    documents_text = []
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' not found.")
        print("Please create the 'source_docs' directory and add your PDF files.")
        return documents_text

    for filename in os.listdir(directory):
        if filename.endswith(DOCUMENTS_PATTERN):
            filepath = os.path.join(directory, filename)
            try:
                reader = PdfReader(filepath)
                text = ""
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text
                documents_text.append(text)
                print(f"Successfully loaded '{filename}'.")
            except Exception as e:
                print(f"Error loading '{filename}': {e}")
    return documents_text

# --- 3. Text Chunking ---
def chunk_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Chunks text using a tokenizer."""
    if not text:
        return []
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    chunks = []
    for i in range(0, len(tokens), chunk_size - chunk_overlap):
        chunk_tokens = tokens[i:i + chunk_size]
        chunks.append(tokenizer.decode(chunk_tokens))
    return chunks

# --- 4. Database Storage ---
def store_chunks_in_db(chunks, collection_name):
    """Stores chunks in a ChromaDB collection."""
    if not chunks:
        print("No chunks to store.")
        return
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)
    collection = client.get_or_create_collection(name=collection_name)

    ids = [f"chunk_{i}" for i, _ in enumerate(chunks)]
    collection.add(
        documents=chunks,
        ids=ids
    )
    print(f"Stored {len(chunks)} chunks in ChromaDB collection '{collection_name}'.")

# --- Main Execution ---
if __name__ == "__main__":
    print("--- Starting Data Ingestion for RAG ---")
    texts = load_pdf_documents(SOURCE_DIRECTORY)

    if not texts:
        print(f"No documents loaded. Please check the '{SOURCE_DIRECTORY}' directory.")
    else:
        for i, text in enumerate(texts):
            print(f"\n--- Processing document {i+1} ---")
            print("Chunking text...")
            chunks = chunk_text(text)
            print(f"Created {len(chunks)} chunks.")

            print("Storing chunks in ChromaDB...")
            store_chunks_in_db(chunks, CHROMA_COLLECTION_NAME)

    print("\n--- Data Ingestion Complete ---")
