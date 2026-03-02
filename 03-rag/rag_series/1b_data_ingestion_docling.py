import chromadb
import os
import pickle
import random
from docling.document_converter import (
    DocumentConverter,
    InputFormat,
    PdfFormatOption,
    StandardPdfPipeline,
)
from docling_core.types.doc.document import DoclingDocument
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer

# --- 1. Configuration ---
SOURCE_DIRECTORY = "rag_series/source_docs"
PROCESSED_DIRECTORY = "rag_series/processed_docs"
DOCUMENTS_PATTERN = ".pdf"
CHROMA_PERSIST_DIRECTORY = "rag_series/chroma_db_docling"
CHROMA_COLLECTION_NAME = "rag_collection_docling_hf_chunker"
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


# --- 2. Document Parsing with Docling ---
def parse_and_save_doctags(directory):
    """Parses PDFs using docling and saves the DoclingDocument object."""
    if not os.path.exists(directory):
        print(f"Error: Source directory '{directory}' not found.")
        return

    pdf_options = PdfFormatOption(pipeline_cls=StandardPdfPipeline)
    converter = DocumentConverter(format_options={InputFormat.PDF: pdf_options})

    for filename in os.listdir(directory):
        if filename.endswith(DOCUMENTS_PATTERN):
            filepath = os.path.join(directory, filename)
            processed_filepath = os.path.join(PROCESSED_DIRECTORY, f"{filename}.pkl")

            if os.path.exists(processed_filepath):
                print(f"'{filename}' already processed. Skipping parsing.")
                continue

            try:
                print(f"Parsing '{filename}' with docling...")
                docling_document = converter.convert(filepath).document

                with open(processed_filepath, "wb") as f:
                    pickle.dump(docling_document, f)
                print(
                    f"Saved DoclingDocument for '{filename}' to '{processed_filepath}'."
                )

            except Exception as e:
                print(f"Error parsing '{filename}': {e}")


# --- 3. Text Extraction and Chunking ---
def load_and_chunk_with_docling(directory):
    """Loads DoclingDocument objects, chunks them, and returns text chunks with metadata."""
    all_chunks = []
    if not os.path.exists(directory):
        print(f"Error: Processed directory '{directory}' not found.")
        return all_chunks

    hf_tokenizer = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(EMBED_MODEL_ID)
    )
    chunker = HybridChunker(contextualize=True, tokenizer=hf_tokenizer)

    for filename in os.listdir(directory):
        if filename.endswith(".pkl"):
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, "rb") as f:
                    docling_doc = pickle.load(f)

                print(f"Chunking '{filename}' ...")

                docling_chunks = list(chunker.chunk(docling_doc))

                for chunk in docling_chunks:
                    page_no = None
                    section = None

                    if chunk.meta and chunk.meta.doc_items:
                        if chunk.meta.doc_items[0].prov:
                            page_no = chunk.meta.doc_items[0].prov[0].page_no

                    if hasattr(chunk.meta, "headings") and chunk.meta.headings:
                        section = " > ".join(chunk.meta.headings)

                    all_chunks.append(
                        {
                            "text": chunk.text,
                            "metadata": {
                                "filename": os.path.basename(filename),
                                "page_no": page_no if page_no is not None else -1,
                                "section": section if section is not None else "",
                            },
                        }
                    )

                print(f"Created {len(docling_chunks)} chunks from '{filename}'.")

            except Exception as e:
                print(f"Error loading or chunking '{filename}': {e}")
    return all_chunks


# --- 4. Database Storage ---
def store_chunks_in_db(chunks, collection_name):
    """Stores chunks in a ChromaDB collection."""
    if not chunks:
        print("No chunks to store.")
        return

    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)
    collection = client.get_or_create_collection(name=collection_name)

    # Prepare documents, metadatas, and ids for ChromaDB
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]
    ids = [f"chunk_{i}" for i, _ in enumerate(chunks)]

    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    print(f"\nStored {len(chunks)} chunks in ChromaDB collection '{collection_name}'.")


# --- Main Execution ---
if __name__ == "__main__":
    print("--- Starting Advanced Data Ingestion with Docling ---")

    # Step 1: Parse PDFs and save DoclingDocument
    parse_and_save_doctags(SOURCE_DIRECTORY)

    # Step 2: Load and chunk the documents with docling
    print("\n--- Loading and Chunking with docling---")
    all_document_chunks = load_and_chunk_with_docling(PROCESSED_DIRECTORY)

    # Step 3: Store the final chunks in ChromaDB
    if all_document_chunks:
        store_chunks_in_db(all_document_chunks, CHROMA_COLLECTION_NAME)
        # Print 5 random chunks to inspect the output
        print("\n--- Sample of 5 Random Chunks ---")
        for i, chunk in enumerate(random.sample(all_document_chunks, 5)):
            print(
                f"[CHUNK {i + 1}]\n{chunk['text']}\nMETADATA: {chunk['metadata']}\n{'-' * 20}"
            )
    else:
        print("No chunks were created. Halting process.")

    print("\n--- Advanced Data Ingestion Complete ---")
