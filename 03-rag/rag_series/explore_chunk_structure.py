import os
import pickle
import pprint

from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer

# --- Configuration ---
PROCESSED_DIRECTORY = "rag_series/processed_docs"
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


def inspect_chunks():
    """Loads a processed document, chunks it, and prints the structure of the first 5 chunks."""
    # 1. Find a processed document to inspect
    try:
        processed_files = [
            f for f in os.listdir(PROCESSED_DIRECTORY) if f.endswith(".pkl")
        ]
        if not processed_files:
            print(f"No processed .pkl files found in '{PROCESSED_DIRECTORY}'.")
            print("Please run the ingestion script first to generate a .pkl file.")
            return
    except FileNotFoundError:
        print(f"Error: Processed directory '{PROCESSED_DIRECTORY}' not found.")
        return

    target_file = processed_files[0]
    filepath = os.path.join(PROCESSED_DIRECTORY, target_file)
    print(f"--- Inspecting chunks from: {target_file} ---")

    # 2. Load the DoclingDocument
    try:
        with open(filepath, "rb") as f:
            docling_doc = pickle.load(f)
    except Exception as e:
        print(f"Error loading '{filepath}': {e}")
        return

    print("\n--- Inspecting Section Hierarchy in DoclingDocument ---")
    section_header_found = False
    if hasattr(docling_doc, "texts"):
        # Find a section header that is not a top-level title
        for item in docling_doc.texts:
            if (
                hasattr(item, "label")
                and item.label == "section_header"
                and item.level > 1
            ):
                print("Found a subsection header (level > 1):")
                pprint.pprint(item.__dict__)
                print("\nParent of this item:")
                pprint.pprint(item.parent)
                section_header_found = True
                break  # Just inspect the first one we find
    if not section_header_found:
        print("Could not find a subsection header to inspect.")

    # 3. Initialize Chunker
    hf_tokenizer = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(EMBED_MODEL_ID)
    )
    chunker = HybridChunker(contextualize=True, tokenizer=hf_tokenizer)

    # 4. Generate and inspect chunks
    try:
        docling_chunks = list(chunker.chunk(docling_doc))
        print(f"Generated {len(docling_chunks)} chunks. Inspecting the first 5...\n")

        for i, chunk in enumerate(docling_chunks[10:11]):
            print(f"--- CHUNK {i+1} ---")
            print("Chunk Text:", chunk.text[:150].replace("\n", " ") + "...")

            print("\n[chunk.meta.__dict__]")
            if hasattr(chunk, "meta") and chunk.meta:
                pprint.pprint(chunk.meta.__dict__)

                if hasattr(chunk.meta, "context") and chunk.meta.context:
                    print("\n[chunk.meta.context.__dict__]")
                    pprint.pprint(chunk.meta.context.__dict__)
                else:
                    print("\nNo 'context' attribute found in meta.")
            else:
                print("No 'meta' attribute found.")

            print("-" * 50 + "\n")

    except Exception as e:
        print(f"An error occurred during chunking or inspection: {e}")


if __name__ == "__main__":
    inspect_chunks()
