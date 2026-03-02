
# rag_series/3_generation.py

import chromadb
import litellm

# --- 1. Configuration ---
# CHROMA_PERSIST_DIRECTORY = "rag_series/chroma_db"
# CHROMA_COLLECTION_NAME = "rag_collection"
CHROMA_PERSIST_DIRECTORY = "rag_series/chroma_db_docling"
CHROMA_COLLECTION_NAME = "rag_collection_docling_hf_chunker"

# --- LiteLLM Configuration ---
litellm.api_base = "https://litellm-proxy-1013932759942.europe-west8.run.app"
litellm.api_key = ""
LLM_MODEL = "litellm_proxy/gemini-2.5-pro"
# LLM_MODEL = "litellm_proxy/vertex_ai/mistral-small-2503"


# --- 2. Retrieval ---
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
        return results.get('documents', [[]])[0]
    except Exception as e:
        print(f"Error querying collection: {e}")
        return None


# --- 3. Prompt Formatting ---
def format_prompt(query, context_chunks):
    """Formats the prompt with the user query and retrieved context."""
    if not context_chunks:
        return f"Question: {query}\nAnswer:"

    context = "\n\n---\n\n".join(context_chunks)
    prompt = f"""
    Answer the question based only on the following context:

    {context}

    ---

    Question: {query}
    Answer:
    """
    return prompt


# --- 4. Generation with LiteLLM ---
def generate_answer(prompt):
    """Sends the prompt to the configured LLM using LiteLLM."""
    if litellm.api_base == "YOUR_LITELLM_API_BASE":
        print("\n--- !!! CONFIGURATION REQUIRED !!! ---")
        print("Please update the 'litellm.api_base' and 'LLM_MODEL' variables in this script.")
        return "Configuration needed."

    try:
        print("\n--- Sending prompt to LLM via LiteLLM ---")
        response = litellm.completion(
            model=LLM_MODEL,
            messages=[{"content": prompt, "role": "user"}]
        )
        # Correctly access the message content from the response
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"\n--- LiteLLM Error ---")
        print(f"An error occurred while communicating with the LLM: {e}")
        print("Please check your LiteLLM server connection and configuration.")
        return None


# --- Main Execution ---
if __name__ == "__main__":
    print("--- Starting Generation with LiteLLM for RAG ---")

    try:
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)
        collection = client.get_collection(name=CHROMA_COLLECTION_NAME)
        print(f"Successfully connected to collection '{CHROMA_COLLECTION_NAME}'.")

        user_query = input("Please enter your search query: ")
        print(f"\nUser Query: '{user_query}'")

        print("Retrieving relevant context...")
        retrieved_chunks = query_collection(collection, user_query)

        if retrieved_chunks:
            print(f"Retrieved {len(retrieved_chunks)} chunks.")
            print(f"Retrieved chunks: {retrieved_chunks}")

            final_prompt = format_prompt(user_query, retrieved_chunks)
            
            # Generate a real answer using LiteLLM
            final_answer = generate_answer(final_prompt)
            
            if final_answer:
                print("\n--- Final Answer from LLM ---")
                print(final_answer)
        else:
            print("Could not retrieve relevant context.")

    except Exception as e:
        print(f"\nAn error occurred. Did you run '1_data_ingestion.py' first?")
        print(f"Error details: {e}")

    print("\n--- Generation Complete ---")
