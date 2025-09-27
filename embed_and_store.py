import json
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import config

# --- Helper function to clean metadata ---
def sanitize_metadata(metadata):
    return {k: (v if v is not None else "") for k, v in metadata.items()}

def main():
    """
    Main function to read chunks, generate embeddings, and store them in ChromaDB
    using settings from the config.py file.
    """
    # --- 1. Load the Embedding Model from config ---
    print(f"Loading embedding model: {config.EMBEDDING_MODEL}...")
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    print("Model loaded successfully.")

    # --- 2. Initialize ChromaDB using paths from config ---
    client = chromadb.PersistentClient(path=config.DB_PATH)
    collection = client.get_or_create_collection(name=config.COLLECTION_NAME)
    print(f"ChromaDB collection '{config.COLLECTION_NAME}' ready.")

    # --- 3. Read the Data and Process in Batches ---
    try:
        with open(config.CHUNKED_DATA_PATH, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
            # Use BATCH_SIZE from config
            for i in tqdm(range(0, len(lines), config.BATCH_SIZE), desc="Processing batches"):
                batch_lines = lines[i:i+config.BATCH_SIZE]
                batch_chunks = [json.loads(line) for line in batch_lines]

                ids = [chunk['chunk_id'] for chunk in batch_chunks]
                documents = [chunk['chunk_text'] for chunk in batch_chunks]
                metadatas = [sanitize_metadata(chunk['metadata']) for chunk in batch_chunks]

                # --- 4. Generate Embeddings for the Batch ---
                embeddings = model.encode(documents, show_progress_bar=False).tolist()

                # --- 5. Add the Batch to ChromaDB ---
                collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas
                )

        print("\n" + "="*50)
        print("🎉 Indexing complete!")
        print(f"Total chunks added to the collection: {collection.count()}")
        print(f"Database is stored at: {config.DB_PATH}")
        print("="*50)

    except FileNotFoundError:
        print(f"Error: Input file not found at {config.CHUNKED_DATA_PATH}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()