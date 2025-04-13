import os
import uuid
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
import fickling

# --- Configuration ---
load_dotenv()
qdrant_host = os.getenv("QDRANT_HOST", "localhost")
qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
pickle_file_path = "pdf_embeddings_cache.pkl" # Input file
collection_name = "cosmo_challenge" # Name for the Qdrant collection

print("--- Vector Store Setup Script ---")

# --- Load Cached Data ---
def load_data_from_pickle(file_path):
    """Loads processed data (text, embedding) from a pickle file."""
    if os.path.exists(file_path):
        print(f"\nLoading cached data from: {file_path}")
        try:
            with open(file_path, 'rb') as f:
                data = fickling.load(f)
            print(f"Successfully loaded {len(data)} items from cache.")
            if isinstance(data, list) and len(data) > 0 and 'text' in data[0] and 'embedding' in data[0]:
                 first_embedding_dim = len(data[0]['embedding'])
                 print(f"Detected embedding dimension: {first_embedding_dim}")
                 return data, first_embedding_dim
            else:
                 print("Error: Loaded data is not in the expected format.")
                 return None, 0
        except Exception as e:
            print(f"Error loading data from pickle file: {e}")
            return None, 0
    else:
        print(f"Error: Cache file not found at {file_path}. Cannot proceed.")
        return None, 0

# --- Main Execution Logic ---
if __name__ == "__main__":
    processed_data, vector_dim = load_data_from_pickle(pickle_file_path)

    if not processed_data:
        print("Exiting script because data could not be loaded.")
        exit()

    # --- Initialize Qdrant Client ---
    print(f"\nConnecting to Qdrant at {qdrant_host}:{qdrant_port}...")
    try:
        client = QdrantClient(host=qdrant_host, port=qdrant_port)
        # Optional: Check connection/server info
        # print(client.get_collections()) # Example check
        print("Successfully connected to Qdrant.")
    except Exception as e:
        print(f"Error connecting to Qdrant: {e}")
        print("Please ensure Qdrant is running and accessible.")
        exit()

    # --- Create Qdrant Collection ---
    try:
        print(f"\nAttempting to create or recreate collection: '{collection_name}'")
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_dim,
                distance=models.Distance.COSINE
            )
        )
        print(f"Collection '{collection_name}' created successfully.")
    except Exception as e:
        print(f"Error creating Qdrant collection: {e}")
        # Attempt to proceed if collection might already exist correctly
        try:
             collection_info = client.get_collection(collection_name=collection_name)
             if collection_info.vectors_config.params.size != vector_dim:
                  print(f"Error: Existing collection '{collection_name}' has dimension {collection_info.vectors_config.params.size}, but data has dimension {vector_dim}.")
                  exit()
             print(f"Collection '{collection_name}' already exists with correct dimension. Will upsert data.")
        except Exception as e2:
             print(f"Could not verify existing collection: {e2}")
             exit()


    # --- Prepare and Upsert Data into Qdrant ---
    points_to_upsert = []
    for i, item in enumerate(processed_data):
         # Skip if embedding is somehow invalid (e.g., None from previous step)
         if item.get('embedding') is None or len(item['embedding']) != vector_dim:
              print(f"Warning: Skipping item {i} due to invalid embedding.")
              continue

         points_to_upsert.append(
            models.PointStruct(
                id=str(uuid.uuid4()), # Unique ID for each point
                vector=item['embedding'],
                payload={"text": item['text']} # Store text chunk
            )
        )

    if not points_to_upsert:
        print("\nError: No valid points prepared for upsertion.")
        exit()

    print(f"\nPreparing to upsert {len(points_to_upsert)} points into '{collection_name}'...")

    # Upsert in batches
    batch_size = 100
    for i in range(0, len(points_to_upsert), batch_size):
        batch = points_to_upsert[i:i + batch_size]
        try:
            client.upsert(
                collection_name=collection_name,
                points=batch,
                wait=True # Wait for acknowledgment
            )
            print(f"Upserted batch {i//batch_size + 1}/{(len(points_to_upsert) + batch_size - 1)//batch_size}")
        except Exception as e:
            print(f"Error upserting batch starting at index {i}: {e}")
            # Decide if you want to stop or continue on batch errors

    print("\n--- Vector Store Setup Complete ---")
    # You can verify the count in Qdrant
    try:
        count = client.count(collection_name=collection_name, exact=True)
        print(f"Collection '{collection_name}' now contains {count.count} points.")
    except Exception as e:
        print(f"Could not verify point count: {e}")
