import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models # type: ignore
from openai import OpenAI # type: ignore

# --- Configuration ---
load_dotenv()
qdrant_host = os.getenv("QDRANT_HOST", "localhost")
qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
collection_name = "cosmo_challenge" # MUST match the collection name used in vector_store.py
embedding_model = "text-embedding-3-small" # MUST match the model used for initial embeddings

# --- Initialize OpenAI Client ---
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
   print("Error: OPENAI_API_KEY not found in environment variables.")
   exit()
openai_client = OpenAI()

# --- Initialize Qdrant Client ---
try:
    qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
    qdrant_client.get_collection(collection_name=collection_name)
    print(f"Connected to Qdrant and collection '{collection_name}' found.")
except Exception as e:
    print(f"Error connecting to Qdrant or finding collection '{collection_name}': {e}")
    print("Please ensure Qdrant is running and vector_store.py was executed successfully.")
    exit()


# --- Retrieval Function ---
# Updated to access results via response.points
def retrieve_relevant_chunks(query_text: str, top_k: int = 3) -> list[dict]:
    """
    Generates embedding for the query and retrieves top_k relevant chunks from Qdrant.

    Args:
        query_text: The user's query.
        top_k: The number of chunks to retrieve.

    Returns:
        A list of dictionaries, each containing 'text' and 'score',
        or an empty list if an error occurs.
    """
    if not query_text:
        print("Error: Query text cannot be empty.")
        return []

    print(f"\nRetrieving top {top_k} relevant chunks for query: '{query_text[:50]}...'")
    try:
        # 1. Generate query embedding
        response = openai_client.embeddings.create(
            input=query_text,
            model=embedding_model
        )
        query_embedding = response.data[0].embedding

        # 2. Search Qdrant using query_points
        # query_points returns a QueryResponse object
        query_response = qdrant_client.query_points(
            collection_name=collection_name,
            query=query_embedding,
            limit=top_k,
            with_payload=True
        )
        # Access the list of points from the response object
        search_points = query_response.points
        print(f"Found {len(search_points)} results from Qdrant.") # <-- Use len() on the .points attribute

        # 3. Extract text and score from results
        relevant_chunks_with_scores = [
            {"text": hit.payload['text'], "score": hit.score}
            # Iterate over search_points (the list within the response)
            for hit in search_points if hit.payload and 'text' in hit.payload
        ]
        return relevant_chunks_with_scores

    except Exception as e:
        print(f"Error during retrieval: {e}")
        return []

# --- Example Usage ---
if __name__ == "__main__":
    print("\n--- Retrieval Test Script ---")

    queries = [
        "How many samples are there in DeltaBench?",
        "What is the most common error type in Math problems according to DeltaBench?",
        "What is the HitRate@1 for Qwen2.5-Math-PRM-7B?",
        "What is the percentage reduction in self-critique performance for DeepSeek-R1 compared to cross-model critique?"
    ]

    ans = [
        "1236",
        "Reasoning Error (25.3%)",
        "49.15%",
        "36%"
    ]

    for query in queries:
        # The function now returns a list of dictionaries
        retrieved_results = retrieve_relevant_chunks(query_text=query, top_k=3)

        if retrieved_results:
            print(f"\n--- Retrieved Chunks for Query: '{query}' ---")
            # Iterate through the list of dictionaries
            for i, result_item in enumerate(retrieved_results):
                # Access 'score' and 'text' from each dictionary
                print(f"\nChunk {i+1} (Score: {result_item['score']:.4f}):")
                print(result_item['text'][:400] + "...") # Print the first 400 chars
            print("\n------------------------------------------")
        else:
            print(f"\nNo relevant chunks were retrieved for the query: '{query}'")