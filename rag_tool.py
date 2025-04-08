import os
from dotenv import load_dotenv
from openai import OpenAI
from composio_openai import ComposioToolSet
from retrieval import retrieve_relevant_chunks

# Load environment variables
load_dotenv()

# Initialize OpenAI client
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    print("Error: OPENAI_API_KEY not found.")
    exit()
openai_client = OpenAI(api_key=openai_api_key)

# Initialize Composio toolset (optional tools can be added later)
composio_api_key = os.getenv("COMPOSIO_API_KEY")
if not composio_api_key:
    print("Error: COMPOSIO_API_KEY not found.")
    exit()
toolset = ComposioToolSet(api_key=composio_api_key)

def generate_answer(query: str, top_k: int = 3) -> str:
    """
    Generates an answer to the query based solely on the retrieved context.

    Args:
        query: The user's query string.
        top_k: Number of top relevant chunks to retrieve (default is 3).

    Returns:
        A string containing the answer or an error message.
    """
    try:
        # Retrieve relevant chunks from Qdrant via retrieval.py
        retrieved_results = retrieve_relevant_chunks(query_text=query, top_k=top_k)
        if not retrieved_results:
            return "No relevant information found in the context."

        # Concatenate retrieved chunk texts into a single context string
        context_string = "\n\n".join([result["text"] for result in retrieved_results])

        # Construct the prompt as specified
        prompt = f"""
        You are an assistant designed to answer questions based ONLY on the provided context below.
        Read the context carefully.
        Answer the user's question using only the information given in the context.
        Do not use any external knowledge or make assumptions.
        If the context does not contain the information needed to answer the question, state that clearly.

        Provided Context:
        ---
        {context_string}
        ---

        User's Question: {query}

        Answer based solely on the context:
        """

        # Generate answer using OpenAI ChatGPT
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        answer = response.choices[0].message.content.strip()
        return answer

    except Exception as e:
        print(f"Error generating answer: {e}")
        return "An error occurred while generating the answer."

# Example usage
if __name__ == "__main__":
    sample_query = "How many samples are there in DeltaBench?"
    result = generate_answer(sample_query)
    print(f"Query: {sample_query}\nAnswer: {result}")