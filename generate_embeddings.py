import pypdf
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
import time
import pickle


# Load environment variables (if you store the PDF path there, otherwise adjust)
load_dotenv()

# --- Configuration ---
pdf_path = "detect_errors.pdf" # Or get it from config/env
pickle_file_path = "pdf_embeddings_cache.pkl"

chunk_size = 1000
chunk_overlap = 150 # Overlap helps maintain context between chunks

openai_api_key = os.getenv("OPENAI_API_KEY")

# --------------- LOGIC ---------------
# --------------- PDF Loading Logic ---------------
def load_pdf_text(file_path):
    """Loads text content from a PDF file."""
    text = ""
    try:
        reader = pypdf.PdfReader(file_path)
        print(f"Loading PDF: {file_path}")
        print(f"Number of pages: {len(reader.pages)}")
        for page_num, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text: # Ensure text was extracted
                    text += page_text + "\n" # Add newline as page separator
                else:
                    print(f"Warning: No text extracted from page {page_num + 1}")
            except Exception as e:
                print(f"Error extracting text from page {page_num + 1}: {e}")
        print("PDF loaded successfully.")
        return text
    except FileNotFoundError:
        print(f"Error: PDF file not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return None
    
# --------------- Chunking Logic ---------------
def chunk_text(text, size, overlap):
    """Splits text into chunks using RecursiveCharacterTextSplitter."""
    if not text:
        print("Error: No text provided for chunking.")
        return []
    print(f"Chunking text with chunk_size={size}, chunk_overlap={overlap}...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        length_function=len,
        add_start_index=True, # Useful for potential source tracking later
    )
    # Langchain's split_text returns Document objects by default if input is a single string.
    # To get just the text chunks (strings), we use create_documents which returns Document objs, 
    # then extract page_content. Or simply use text_splitter.split_text directly.
    chunks = text_splitter.split_text(text)
    print(f"Successfully split text into {len(chunks)} chunks.")
    return chunks

# --------------- Embedding Logic ---------------
def generate_embeddings(chunks, model):
    """Generates embeddings for a list of text chunks using OpenAI."""
    if not chunks:
        print("No chunks provided for embedding.")
        return []

    print(f'\n\n---------------------------- Embedding ----------------------------')
    print(f"Generating embeddings for {len(chunks)} chunks using model '{model}'...")
    embeddings = []
    # Consider adding batching for efficiency if you have many chunks
    for i, chunk in enumerate(chunks):
        try:
            response = client.embeddings.create(input=chunk, model=model)
            embeddings.append(response.data[0].embedding)
            print(f"Generated embedding for chunk {i+1}/{len(chunks)}")
            # Add a small delay to avoid hitting rate limits aggressively
            time.sleep(0.1)
        except Exception as e:
            print(f"Error generating embedding for chunk {i+1}: {e}")
            # Decide how to handle errors: skip chunk, retry, stop?
            embeddings.append(None) # Placeholder for failed embedding
    print("Embedding generation complete.")
    return embeddings

# --------------- Pickling ---------------
def save_data_to_pickle(data, file_path):
    """Saves the processed data (chunks and embeddings) to a pickle file."""
    if not data:
        print("No processed data to save.")
        return False

    print(f"\nAttempting to save processed data to: {file_path}")
    try:
        with open(file_path, 'wb') as f: # Open in binary write mode
            pickle.dump(data, f)
        print(f"Successfully saved {len(data)} items to {file_path}")
        return True
    except Exception as e:
        print(f"Error saving data to pickle file: {e}")
        return False



# --------------- Execution ---------------
if not os.path.exists(pdf_path):
   print(f"Error: The file '{pdf_path}' does not exist. Please check the path.")
   # Exit or handle this error appropriately
   exit()

full_pdf_text = load_pdf_text(pdf_path)

if full_pdf_text:
    print(f'\n\n---------------------------- PDF Load ----------------------------')
    print(f"Successfully extracted {len(full_pdf_text)} characters from the PDF.")

if full_pdf_text: # Only chunk if text extraction was successful
    text_chunks = chunk_text(full_pdf_text, chunk_size, chunk_overlap)
    if text_chunks:
        print(f'\n\n---------------------------- Chunking ----------------------------')
        print(f"Example chunk (first 100 chars): {text_chunks[0][:100]}...")
        # You now have a list of text chunks in the 'text_chunks' variable
    else:
        print("Chunking resulted in no chunks.")


embedding_model = "text-embedding-3-small"
client = OpenAI()
chunk_embeddings = []
if 'text_chunks' in locals() and text_chunks: # Check if chunking was successful
    chunk_embeddings = generate_embeddings(text_chunks, embedding_model)
    if chunk_embeddings:
       # Filter out potential None values if errors occurred
       valid_embeddings = [emb for emb in chunk_embeddings if emb is not None]
       print(f"Successfully generated {len(valid_embeddings)} embeddings.")
       # print(f"Dimension of first embedding: {len(chunk_embeddings[0])}")
    else:
       print("Embedding generation failed or yielded no results.")

processed_data = []
if 'text_chunks' in locals() and text_chunks and chunk_embeddings and len(text_chunks) == len(chunk_embeddings):
   for text, embedding in zip(text_chunks, chunk_embeddings):
       if embedding: # Only include if embedding was successful
           processed_data.append({"text": text, "embedding": embedding})
   print(f"Created processed data structure with {len(processed_data)} items.")
else:
   print("Could not combine chunks and embeddings due to previous errors or mismatches.")

if 'processed_data' in locals() and processed_data:
    save_successful = save_data_to_pickle(processed_data, pickle_file_path)
    if save_successful:
        print("Data saved. You can load this next time instead of reprocessing the PDF.")
    else:
        print("Failed to save data to pickle file.")