# Import necessary libraries
# pypdf for reading PDF files
# SentenceSplitter from llama_index for text chunking
import pypdf
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
import os # Used for checking file existence

# --- Functions ---

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts text content from a given PDF file.

    Args:
        pdf_path: The file path to the PDF document.

    Returns:
        A string containing the extracted text from the PDF.
        Returns an empty string if the file doesn't exist or an error occurs.
    """
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        return ""

    text = ""
    try:
        # Open the PDF file
        with open(pdf_path, 'rb') as file:
            reader = pypdf.PdfReader(file)
            num_pages = len(reader.pages)
            print(f"Reading {num_pages} pages from {pdf_path}...")

            # Iterate through each page and extract text
            for page_num in range(num_pages):
                page = reader.pages[page_num]
                page_text = page.extract_text()
                if page_text: # Ensure text was extracted
                    text += page_text + "\n" # Add a newline between pages
            print("Finished extracting text.")
    except Exception as e:
        print(f"An error occurred while reading the PDF: {e}")
        return "" # Return empty string on error

    return text

def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Chunks the input text into smaller segments using LlamaIndex's SentenceSplitter.

    Args:
        text: The input text string to be chunked.
        chunk_size: The target size for each chunk.
        chunk_overlap: The overlap between consecutive chunks.

    Returns:
        A list of strings, where each string is a text chunk.
    """
    if not text:
        print("Input text is empty, skipping chunking.")
        return []

    print(f"Chunking text with chunk_size={chunk_size} and chunk_overlap={chunk_overlap}...")
    # Initialize the SentenceSplitter
    # You can also customize other parameters like paragraph_separator, chunking_regex, etc.
    text_splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # LlamaIndex's splitter often works best with Document objects
    # Create a single Document object containing all the text
    doc = Document(text=text)

    # Split the document into nodes (which contain text chunks)
    nodes = text_splitter.get_nodes_from_documents([doc])

    # Extract the text content from each node
    text_chunks = [node.get_content() for node in nodes]

    print(f"Successfully split text into {len(text_chunks)} chunks.")
    return text_chunks

# %%
# if __name__ == "__main__":
import re
# Define chunking parameters for SentenceSplitter
CHUNK_SIZE = 512  # The maximum size of each text chunk (in tokens/characters, depending on splitter)
CHUNK_OVERLAP = 50 # The number of tokens/characters to overlap between chunks

chunks_dict = {}
# regex = r"([^_]+).*?\.pdf"

# --- Main Execution ---
pth_import = 'data/litt_rev'
for PDF_FILE_PATH in os.listdir(pth_import):
    # match = re.search(regex, PDF_FILE_PATH)
    # if match:
    #     author_name = match.group(1).split()[0]
        # print(author_name )
        # print(PDF_FILE_PATH.split('.')[0] )
    # 1. Extract text from the PDF
    extracted_text = extract_text_from_pdf(os.path.join(pth_import,PDF_FILE_PATH))

    if extracted_text:
        # 2. Chunk the extracted text
        chunks = chunk_text(extracted_text, CHUNK_SIZE, CHUNK_OVERLAP)
        chunks_dict[PDF_FILE_PATH.split('.')[0]] = chunks

        # # 3. (Optional) Print the first few chunks to verify
        if chunks:
        #     print("\n--- Example Chunks ---")
        #     for i, chunk in enumerate(chunks[:3]): # Print the first 3 chunks
        #         print(f"Chunk {i+1}:")
        #         print(chunk)
        #         print("-" * 20)

            # You would typically pass these 'chunks' (or the 'nodes' directly)
            # to the next stage of your LlamaIndex pipeline, such as embedding
            # generation and indexing.
            print(f"\nTotal chunks generated: {len(chunks)}")
        else:
            print("No chunks were generated.")
    else:
        print("Text extraction failed. Please check the PDF file path and integrity.")

import pickle


with open(os.path.join(pth_import, 'dict_extracted_chunks_pdfs.pkl'), 'wb') as file:
    pickle.dump(chunks_dict, file)






















